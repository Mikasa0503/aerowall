"""Initialize and step upstream SingleJuggle with its launch-script configuration.

This smoke test is not the full reset/contact/isolation gate or a training result.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import sys
import traceback
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/JuggleRL_train'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-reset', action='store_true')
    parser.add_argument('--reset-safe-controller', action='store_true')
    parser.add_argument('--check-contacts', action='store_true')
    parser.add_argument('--check-ppo', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    if args.check_ppo and not args.reset_safe_controller:
        parser.error('PPO gate requires the verified reset-safe controller adapter')
    report = {'status': 'starting', 'gate': 'upstream_singlejuggle_smoke', 'pid': os.getpid(),
              'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'full_16_env_gate_passed': False, 'training_gate_passed': False}
    report['source_commit'] = subprocess.check_output(
        ['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    report['file_hashes'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in [Path(__file__), ROOT / 'scripts/python.sh',
                                        ROOT / 'scripts/check_upstream_reset.py',
                                        ROOT / 'scripts/runtime_adapters.py',
                                        ROOT / 'scripts/check_upstream_contacts.py',
                                        ROOT / 'scripts/check_upstream_ppo.py']}
    report['reset_safe_controller'] = args.reset_safe_controller
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(**values):
        report.update(values)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output)
        print(json.dumps(values), flush=True)

    app = None
    try:
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        source = (UPSTREAM / 'scripts/shell/singlejuggle_sim2real.sh').read_text()
        tokens = shlex.split(source.replace('\\\n', ' '), comments=True)
        overrides = tokens[tokens.index('../train.py') + 1:]
        replacements = {'task.env.num_envs': '16', 'wandb.mode': 'disabled', 'headless': 'true'}
        overrides = [item for item in overrides if item.split('=', 1)[0] not in replacements]
        overrides += [f'{key}={value}' for key, value in replacements.items()]
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(UPSTREAM / 'cfg')):
            cfg = compose(config_name='train', overrides=overrides)
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        config_path = args.output.with_suffix('.yaml')
        OmegaConf.save(cfg, config_path)
        sys.argv = [sys.argv[0], '--portable', '--portable-root', str(ROOT / '.cache/kit')]
        record(status='initializing_kit', config=str(config_path), num_envs=cfg.env.num_envs)
        app = init_simulation_app(cfg)
        from omni_drones.envs import IsaacEnv
        from omni_drones.controllers import PID_controller_flightmare
        from omni_drones.utils.torchrl.transforms import PIDRateController_flightmare
        from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
        if args.check_contacts:
            from check_upstream_contacts import contact_reporting_before_initialization
            with contact_reporting_before_initialization():
                base = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=True)
        else:
            base = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=True)
        controller = PID_controller_flightmare(cfg.sim.dt, base.drone.params, base.device).to(base.device)
        # Observe the controller output before upstream nan_to_num can hide errors.
        def finite_controller_output(module, inputs, output):
            assert torch.isfinite(output).all(), 'Non-finite raw controller output'
        controller.register_forward_hook(finite_controller_output)
        transform_type = PIDRateController_flightmare
        if args.reset_safe_controller:
            from runtime_adapters import ResetSafePIDRateController
            transform_type = ResetSafePIDRateController
        env = TransformedEnv(base, Compose(InitTracker(), transform_type(controller))).train()
        env.set_seed(cfg.seed)
        with torch.no_grad():
            td = env.reset()
        resets = 0
        record(status='stepping', observation_shape=list(td['agents', 'observation'].shape))
        for step in range(args.steps):
            # Seeded diagnostic excitation through the unmodified CTBR transform.
            action = torch.randn(16, 1, 4, device=base.device) * 0.15
            action[..., 3] += 0.32
            td.set(('agents', 'action'), action)
            with torch.no_grad():
                nxt = env.step(td)['next']
            for key, value in nxt.items(True, True):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    assert torch.isfinite(value).all(), f'Non-finite {key} at step {step}'
            assert torch.isfinite(base.ball.get_velocities()).all()
            done = nxt['done']
            if done.any():
                resets += int(done.sum().item())
                nxt.set('_reset', done)
                with torch.no_grad():
                    td = env.reset(nxt)
            else:
                td = nxt
            if (step + 1) % 100 == 0:
                record(completed_steps=step + 1, reset_episodes=resets)
        if args.check_reset:
            from check_upstream_reset import check_selective_reset, scene_inventory
            with torch.no_grad():
                checks = check_selective_reset(env, base, controller, td)
            record(reset_checks=checks, scene_inventory=scene_inventory())
            assert checks['physics_reset_passed'], 'Selective physics reset failed'
            assert checks['controller_reset_passed'], 'Controller retained prior episode state'
        if args.check_contacts:
            from check_upstream_contacts import check_contacts
            with torch.no_grad():
                contact_checks = check_contacts(env, base, record)
            assert contact_checks['passed'], 'Controlled contact/isolation checks failed'
        if args.check_ppo:
            from check_upstream_ppo import check_ppo
            checkpoint = ROOT / 'checkpoints' / (args.output.stem + '.pt')
            checks = check_ppo(env, base, cfg, checkpoint, record)
            record(ppo_checks=checks, training_gate_passed=checks['passed'])
        record(status='passed', completed_steps=args.steps, reset_episodes=resets,
               scope='Only explicitly recorded checks are claimed; no task performance or full project completion claim')
        return 0
    except Exception as error:
        record(status='failed', error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
