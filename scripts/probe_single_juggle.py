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
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-reset', action='store_true')
    parser.add_argument('--reset-safe-controller', action='store_true')
    parser.add_argument('--check-contacts', action='store_true')
    parser.add_argument('--check-reaction', action='store_true')
    parser.add_argument('--check-contact-geometry', action='store_true')
    parser.add_argument('--enable-body-collisions', action='store_true')
    parser.add_argument('--contact-pitch-rate', type=float, default=0.)
    parser.add_argument('--check-ppo', action='store_true')
    parser.add_argument('--check-render', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    if not 16 <= args.num_envs <= 512:
        parser.error('deployment probes currently support 16 to 512 environments')
    if args.check_ppo and not args.reset_safe_controller:
        parser.error('PPO gate requires the verified reset-safe controller adapter')
    if args.enable_body_collisions and not args.check_contact_geometry:
        parser.error('Body-collision override is currently restricted to geometry validation')
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
                                        ROOT / 'scripts/check_upstream_reaction.py',
                                        ROOT / 'scripts/check_contact_geometry.py',
                                        ROOT / 'scripts/contact_geometry.py',
                                        ROOT / 'scripts/check_upstream_ppo.py',
                                        ROOT / 'scripts/check_upstream_render.py']}
    report['reset_safe_controller'] = args.reset_safe_controller
    report['body_collisions_override'] = args.enable_body_collisions
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
        replacements = {'task.env.num_envs': str(args.num_envs), 'wandb.mode': 'disabled', 'headless': 'true'}
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
        if args.check_contacts or args.check_reaction or args.check_contact_geometry:
            from check_upstream_contacts import contact_reporting_before_initialization
            with contact_reporting_before_initialization(enable_body_collisions=args.enable_body_collisions):
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
            action = torch.randn(base.num_envs, 1, 4, device=base.device) * 0.15
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
        if args.check_reaction:
            from check_upstream_reaction import check_reaction
            with torch.no_grad():
                checks = check_reaction(env, base, record)
            assert checks['passed'], 'Ball-bat reaction momentum check failed'
        if args.check_contact_geometry:
            from check_contact_geometry import check_geometry
            with torch.no_grad():
                checks = check_geometry(env, base, record, pitch_rate=args.contact_pitch_rate)
            assert checks['passed'], 'Physical contact geometry checks failed'
        if args.check_ppo:
            from check_upstream_ppo import check_ppo
            checkpoint = ROOT / 'checkpoints' / (args.output.stem + '.pt')
            checks = check_ppo(env, base, cfg, checkpoint, record)
            record(ppo_checks=checks, training_gate_passed=checks['passed'])
        if args.check_render:
            from check_upstream_render import check_render
            with torch.no_grad():
                checks = check_render(env, base, ROOT / 'artifacts' / args.output.stem, record)
            record(render_checks=checks, rendering_tested=True)
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
