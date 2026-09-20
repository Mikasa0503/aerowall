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

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/JuggleRL_train'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    report = {'status': 'starting', 'gate': 'upstream_singlejuggle_smoke', 'pid': os.getpid(),
              'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'full_16_env_gate_passed': False, 'training_gate_passed': False}
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
        base = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=True)
        controller = PID_controller_flightmare(cfg.sim.dt, base.drone.params, base.device).to(base.device)
        # Observe the controller output before upstream nan_to_num can hide errors.
        def finite_controller_output(module, inputs, output):
            assert torch.isfinite(output).all(), 'Non-finite raw controller output'
        controller.register_forward_hook(finite_controller_output)
        env = TransformedEnv(base, Compose(InitTracker(), PIDRateController_flightmare(controller))).train()
        env.set_seed(cfg.seed)
        td = env.reset()
        resets = 0
        record(status='stepping', observation_shape=list(td['agents', 'observation'].shape))
        for step in range(args.steps):
            # Seeded diagnostic excitation through the unmodified CTBR transform.
            action = torch.randn(16, 1, 4, device=base.device) * 0.15
            action[..., 3] += 0.32
            td.set(('agents', 'action'), action)
            nxt = env.step(td)['next']
            for key, value in nxt.items(True, True):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    assert torch.isfinite(value).all(), f'Non-finite {key} at step {step}'
            assert torch.isfinite(base.ball.get_velocities()).all()
            done = nxt['done']
            if done.any():
                resets += int(done.sum().item())
                nxt.set('_reset', done)
                td = env.reset(nxt)
            else:
                td = nxt
            if (step + 1) % 100 == 0:
                record(completed_steps=step + 1, reset_episodes=resets)
        record(status='passed', completed_steps=args.steps, reset_episodes=resets,
               scope='upstream initialization and finite stepping only; controlled contacts and isolation remain untested')
        return 0
    except Exception as error:
        record(status='failed', error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
