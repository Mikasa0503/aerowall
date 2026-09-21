"""Development WallRally training with upstream PPO/CTBR and durable evidence.

Counts all collected transitions, including a resumed run's prior training.
Resume restores learning/RNG state but starts fresh simulator episodes; it does
not claim bitwise continuation. Upstream hit statistics are diagnostic proxies.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM = ROOT / 'third_party/JuggleRL_train'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--recovery-skill',action='store_true',help='Freeze launch actor; train post-launch recovery actor only')
    parser.add_argument('--initialize-policy', type=Path, help='Transfer weights only; prior frames remain in the budget')
    parser.add_argument('--physics-dt',type=float,default=.0025)
    parser.add_argument('--wall-config',type=Path,default=ROOT/'configs/wall_single_return.yaml')
    args = parser.parse_args()
    args.wall_config=args.wall_config.resolve()
    assert args.physics_dt in (.02,.01,.005,.0025,.00125)
    if min(args.updates, args.save_every) < 1 or not 16 <= args.num_envs <= 512:
        parser.error('Require positive update/save counts and 16–512 environments')
    assert not (args.resume and args.initialize_policy), 'Choose resume or transfer, not both'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'pid': os.getpid(), 'phase': 'development_wall_rally',
              'seed': args.seed, 'updates_requested_this_run': args.updates, 'performance_claim': False,
              'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / 'scripts/runtime_adapters.py', ROOT / 'scripts/source_evidence.py', ROOT / 'scripts/python.sh', ROOT / 'aerowall/envs/aligned_juggle.py', ROOT / 'aerowall/contact_router.py', ROOT / 'aerowall/rally_events.py', ROOT / 'aerowall/envs/wall_rally.py', ROOT / 'aerowall/learning/wall_policy.py', ROOT / 'aerowall/learning/phase_recovery.py', ROOT / 'aerowall/learning/recovery_skill_policy.py', ROOT / 'aerowall/collider_bounds.py', args.wall_config]}}
    def record(**values):
        report.update(values)
        tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(report, indent=2) + '\n'); tmp.replace(args.output)
        print(json.dumps(values), flush=True)
    app = None
    record()
    try:
        from source_evidence import snapshot_sources, verify_sources
        record(source_snapshot=snapshot_sources(ROOT, report['source_hashes'], args.output.with_suffix('.sources')))
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        tokens = shlex.split((UPSTREAM / 'scripts/shell/singlejuggle_sim2real.sh').read_text().replace('\\\n', ' '), comments=True)
        overrides = tokens[tokens.index('../train.py') + 1:]
        replacements = {'task.env.num_envs': str(args.num_envs), 'wandb.mode': 'disabled', 'headless': 'true', 'seed': str(args.seed)}
        overrides = [v for v in overrides if v.split('=', 1)[0] not in replacements]
        overrides += [f'{k}={v}' for k, v in replacements.items()]
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(UPSTREAM / 'cfg')):
            cfg = compose(config_name='train', overrides=overrides)
        OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
        cfg.sim.dt=args.physics_dt;cfg.sim.substeps=round(.02/args.physics_dt)
        wall_cfg=OmegaConf.load(args.wall_config)
        cfg.wall_fixture=wall_cfg.wall_fixture;cfg.wall_task=wall_cfg.wall_task
        # This run's actual budget overrides the author shell's two-billion cap.
        frames_per_batch = args.num_envs * int(cfg.algo.train_every)
        cfg.total_frames = args.updates * frames_per_batch
        config_text = OmegaConf.to_yaml(cfg)
        args.output.with_suffix('.yaml').write_text(config_text)
        learning_config = OmegaConf.to_container(cfg, resolve=True)
        for key in ['total_frames', 'save_interval', 'eval_interval', 'max_iters']:
            learning_config.pop(key, None)
        config_hash = hashlib.sha256(json.dumps(learning_config, sort_keys=True).encode()).hexdigest()
        sys.argv = [sys.argv[0], '--portable', '--portable-root', str(ROOT / '.cache/kit')]
        app = init_simulation_app(cfg)
        from omni_drones.envs import IsaacEnv
        from omni_drones.controllers import PID_controller_flightmare
        from omni_drones.learning import MAPPOPolicy
        from omni_drones.utils.torchrl import SyncDataCollector
        from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
        from runtime_adapters import ResetSafePIDRateController
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        from check_upstream_contacts import contact_reporting_before_initialization
        from aerowall.envs.wall_rally import WallRally
        from aerowall.learning.wall_policy import WallMAPPOPolicy
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        import carb.settings
        with contact_reporting_before_initialization(enable_body_collisions=True):
            base = WallRally(cfg, headless=True)
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
        controller = PID_controller_flightmare(.02, base.drone.params, base.device).to(base.device)
        controller_calls = 0
        def finite_output(module, inputs, output):
            nonlocal controller_calls
            controller_calls += 1
            assert torch.isfinite(output).all(), 'Non-finite controller output before upstream cleanup'
        controller.register_forward_hook(finite_output)
        env = TransformedEnv(base, Compose(InitTracker(), ResetSafePIDRateController(controller))).train()
        env.set_seed(args.seed)
        policy_class=WallMAPPOPolicy
        if args.recovery_skill:
            from aerowall.learning.recovery_skill_policy import RecoverySkillPolicy
            policy_class=RecoverySkillPolicy
            assert args.initialize_policy or args.resume, 'Recovery skill needs a frozen launch source'
        policy = policy_class(cfg.algo, agent_spec=env.agent_spec['drone'], device=base.device)
        record(policy_mode='frozen_launch_learned_recovery' if args.recovery_skill else 'single_wall_actor')
        prior_frames = 0
        if args.initialize_policy:
            loaded = torch.load(args.initialize_policy, map_location=base.device)
            source_width=loaded['policy']['actor_params']['module','encoder','0','weight'].shape[-1]
            assert source_width in (24,43), 'Only known Juggle or Wall actor layouts are supported'
            if source_width==24:
                policy.initialize_juggle_actor(loaded)
                audit=policy.audit_juggle_actor(loaded)
            else:
                assert 'aerowall/learning/wall_policy.py' in loaded['source_hashes'], 'Wall transfer requires recorded Wall policy source'
                policy.initialize_wall_actor(loaded)
                audit=policy.audit_wall_actor(loaded)
            record(actor_transfer_audit=audit,source_actor_width=source_width,prior_checkpoint_updates=loaded.get('n_updates'))
            if args.recovery_skill:policy.initialize_launch_actor(loaded)
            prior_frames = loaded['environment_frames']
            record(initialize_policy=str(args.initialize_policy), initialize_policy_sha256=hashlib.sha256(args.initialize_policy.read_bytes()).hexdigest(), transfer_semantics='Actor-only initialization: 24-feature Juggle actor extended or compatible 43-feature Wall actor copied; new critic, value normalizer, optimizers; prior frames counted; update counter starts fresh')
        if args.resume:
            loaded = torch.load(args.resume, map_location=base.device)
            assert ('frozen_launch_params' in loaded['policy'])==args.recovery_skill, 'Resume changes skill routing'
            assert loaded['learning_config_hash'] == config_hash, 'Resume changes the learning configuration'
            policy.load_state_dict(loaded['policy'])
            policy.actor_opt.load_state_dict(loaded['actor_optimizer']); policy.critic_opt.load_state_dict(loaded['critic_optimizer'])
            policy.n_updates = loaded['n_updates']; prior_frames = loaded['environment_frames']
            random.setstate(loaded['python_rng']); np.random.set_state(loaded['numpy_rng'])
            torch.set_rng_state(loaded['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in loaded['cuda_rng']])
            record(resume=str(args.resume), resume_semantics='learning/RNG state restored; new physics episodes')
        verify_sources(ROOT, report['source_hashes'])
        record(initialization_source_consistency=True)
        collector = SyncDataCollector(env, policy=policy, frames_per_batch=frames_per_batch,
                                      total_frames=(args.updates + 1) * frames_per_batch,
                                      device=base.device, return_same_td=True)
        checkpoint_dir = ROOT / 'checkpoints' / args.output.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        metric_path = args.output.with_suffix('.metrics.jsonl')
        started = time.monotonic()
        record(status='training', physics_dt=args.physics_dt, policy_dt=.02, physics_substeps=base.substeps, num_envs=base.num_envs, frames_per_batch=frames_per_batch,
               prior_environment_frames=prior_frames, learning_config_hash=config_hash,
               budget_this_run=args.updates * frames_per_batch, metrics=str(metric_path), checkpoints=str(checkpoint_dir))
        for index, data in enumerate(collector):
            assert not any(v.requires_grad for _, v in data.items(True, True) if isinstance(v, torch.Tensor)), 'Collector retained autograd state'
            for key in [('next', 'agents', 'observation'), ('next', 'agents', 'reward')]:
                assert torch.isfinite(data[key]).all(), f'Non-finite collector {key}'
            if args.recovery_skill:
                record(skill_rollout_audit=policy.audit_rollout(data.to_tensordict()))
                if index==0:
                    empty=data[:,0].to_tensordict().clone();empty[policy.mask_key].zero_()
                    before=policy.actor_params.clone()
                    assert policy.update_actor(empty)['actor_update_skipped']==1.
                    assert all(torch.equal(v,policy.actor_params[k]) for k,v in before.items(True,True))
                    if not args.resume:assert not policy.actor_opt.state, 'Empty actor update created optimizer state'
                    record(empty_recovery_actor_update_skipped=True)
            info = {k: float(v) for k, v in policy.train_op(data.to_tensordict()).items()}
            if args.recovery_skill:
                frozen_source=loaded['policy']['frozen_launch_params'] if args.resume else loaded['policy']['actor_params']
                assert all(torch.equal(v,frozen_source[k]) for k,v in policy.frozen_launch_params.items(True,True)), 'Frozen launch actor changed'

            assert all(math.isfinite(v) for v in info.values()), 'Non-finite learning metric'
            done = data['next', 'done'].squeeze(-1)
            episodes = int(done.sum().item())
            stats = {}
            if episodes:
                for key in ['return', 'episode_len', 'num_true_hits', 'wrong_hit', 'ball_too_low', 'drone_too_low']:
                    value = data['next', 'stats', key][done]
                    stats[key] = float(value.mean().item())
            assert controller_calls == base.contact_totals['policy_steps']
            assert base.contact_totals['physics_steps'] == controller_calls * base.substeps
            completed = index + 1
            frames = prior_frames + int(collector._frames)
            row = {'update_this_run': completed, 'total_updates': policy.n_updates,
                   'environment_frames': frames, 'completed_episodes': episodes,
                   'legacy_collector_stats_unvalidated': stats,
                   'terminal_episodes': list(base.completed_episodes), 'metrics': info, 'launch_stats': dict(base.launch_stats), 'recovery_stats':dict(base.recovery_stats),
                   'elapsed_seconds': time.monotonic() - started, 'rollout_fps': collector._fps, 'controller_calls': controller_calls, 'contact_totals': dict(base.contact_totals), 'wall_totals':dict(base.wall_totals), 'reset_boundary_failures':list(base.reset_boundary_failures)}
            with metric_path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            base.completed_episodes.clear()
            if completed == 1 or completed % args.save_every == 0 or completed == args.updates:
                path = checkpoint_dir / f'frames-{frames:012d}.pt'
                payload = {'policy': policy.state_dict(), 'actor_optimizer': policy.actor_opt.state_dict(),
                           'critic_optimizer': policy.critic_opt.state_dict(), 'n_updates': policy.n_updates,
                           'environment_frames': frames, 'learning_config_hash': config_hash,
                           'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                           'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                           'source_hashes': report['source_hashes'], 'upstream_commit': report['upstream_commit'],
                           'scope': 'development training; contact-based success evaluation required'}
                temp = path.with_suffix('.tmp'); torch.save(payload, temp); temp.replace(path)
                record(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if completed == 1 or completed % 10 == 0 or completed == args.updates:
                record(progress=row)
            if completed == args.updates:
                break
        if args.recovery_skill:
            restored=policy_class(cfg.algo,agent_spec=env.agent_spec['drone'],device=base.device)
            restored.load_state_dict(torch.load(path,map_location=base.device)['policy'])
            sample=data[:,0].to_tensordict()
            with torch.no_grad():
                original=policy(sample.clone(),deterministic=True)
                replay=restored(sample.clone(),deterministic=True)
            assert torch.equal(original[policy.act_name],replay[policy.act_name])
            assert torch.equal(original[policy.mask_key],replay[policy.mask_key])
            record(skill_checkpoint_actions_exact=True,frozen_launch_unchanged=True)
        record(status='passed', budget_completed=True, environment_frames=frames,
               frames_this_run=int(collector._frames), elapsed_training_seconds=time.monotonic()-started,
               scope='Training budget completed; no learned wall-return success claim')
        return 0
    except Exception as error:
        record(status='failed', error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
