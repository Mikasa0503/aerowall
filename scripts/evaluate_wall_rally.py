"""Deterministic development evaluation from a frozen serving-state roster.

Only the first episode in each slot is scored. Subsequent resets keep physics
healthy but cannot contribute to its outcome. This is not the formal 60 s suite.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['output', 'checkpoint', 'config', 'scenarios']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--initialize-juggle', action='store_true', help='Evaluate the original actor before any WallRally updates')
    a = p.parse_args()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {'status': 'initializing', 'pid': os.getpid(), 'scope': 'frozen development serving episodes; not formal evaluation',
              'checkpoint': str(a.checkpoint), 'checkpoint_sha256': sha(a.checkpoint),
              'scenario_sha256': sha(a.scenarios), 'config_sha256': sha(a.config),
              'source_hashes': {name: sha(ROOT/name) for name in ['scripts/evaluate_wall_rally.py', 'aerowall/envs/wall_rally.py',
                  'aerowall/envs/aligned_juggle.py', 'aerowall/contact_router.py', 'aerowall/collider_bounds.py',
                  'aerowall/rally_events.py', 'aerowall/learning/wall_policy.py', 'scripts/runtime_adapters.py']}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def record(**values):
        report.update(values)
        tmp = a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(a.output)
        print(json.dumps(values), flush=True)
    app = None
    record()
    try:
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        roster = json.loads(a.scenarios.read_text()); n = roster['count']
        assert n == 100, 'Development roster must contain 100 scenarios'
        cfg = OmegaConf.load(a.config); OmegaConf.set_struct(cfg, False)
        cfg.env.num_envs = cfg.task.env.num_envs = n
        cfg.headless = True; cfg.wandb.mode = 'disabled'; cfg.record_contact_kinematics = True
        sys.argv = [sys.argv[0], '--portable', '--portable-root', str(ROOT/'.cache/kit')]
        app = init_simulation_app(cfg)
        from aerowall.envs.wall_rally import WallRally
        from aerowall.learning.wall_policy import WallMAPPOPolicy
        from check_upstream_contacts import contact_reporting_before_initialization
        from runtime_adapters import ResetSafePIDRateController
        from omni_drones.controllers import PID_controller_flightmare
        from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        import carb.settings
        with contact_reporting_before_initialization(enable_body_collisions=True):
            base = WallRally(cfg, headless=True)
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
        controller = PID_controller_flightmare(.02, base.drone.params, base.device).to(base.device)
        env = TransformedEnv(base, Compose(InitTracker(), ResetSafePIDRateController(controller))).eval()
        policy = WallMAPPOPolicy(cfg.algo, agent_spec=env.agent_spec['drone'], device=base.device)
        payload = torch.load(a.checkpoint, map_location=base.device)
        if a.initialize_juggle:
            policy.initialize_juggle_actor(payload)
            record(actor_transfer_audit=policy.audit_juggle_actor(payload), wall_training_updates=0)
        else:
            policy.load_state_dict(payload['policy'])
        policy.eval()
        env.set_seed(roster['seed'])
        with torch.no_grad(): td = env.reset()
        def snapshot():
            dp, dq = base.drone.get_world_poses()
            return {'drone_position': dp[:,0].clone(), 'drone_quaternion_wxyz': dq[:,0].clone(),
                    'drone_velocity': base.drone.get_velocities()[:,0].clone(),
                    'ball_position': base.ball.get_world_poses()[0][:,0].clone(),
                    'ball_velocity': base.ball.get_velocities()[:,0].clone()}
        initial = snapshot()
        delta = {k: float(np.max(np.abs(v.cpu().numpy()-np.array(roster['initial_state'][k])))) for k,v in initial.items()}
        assert all(v <= 1e-6 for v in delta.values()), delta
        assert np.allclose(base.envs_positions.cpu().numpy(), roster['env_origins'], atol=1e-6, rtol=0)
        assert np.allclose(base.current_restitution.cpu().numpy(), roster['ball_material_restitution'], atol=1e-6, rtol=0)
        record(status='evaluating', initial_state_differences=delta, trained_frames=payload['environment_frames'],
               physics_dt=float(cfg.sim.dt), policy_dt=.02, episode_seconds=base.max_episode_length*.02,
               wall_fixture=OmegaConf.to_container(cfg.wall_fixture), wall_task=OmegaConf.to_container(cfg.wall_task))
        finished = torch.zeros(n, dtype=torch.bool, device=base.device)
        outcomes = [None]*n; trajectories = []; start = time.monotonic()
        with torch.no_grad(), a.output.with_suffix('.events.jsonl').open('w') as events:
            for step in range(base.max_episode_length):
                active = ~finished.clone()
                policy(td, deterministic=True); action = td['agents','action'].clone()
                nxt = env.step(td)['next']; state = snapshot()
                assert all(torch.isfinite(v).all() for v in state.values())
                for event in base.last_events:
                    if bool(active[event['env_id']]):
                        events.write(json.dumps({**event, 'policy_step': step,
                            'time': step*.02+event.get('physics_time_offset', .02)})+'\n')
                done = nxt['done'].flatten()
                for i in (active & done).nonzero().flatten().cpu().tolist():
                    s = base.router.batch.states[i]
                    outcomes[i] = {'scenario_id': i, 'policy_steps': step+1, **vars(s),
                                   'physics_steps': s.steps,
                                   'legal_caps': int(base.stats['num_true_hits'][i])}
                    finished[i] = True
                trajectories.append({**{k:v.cpu().numpy() for k,v in state.items()},
                    'action':action[:,0].cpu().numpy(), 'active_before':active.cpu().numpy(),
                    'target':base.targets.cpu().numpy().copy()})
                base.completed_episodes.clear()
                if (step+1) % 100 == 0:
                    record(completed_policy_steps=step+1, completed_scenarios=int(finished.sum()))
                if finished.all(): break
                if done.any():
                    nxt['_reset'] = nxt['done']; td = env.reset(nxt)
                else: td = nxt
            assert finished.all(), 'Every first episode must reach a declared terminal or truncation'
        assert base.contact_totals['policy_steps'] == len(trajectories)
        assert base.contact_totals['physics_steps'] == len(trajectories)*base.substeps
        path = a.output.with_suffix('.trajectory.npz')
        np.savez_compressed(path, **{k:np.stack([r[k] for r in trajectories]) for k in trajectories[0]}, dt=.02, physics_dt=float(cfg.sim.dt))
        reasons = {}
        for row in outcomes: reasons[row['reason']] = reasons.get(row['reason'], 0)+1
        record(status='passed', outcomes=outcomes, completed_scenarios=n, elapsed_seconds=time.monotonic()-start,
               one_rally_rate=sum(r['rallies']>=1 for r in outcomes)/n,
               ten_consecutive_rally_rate=sum(r['max_streak']>=10 for r in outcomes)/n,
               reasons=reasons, trajectory=str(path), trajectory_sha256=sha(path),
               note='Only first episodes scored; no stitched resets; development duration only')
        return 0
    except Exception as e:
        record(status='failed', error=repr(e), traceback=traceback.format_exc()); return 1
    finally:
        if app is not None: app.close()


if __name__ == '__main__':
    raise SystemExit(main())
