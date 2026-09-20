"""Fixed first-episode development evaluation and raw contact/state export.

Only the initial reset defines a scored scenario. Resetting finished slots keeps
PhysX healthy but never contributes further contacts to that scenario. Top-cap
classification is an explicitly provisional geometric diagnostic until tested
against dedicated side/underside fixtures; do not infer a formal success gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scenarios', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260921)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'pid': os.getpid(), 'scope': '100 first-episode development scenarios',
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'classifier_sha256': hashlib.sha256((ROOT/'scripts/contact_geometry.py').read_bytes()).hexdigest(),
              'formal_success_gate_validated': False}
    def record(**values):
        report.update(values)
        temporary = args.output.with_suffix('.tmp'); temporary.write_text(json.dumps(report, indent=2)+'\n'); temporary.replace(args.output)
        print(json.dumps(values), flush=True)
    app = None
    record()
    try:
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        cfg = OmegaConf.load(args.config)
        cfg.task.env.num_envs = 100; cfg.env.num_envs = 100
        cfg.headless = True; cfg.wandb.mode = 'disabled'; cfg.seed = args.seed
        sys.argv = [sys.argv[0], '--portable', '--portable-root', str(ROOT / '.cache/kit')]
        app = init_simulation_app(cfg)
        from omni_drones.envs import IsaacEnv
        from omni_drones.controllers import PID_controller_flightmare
        from omni_drones.learning import MAPPOPolicy
        from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse
        from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
        from omni.isaac.core.prims import RigidPrimView
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType, SETTING_DISABLE_CONTACT_PROCESSING
        from pxr import PhysicsSchemaTools, UsdGeom
        import omni.usd
        import carb.settings
        from runtime_adapters import ResetSafePIDRateController
        from check_upstream_contacts import contact_reporting_before_initialization
        from contact_geometry import classify_cylinder_cap
        with contact_reporting_before_initialization():
            base = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=True)
        controller = PID_controller_flightmare(cfg.sim.dt, base.drone.params, base.device).to(base.device)
        env = TransformedEnv(base, Compose(InitTracker(), ResetSafePIDRateController(controller))).eval()
        policy = MAPPOPolicy(cfg.algo, agent_spec=env.agent_spec['drone'], device=base.device)
        payload = torch.load(args.checkpoint, map_location=base.device)
        policy.load_state_dict(payload['policy']); policy.eval()
        bats = RigidPrimView('/World/envs/env_*/Air_0/bat', name='evaluation_bats', reset_xform_properties=False)
        bats.initialize()
        bat_order = torch.tensor(sorted(range(100), key=lambda row: int(re.search(r'/env_(\d+)/', bats.prim_paths[row]).group(1))), device=base.device)
        com_local = bats.get_coms()[0].reshape(100,3)[bat_order]
        stage = omni.usd.get_context().get_stage()
        cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath('/World/envs/env_0/Air_0/bat/collisions'))
        radius, height = cylinder.GetRadiusAttr().Get(), cylinder.GetHeightAttr().Get()
        assert cylinder.GetAxisAttr().Get() == 'Z'
        transform = UsdGeom.Xformable(cylinder).GetLocalTransformation()
        assert np.allclose(np.array(transform), np.eye(4)), 'Classifier requires identity collider-local transform'
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
        interface = get_physx_simulation_interface()
        env.set_seed(args.seed)
        with torch.no_grad():
            td = env.reset()
        def state():
            dp, dq = base.drone.get_world_poses(); bp, bq = bats.get_world_poses()
            return {'drone_position': dp[:, 0].clone(), 'drone_quaternion_wxyz': dq[:, 0].clone(),
                    'drone_velocity': base.drone.get_velocities()[:, 0].clone(),
                    'ball_position': base.ball.get_world_poses()[0][:, 0].clone(),
                    'ball_velocity': base.ball.get_velocities()[:, 0].clone(),
                    'bat_position': bp[bat_order].clone(), 'bat_quaternion_wxyz': bq[bat_order].clone(),
                    'bat_velocity': bats.get_velocities()[bat_order].clone()}
        initial = state()
        roster = {'seed': args.seed, 'count': 100, 'initial_state': {k:v.cpu().tolist() for k,v in initial.items()},
                  'env_origins': base.envs_positions.cpu().tolist(), 'ball_material_restitution':base.current_restitution.cpu().tolist(),
                  'config_task': OmegaConf.to_container(cfg.task, resolve=True)}
        if args.scenarios.exists():
            assert json.loads(args.scenarios.read_text()) == roster, 'Seeded reset differs from frozen scenario roster'
        else:
            args.scenarios.parent.mkdir(parents=True, exist_ok=True)
            args.scenarios.write_text(json.dumps(roster, indent=2)+'\n')
        record(status='evaluating', scenarios=str(args.scenarios), scenario_sha256=hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
               trained_environment_frames=payload['environment_frames'], collider={'radius':radius,'height':height,'axis':'Z'},
               scoring='first positive impulse >1e-6 N s per FOUND/LOST contact; raw entry counts remain diagnostic',
               top_classification='provisional: upper half, cap-normal alignment >=0.8, radial extent <= radius+0.01 m')
        finished = torch.zeros(100, dtype=torch.bool, device=base.device)
        ball_bat_entries = [0]*100; top_entries = [0]*100
        contact_active = set(); credited_contacts = set(); events = []; outcomes = [None]*100; trajectories = []
        event_path = args.output.with_suffix('.events.jsonl')
        with torch.no_grad(), event_path.open('w') as event_file:
            for step in range(base.max_episode_length):
                before = state()
                active_before = ~finished.clone()
                policy(td, deterministic=True)
                action = td['agents','action'].clone()
                nxt = env.step(td)['next']
                after = state()
                assert all(torch.isfinite(v).all() for v in after.values()), 'Non-finite physical state'
                normal = quat_rotate(after['bat_quaternion_wxyz'], torch.tensor([0.,0.,1.],device=base.device).expand(100,3))
                com = after['bat_position'] + quat_rotate(after['bat_quaternion_wxyz'], com_local)
                illegal = {}
                headers, points = interface.get_contact_report()
                for header in headers:
                    paths = {key:str(PhysicsSchemaTools.intToSdfPath(getattr(header,key))) for key in ['actor0','actor1','collider0','collider1']}
                    ids = set(int(match) for value in paths.values() for match in re.findall(r'/env_(\d+)/',value))
                    if not ids:
                        continue
                    assert len(ids) == 1, 'Cross-environment contact in evaluation'
                    index = next(iter(ids)); key = tuple(sorted([paths['collider0'],paths['collider1']]))
                    if header.type == ContactEventType.CONTACT_LOST:
                        contact_active.discard(key)
                        credited_contacts.discard(key)
                    if not bool(active_before[index]):
                        continue
                    new_entry = header.type == ContactEventType.CONTACT_FOUND and key not in contact_active
                    if header.type == ContactEventType.CONTACT_FOUND:
                        contact_active.add(key)
                    event = {'step':step,'time':(step+1)*base.dt,'scenario_id':index,'type':str(header.type),'new_entry':new_entry,**paths}
                    actors = [paths['actor0'],paths['actor1']]
                    ball = f'/World/envs/env_{index}/ball'; bat = f'/World/envs/env_{index}/Air_0/bat'
                    ball_bat = set(actors) == {ball,bat}
                    top = False
                    impulse_norm = 0.0
                    if header.num_contact_data:
                        contact_rows = [points[header.contact_data_offset+j] for j in range(header.num_contact_data)]
                        impulse_norm = sum(float(torch.linalg.norm(torch.tensor(list(row.impulse))).item()) for row in contact_rows)
                        point = contact_rows[0]
                        point_world = torch.tensor(list(point.position),device=base.device)
                        direction = torch.tensor(list(point.normal),device=base.device)
                        top,local,axis,alignment = classify_cylinder_cap(point_world,direction,after['bat_position'][index],after['bat_quaternion_wxyz'][index],radius)
                        point_velocity = after['bat_velocity'][index,:3] + torch.cross(after['bat_velocity'][index,3:],point_world-com[index],dim=0)
                        event.update(position=point_world.cpu().tolist(),normal=direction.cpu().tolist(),bat_local_point=local.cpu().tolist(),
                                     bat_normal=normal[index].cpu().tolist(),bat_contact_point_velocity=point_velocity.cpu().tolist(),
                                     ball_velocity_before=before['ball_velocity'][index].cpu().tolist(),ball_velocity_after=after['ball_velocity'][index].cpu().tolist(),
                                     cap_normal_alignment=alignment)
                    if new_entry and ball_bat:
                        ball_bat_entries[index] += 1
                    # A contact margin may report FOUND before an actual impulse.
                    # Credit the first positive top-cap impulse within FOUND/LOST.
                    credited_now = ball_bat and top and impulse_norm > 1e-6 and key not in credited_contacts
                    if credited_now:
                        top_entries[index] += 1
                        credited_contacts.add(key)
                    event.update(impulse_norm=impulse_norm,credited_top_impact=credited_now)
                    event['provisional_top_contact'] = ball_bat and top
                    if impulse_norm > 1e-6 and ball_bat and not top:
                        illegal[index] = 'non_cap_bat_impact'
                    if impulse_norm > 1e-6 and not ball_bat:
                        if ball in actors: illegal[index] = 'ball_contact_outside_bat'
                        elif any('/Air_0/' in p for p in actors) and not all('/Air_0/' in p for p in actors):
                            illegal[index] = 'drone_external_contact'
                    event_file.write(json.dumps(event)+'\n'); events.append(event)
                done = nxt['done'].flatten()
                for index in range(100):
                    if bool(active_before[index]) and (bool(done[index]) or index in illegal or step+1 == base.max_episode_length):
                        reason = illegal.get(index,'upstream_termination' if bool(done[index]) else 'time_limit')
                        outcomes[index] = {'scenario_id':index,'steps':step+1,'reason':reason,
                                           'ball_bat_entries':ball_bat_entries[index],'provisional_top_entries':top_entries[index],
                                           'upstream_stats':{k:float(nxt['stats',k][index].flatten()[0]) for k in ['num_true_hits','wrong_hit','ball_too_low','ball_too_high','ball_too_far','drone_too_low','drone_too_high','truncated']}}
                        finished[index] = True
                trajectories.append({**{k:v.cpu().numpy() for k,v in after.items()},'action':action[:,0].cpu().numpy(),
                                     'active_before':active_before.cpu().numpy(),'bat_normal':normal.cpu().numpy()})
                if finished.all(): break
                if done.any():
                    nxt.set('_reset',nxt['done']); td = env.reset(nxt)
                    reset_ids = set(done.nonzero().flatten().cpu().tolist())
                    contact_active = {pair for pair in contact_active if not any(int(i) in reset_ids for path in pair for i in re.findall(r'/env_(\d+)/',path))}
                    credited_contacts.intersection_update(contact_active)
                else: td = nxt
                if (step+1)%100 == 0: record(completed_steps=step+1,completed_scenarios=int(finished.sum()))
        trajectory_path = args.output.with_suffix('.trajectory.npz')
        np.savez_compressed(trajectory_path, **{k:np.stack([row[k] for row in trajectories]) for k in trajectories[0]}, dt=base.dt)
        args.output.with_suffix('.outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        record(status='passed', completed_scenarios=int(finished.sum()), steps=len(trajectories), outcomes=outcomes,
               ball_bat_entry_distribution=ball_bat_entries, provisional_top_entry_distribution=top_entries,
               provisional_five_top_contact_rate=sum(v>=5 for v in top_entries)/100,
               raw_events=str(event_path),trajectory=str(trajectory_path),
               trajectory_sha256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
               note='Infrastructure evaluation completed; provisional classifier and upstream episode boundaries do not establish full task success')
        return 0
    except Exception as error:
        record(status='failed',error=repr(error),traceback=traceback.format_exc()); return 1
    finally:
        if app is not None: app.close()


if __name__ == '__main__':
    raise SystemExit(main())
