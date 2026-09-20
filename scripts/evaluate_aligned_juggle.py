"""Frozen-scenario juggling evaluation with body collisions and aligned bat geometry.

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
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scenarios', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--lazy-contacts',action='store_true',help='Read GPU buffers only when a current pair header lacks valid CPU points')
    parser.add_argument('--contact-task', action='store_true', help='Use actual-contact curriculum boundaries, removing legacy hit cooldown termination')
    parser.add_argument('--physics-dt',type=float,default=.02)
    args = parser.parse_args()
    assert args.physics_dt in (.02,.01,.005,.0025,.00125)
    assert args.contact_task or args.physics_dt == .02
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'pid': os.getpid(), 'scope': '100 frozen development scenarios; original checkpoint on body-enabled aligned WallContactScene',
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'classifier_sha256': hashlib.sha256((ROOT/'scripts/contact_geometry.py').read_bytes()).hexdigest(),
              'formal_success_gate_validated': False,
              'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['aerowall/envs/aligned_juggle.py','aerowall/contact_router.py','aerowall/rally_events.py','scripts/runtime_adapters.py']}}
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
        cfg.wall_fixture={'center':[2.5,0.,3.],'dimensions':[.2,4.,6.],'restitution':.8,'align_cap_to_visual_top':True}
        cfg.sim.dt=args.physics_dt;cfg.sim.substeps=round(.02/args.physics_dt)
        cfg.record_contact_kinematics=args.contact_task
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
        from aerowall.envs.wall_scene import WallContactScene
        from aerowall.contact_router import WallContactRouter
        from aerowall.rally_events import Kind,ILLEGAL
        with contact_reporting_before_initialization(enable_body_collisions=True):
            if args.contact_task:
                from aerowall.envs.aligned_juggle import AlignedJuggle
                base = AlignedJuggle(cfg,headless=True)
            else:
                base = WallContactScene(cfg,headless=True)
        router=base.router if args.contact_task else WallContactRouter(base,eager_gpu=not args.lazy_contacts)
        record(physics_dt=args.physics_dt,policy_dt=.02,physics_substeps=cfg.sim.substeps,episode_rule='physical_contact_curriculum' if args.contact_task else 'legacy_upstream')
        record(contact_readback='lazy_current_pairs' if args.lazy_contacts else 'eager_all_pairs',bat_overlay=base.bat_overlay,body_collisions_enabled=True,wall_fixture=OmegaConf.to_container(cfg.wall_fixture,resolve=True),
               scope_note=('Physical-contact curriculum boundaries' if args.contact_task else 'Original SingleJuggle boundaries')+'; not WallRally policy evaluation')
        controller = PID_controller_flightmare(.02, base.drone.params, base.device).to(base.device)
        assert abs(float(controller.dt)-.02)<1e-9 and abs(base.drone.dt-args.physics_dt)<1e-9
        controller_calls = 0
        def count_controller(module, inputs, output):
            nonlocal controller_calls
            controller_calls += 1
            assert torch.isfinite(output).all()
        controller.register_forward_hook(count_controller)
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
        assert np.allclose(np.array(transform)[:3,:3],np.eye(3))
        collider_offset=torch.tensor(tuple(transform.ExtractTranslation()),device=base.device)
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
        actual_path=args.output.with_suffix('.initial-scenarios.json')
        actual_path.write_text(json.dumps(roster,indent=2)+'\n')
        expected=json.loads(args.scenarios.read_text())
        differences={}
        for key,value in initial.items():
            reference=np.array(expected['initial_state'][key]);actual=value.cpu().numpy()
            differences[key]=float(np.max(np.abs(reference-actual)))
        primary=['drone_position','drone_quaternion_wxyz','drone_velocity','ball_position','ball_velocity']
        other_keys=['seed','count','env_origins','ball_material_restitution','config_task']
        record(initial_state_max_absolute_differences=differences,
               initial_nonstate_matches={k:expected[k]==roster[k] for k in other_keys},
               actual_scenarios=str(actual_path),actual_scenario_sha256=hashlib.sha256(actual_path.read_bytes()).hexdigest(),
               comparison_scope='Frozen drone/ball initial state within 1e-6; derived bat-link state may change with the declared geometry/COM overlay')
        assert all(differences[k]<=1e-6 for k in primary),'Primary frozen drone/ball initial states differ'
        assert all(expected[k]==roster[k] for k in other_keys),'Frozen reset metadata/task configuration differs'
        record(status='evaluating', scenarios=str(args.scenarios), scenario_sha256=hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
               trained_environment_frames=payload['environment_frames'], collider={'radius':radius,'height':height,'axis':'Z'},
               scoring='first positive impulse >1e-6 N s per FOUND/LOST contact; raw entry counts remain diagnostic',
               top_classification='provisional: upper half, cap-normal alignment >=0.8, radial extent <= radius+0.01 m')
        finished = torch.zeros(100, dtype=torch.bool, device=base.device)
        ball_bat_entries = [0]*100; top_entries = [0]*100
        contact_active = set(); credited_contacts = set(); events = []; outcomes = [None]*100; trajectories = []
        event_path = args.output.with_suffix('.events.jsonl')
        evaluation_started=time.perf_counter();gpu_queries=0
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
                impacts,routed=(base.last_impacts,base.last_events) if args.contact_task else router.read()
                gpu_queries+=base.action_gpu_queries if args.contact_task else router.gpu_queries_this_step
                for index,rows in enumerate(impacts):
                    bad=sorted({row.kind.value for row in rows if row.kind in ILLEGAL})
                    if bad:illegal[index]=bad[0]
                if args.contact_task:
                    illegal={i:reason for i,reason in enumerate(base.action_failure_reason) if reason is not None}
                for event0 in routed:
                    index=event0['env_id']
                    if not bool(active_before[index]):continue
                    actors=list(event0['pair']);ball=f'/World/envs/env_{index}/ball';bat=f'/World/envs/env_{index}/Air_0/bat'
                    ball_bat=set(actors)=={ball,bat};new_entry=event0['edge']=='found'
                    points0=event0['points'];top=event0['kind']=='cap' and bool(points0)
                    credited_now=event0['curriculum_cap_credit'] if args.contact_task else top and event0['credited'] and index not in illegal
                    if ball_bat and new_entry:ball_bat_entries[index]+=1
                    if credited_now:top_entries[index]+=1
                    event={**event0,'step':step,'time':step*.02+event0.get('physics_time_offset',.02),'scenario_id':index,
                           'type':'ContactEventType.CONTACT_'+event0['edge'].upper(),
                           'actor0':actors[0],'actor1':actors[1],'new_entry':new_entry,
                           'impulse_norm':sum(p['impulse'] for p in points0),
                           'credited_top_impact':credited_now,'provisional_top_contact':top,
                           'same_step_illegal_priority':event0.get('physics_step_illegal_priority',index in illegal), 'policy_interval_has_illegal_contact':index in illegal, 'policy_step':step, 'physics_step':step*cfg.sim.substeps+event0.get('physics_substep',0)}
                    if points0:
                        point_world=torch.tensor(points0[0]['point'],device=base.device)
                        point_velocity=after['bat_velocity'][index,:3]+torch.cross(after['bat_velocity'][index,3:],point_world-com[index],dim=0)
                        event.update(position=points0[0]['point'],normal=points0[0]['normal'],bat_normal=normal[index].cpu().tolist(),
                                     bat_contact_point_velocity=point_velocity.cpu().tolist(),
                                     ball_velocity_before=before['ball_velocity'][index].cpu().tolist(),
                                     ball_velocity_after=after['ball_velocity'][index].cpu().tolist())
                        if args.contact_task:
                            for key in ['ball_velocity_before','ball_velocity_after','bat_normal','bat_contact_point_velocity']:event[key]=event0[key]
                    event_file.write(json.dumps(event)+'\n');events.append(event)
                done = nxt['done'].flatten()
                for index in range(100):
                    if bool(active_before[index]) and (bool(done[index]) or index in illegal or step+1 == base.max_episode_length):
                        is_truncated = bool(nxt['truncated'][index].flatten()[0])
                        reason = illegal.get(index,'time_limit' if is_truncated or not bool(done[index]) else 'boundary' if args.contact_task else 'upstream_termination')
                        outcomes[index] = {'scenario_id':index,'steps':step+1,'reason':reason,'truncated':is_truncated,
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
                    router.reset(reset_ids)
                else: td = nxt
                if (step+1)%100 == 0: record(completed_steps=step+1,completed_scenarios=int(finished.sum()))
        if args.contact_task:
            assert controller_calls == base.contact_totals['policy_steps'] == len(trajectories)
            assert base.contact_totals['physics_steps'] == controller_calls * base.substeps
            record(timing_audit={'controller_calls':controller_calls, 'policy_steps':len(trajectories), 'physics_steps':base.contact_totals['physics_steps'], 'policy_dt':.02, 'physics_dt':args.physics_dt},reset_reentries=router.reset_reentries)
        trajectory_path = args.output.with_suffix('.trajectory.npz')
        np.savez_compressed(trajectory_path, **{k:np.stack([row[k] for row in trajectories]) for k in trajectories[0]}, dt=.02,physics_dt=args.physics_dt)
        args.output.with_suffix('.outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        record(status='passed',evaluation_elapsed_seconds=time.perf_counter()-evaluation_started,gpu_contact_buffer_queries=gpu_queries, completed_scenarios=int(finished.sum()), steps=len(trajectories), outcomes=outcomes,
               ball_bat_entry_distribution=ball_bat_entries, provisional_top_entry_distribution=top_entries,
               provisional_five_top_contact_rate=sum(v>=5 for v in top_entries)/100,
               raw_events=str(event_path),trajectory=str(trajectory_path),
               trajectory_sha256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
               note='Aligned body-enabled juggling only. Same initial roster; episode rule recorded explicitly. Not formal WallRally success.')
        return 0
    except Exception as error:
        record(status='failed',error=repr(error),traceback=traceback.format_exc()); return 1
    finally:
        if app is not None: app.close()


if __name__ == '__main__':
    raise SystemExit(main())
