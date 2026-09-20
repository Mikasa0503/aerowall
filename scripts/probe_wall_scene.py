"""Physical contact routing fixtures on original Air/ball assets plus cloned wall."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--align-cap-to-visual-top',action='store_true');a=p.parse_args()
    report={'status':'initializing','pid':os.getpid(),'scope':'contact-routing fixtures only; no WallRally reward or policy success',
            'file_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/probe_wall_scene.py','aerowall/envs/wall_scene.py','aerowall/contact_router.py','aerowall/rally_events.py','aerowall/gpu_contacts.py']}}
    def record(**values):
        report.update(values);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output);print(json.dumps(values),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);app=None;record()
    try:
        import torch
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        cfg=OmegaConf.load(ROOT/'runs/singlejuggle-dev-001.yaml');OmegaConf.set_struct(cfg,False)
        cfg.env.num_envs=cfg.task.env.num_envs=16;cfg.headless=True;cfg.wandb.mode='disabled'
        cfg.wall_fixture={'center':[2.5,0.,3.],'dimensions':[.2,4.,6.],'restitution':.8,'align_cap_to_visual_top':a.align_cap_to_visual_top}
        OmegaConf.save(cfg,a.output.with_suffix('.yaml'))
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')];app=init_simulation_app(cfg)
        from aerowall.envs.wall_scene import WallContactScene
        from aerowall.contact_router import WallContactRouter
        from check_upstream_contacts import contact_reporting_before_initialization
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        import carb.settings
        with contact_reporting_before_initialization(enable_body_collisions=True):base=WallContactScene(cfg,headless=True)
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
        router=WallContactRouter(base);base.set_seed(20260921)
        record(bat_overlay=base.bat_overlay,bat_physics_mass=router.bats.get_masses().cpu().tolist(),bat_physics_com_local=router.bats.get_coms()[0].cpu().tolist(),bat_physics_inertias=router.bats.get_inertias().cpu().tolist())
        phases=['cap','non_cap','ball_body','wall','ball_ground','drone_wall','drone_ground']
        records=[];trials=[];stale_gpu_slots=0;device=base.device;n=base.num_envs
        quat=torch.zeros(n,1,4,device=device);quat[...,0]=1
        from omni.isaac.core.prims import RigidPrimView
        rotor=RigidPrimView('/World/envs/env_*/Air_0/rotor_0',name='fixture_rotor',reset_xform_properties=False);rotor.initialize()
        import re
        ro=torch.tensor(sorted(range(n),key=lambda j:int(re.search(r'/env_(\d+)/',rotor.prim_paths[j]).group(1))),device=device)
        for phase in phases:
            record(status='checking',current_phase=phase)
            base.reset()
            dp=base.envs_positions[:,None,:].clone();dp[...,2]+=3.
            park=dp.clone();park[...,0]-=1.5
            zero=torch.zeros(n,1,6,device=device)
            base.drone.set_world_poses(dp,quat);base.drone.set_velocities(zero)
            base.ball.set_world_poses(park,quat);base.ball.set_velocities(zero)
            base.sim.step(render=False) # Flush the preceding fixture's manifolds before new fixture initialization.
            router.reset(range(n))
            bp=router.bats.get_world_poses()[0][router.bat_order].clone()
            bv=zero.clone();ball=park.clone()
            if phase in ['cap','non_cap','ball_body']:
                center=rotor.get_world_poses()[0][ro] if phase=='ball_body' else bp
                offset=torch.tensor([.14,0.,.025] if phase=='non_cap' else [0.,0.,.2],device=device)
                ball=(center+offset)[:,None,:];bv[...,0 if phase=='non_cap' else 2]=-2.
            elif phase=='wall':
                ball=dp.clone();ball[...,0]+=base.wall_front-.35;bv[...,0]=4.
            elif phase=='ball_ground':
                ball=dp.clone();ball[...,0]+=1.;ball[...,2]=.2;bv[...,2]=-2.
            elif phase=='drone_wall':
                dp[...,0]+=base.wall_front-.45;dv=zero.clone();dv[...,0]=6.
                base.drone.set_world_poses(dp,quat);base.drone.set_velocities(dv)
            elif phase=='drone_ground':
                dp[...,2]=.25;dv=zero.clone();dv[...,2]=-2.
                base.drone.set_world_poses(dp,quat);base.drone.set_velocities(dv)
            base.ball.set_world_poses(ball,quat);base.ball.set_velocities(bv)
            seen=set();kind_counts={};positive=0
            for step in range(15):
                before=base.ball.get_velocities().clone();base.sim.step(render=False)
                try:
                    impacts,events=router.read()
                except Exception:
                    from pxr import PhysicsSchemaTools
                    headers,_=router.interface.get_contact_report()
                    debug=[]
                    for sensor in router.sensors:
                        if '/env_0/' not in sensor.source_path:continue
                        force,point,normal,separation,counts,starts=sensor.view.get_contact_force_data(dt=1.)
                        matrix=sensor.view.get_contact_force_matrix(dt=1.)
                        slots=[]
                        for j,path in enumerate(sensor.filter_paths):
                            count,start=int(counts[0,j]),int(starts[0,j])
                            if count:slots.append({'filter':path,'impulses':force[start:start+count].cpu().tolist(),'points':point[start:start+count].cpu().tolist(),'normals':normal[start:start+count].cpu().tolist(),'aggregate':matrix[0,j].cpu().tolist()})
                        if slots:debug.append({'source':sensor.source_path,'slots':slots})
                    record(failure_step=step,contact_debug=debug,
                           cpu_pairs=[{k:str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ['actor0','actor1','collider0','collider1']} for h in headers if any('/env_0/' in str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ['actor0','actor1'])],
                           fixture_bat_position=bp[0].cpu().tolist(),ball_before=before[0,0].cpu().tolist(),ball_after=base.ball.get_velocities()[0,0].cpu().tolist())
                    raise
                stale_gpu_slots+=len(router.unmatched_gpu_slots)
                assert torch.isfinite(base.ball.get_velocities()).all() and torch.isfinite(base.drone.get_velocities()).all()
                for i,rows in enumerate(impacts):
                    for item in rows:
                        kind_counts[item.kind.value]=kind_counts.get(item.kind.value,0)+1;positive+=1
                        if item.kind.value==phase:seen.add(i)
                    target=base.envs_positions[i].cpu().tolist();target[0]+=base.wall_front;target[2]+=3.
                    router.batch.states[i].advance(rows,target)
                for e in events:
                    e.update(phase=phase,step=step,ball_velocity_before=before[e['env_id'],0].cpu().tolist(),ball_velocity_after=base.ball.get_velocities()[e['env_id'],0].cpu().tolist())
                records.extend(events)
            covered=seen==set(range(n))
            terminal_check=all(s.terminated for s in router.batch.states) if phase not in ['cap','wall'] else all(not s.terminated for s in router.batch.states)
            assert all(s.rallies==0 for s in router.batch.states)
            trial={'phase':phase,'passed':covered and terminal_check,'covered_envs':sorted(seen),'positive_kind_counts':kind_counts,'states':[asdict(s) for s in router.batch.states]}
            trials.append(trial);a.output.with_suffix('.events.json').write_text(json.dumps(records,indent=2)+'\n');record(status='checking',last_trial=trial,trials=trials)
        # Reset only one slot; other slot's counters and contact ledger are unchanged.
        other_before=asdict(router.batch.states[1]);ledger_before=dict(router.batch.ledgers[1].active)
        router.reset([0]);assert asdict(router.batch.states[1])==other_before and router.batch.ledgers[1].active==ledger_before
        assert not router.batch.states[0].terminated and not router.batch.ledgers[0].active
        passed=all(t['passed'] for t in trials)
        record(status='passed' if passed else 'failed',trials=trials,collider_owners=router.links,selective_event_reset_passed=True,unmatched_gpu_slots_ignored=stale_gpu_slots,
               note='All body collision flags enabled for these fixtures. No control/reward or continuous ball-wall-bat success is established.')
        return 0 if passed else 1
    except Exception as e:record(status='failed',error=repr(e),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()
if __name__=='__main__':raise SystemExit(main())
