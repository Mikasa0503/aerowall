"""Tilted physical bat launch toward wall: feasibility fixture, never policy success.

Writes positions/velocities only at fixture initialization. Subsequent motion is
uncontrolled PhysX; free-falling drone failures are retained, not called recovery.
"""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--dt',type=float,default=.02);p.add_argument('--matched-initial-state',action='store_true');p.add_argument('--ccd',action='store_true');a=p.parse_args()
    assert a.dt in (.02,.01,.005,.0025,.00125)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','pid':os.getpid(),'scope':'tilted free-body bat launch fixture; no trained wall return',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def record(**kw):
        report.update(kw);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output);print(json.dumps(kw),flush=True)
    app=None;record()
    try:
        import torch
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        cfg=OmegaConf.load(ROOT/'runs/singlejuggle-dev-001.yaml');OmegaConf.set_struct(cfg,False)
        cfg.sim.dt=a.dt
        cfg.env.num_envs=cfg.task.env.num_envs=16;cfg.headless=True;cfg.wandb.mode='disabled'
        cfg.wall_fixture={'center':[2.5,0.,3.],'dimensions':[.2,4.,6.],'restitution':.8,'align_cap_to_visual_top':True}
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')];app=init_simulation_app(cfg)
        from aerowall.envs.wall_scene import WallContactScene
        from aerowall.contact_router import WallContactRouter
        from aerowall.rally_events import Kind
        from check_upstream_contacts import contact_reporting_before_initialization
        from omni_drones.utils.torch import euler_to_quaternion,quat_rotate
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        import carb.settings
        with contact_reporting_before_initialization(enable_body_collisions=True,enable_ccd=a.ccd):base=WallContactScene(cfg,headless=True)
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
        router=WallContactRouter(base,eager_gpu=False);base.set_seed(20260921);base.reset()
        n=16;device=base.device
        angles=torch.linspace(20,50,n,device=device);rpy=torch.zeros(n,1,3,device=device);rpy[...,1]=angles[:,None]*torch.pi/180
        quat=euler_to_quaternion(rpy);normal=quat_rotate(quat,torch.tensor([0.,0.,1.],device=device).expand(n,1,3))
        dp=base.envs_positions[:,None,:].clone();dp[...,2]+=4.
        zero=torch.zeros(n,1,6,device=device)
        base.drone.set_world_poses(dp,quat);base.drone.set_velocities(zero)
        if a.matched_initial_state:
            # The pinned Air fixed bat joint has zero translation relative to
            # root. No warmup integrates a dt-dependent fall before the trial.
            bat=dp.clone()
            router.reset(range(n))
        else:
            park=dp.clone();park[...,0]-=1.5;base.ball.set_world_poses(park,quat);base.ball.set_velocities(zero)
            base.sim.step(render=False)
            router.reset(range(n))
            bat=router.bats.get_world_poses()[0][router.bat_order][:,None,:]
        ball=bat+normal*(.083+float(base.ball_radius)+.06)
        velocity=zero.clone();velocity[...,:3]=-8.*normal
        base.ball.set_world_poses(ball,quat);base.ball.set_velocities(velocity)
        initial_state={
            'drone_position':base.drone.get_world_poses()[0].cpu().tolist(),
            'drone_quaternion':base.drone.get_world_poses()[1].cpu().tolist(),
            'drone_velocity':base.drone.get_velocities().cpu().tolist(),
            'joint_positions':base.drone._view.get_joint_positions().cpu().tolist(),
            'joint_velocities':base.drone._view.get_joint_velocities().cpu().tolist(),
            'ball_position':base.ball.get_world_poses()[0].cpu().tolist(),
            'ball_velocity':base.ball.get_velocities().cpu().tolist(),
            'ball_restitution':base.current_restitution.cpu().tolist()}
        events=[];trajectories=[];first_cap={};first_wall={}
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        ccd_flags={path:stage.GetPrimAtPath(path).GetAttribute(attr).Get() for path,attr in [('/physicsScene','physxScene:enableCCD'),('/World/envs/env_0/ball','physxRigidBody:enableCCD'),('/World/envs/env_0/Air_0/bat','physxRigidBody:enableCCD')]}
        record(initial_state=initial_state,matched_initial_state=a.matched_initial_state,ccd_requested=a.ccd,ccd_flags=ccd_flags)
        record(status='running',tilt_degrees=angles.cpu().tolist(),initial_ball_position=ball.cpu().tolist(),initial_ball_velocity=velocity.cpu().tolist(),no_control=True,dt=base.dt)
        for step in range(round(2./base.dt)):
            before=base.ball.get_velocities().clone();base.sim.step(render=False)
            impacts,rows=router.read();after=base.ball.get_velocities().clone();position=base.ball.get_world_poses()[0]
            assert torch.isfinite(after).all() and torch.isfinite(base.drone.get_velocities()).all()
            for i,items in enumerate(impacts):
                for item in items:
                    value={'step':step,'time':(step+1)*base.dt,'position':position[i,0].cpu().tolist(),
                           'ball_velocity_before':before[i,0,:3].cpu().tolist(),'ball_velocity_after':after[i,0,:3].cpu().tolist()}
                    if item.kind==Kind.CAP:first_cap.setdefault(i,value)
                    if item.kind==Kind.WALL:first_wall.setdefault(i,value)
                target=base.envs_positions[i].cpu().tolist();target[0]+=base.wall_front;target[2]+=4.
                router.batch.states[i].advance(items,target)
            for row in rows:row.update(step=step,time=(step+1)*base.dt)
            events.extend(rows)
            trajectories.append({'time':(step+1)*base.dt,'ball_position':position.cpu().tolist(),'drone_position':base.drone.get_world_poses()[0].cpu().tolist()})
        outcomes=[{'env_id':i,'tilt_degrees':float(angles[i]),'first_cap':first_cap.get(i),'first_wall':first_wall.get(i),
                   'cap_then_wall_observed':i in first_cap and i in first_wall and first_cap[i]['step']<first_wall[i]['step'],
                   'rally_state':asdict(router.batch.states[i])} for i in range(n)]
        a.output.with_suffix('.events.json').write_text(json.dumps(events)+'\n')
        a.output.with_suffix('.trajectory.json').write_text(json.dumps(trajectories)+'\n')
        record(status='passed',outcomes=outcomes,cap_then_wall_count=sum(x['cap_then_wall_observed'] for x in outcomes),
               note='Passed means fixture ran with valid sensing. Physical cap/wall sequence can occur after a task failure; inspect rally_state. No controller, policy rate, recovery or legal full rally is claimed.')
        return 0
    except Exception as e:record(status='failed',error=repr(e),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()
if __name__=='__main__':raise SystemExit(main())
