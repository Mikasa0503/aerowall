"""Replay pinned HCSP skill checkpoints in their original PRT environment.

This reference run never updates weights or participates in AeroWall's formal
comparison. All first-episode states and contact data are preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
HCSP=ROOT/'third_party/HCSP'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rearm',action='store_true')
    parser.add_argument('--same-side',action='store_true',help='Restrict initial ball y to original support [-2.3,-0.5] and stop on ball ground')
    parser.add_argument('--wall-source',choices=['HCSP','VolleyBots'],required=True)
    parser.add_argument('--skill', choices=['attack','set','receive','pass'], default='attack')
    parser.add_argument('--hover',action='store_true',help='Replay skill/hover policy chaining, including trained recovery')
    parser.add_argument('--default-pass-state',action='store_true',help='Diagnostic only: use upstream default reset instead of unavailable trained-state CSV distributions')
    parser.add_argument('--tensor-contacts',action='store_true',help='Read per-environment GPU contact buffers as a diagnostic')
    args=parser.parse_args()
    assert args.wall_source=='HCSP'
    assert args.same_side, 'Rearming test requires valid same-side initial states'
    assert args.skill=='set' and args.hover and args.tensor_contacts, 'First wall adapter covers Set+Hover with GPU contacts'
    weight_root=HCSP/'scripts/shell/checkpoint' if args.wall_source=='HCSP' else ROOT/'third_party/VolleyBots/checkpoints/hier'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','pid':os.getpid(),'scope':f'HCSP {args.skill} reference replay; original PRT, not CTBR',
            'source_commit':subprocess.check_output(['git','-C',str(HCSP),'rev-parse','HEAD'],text=True).strip(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def record(**values):
        report.update(values);tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
        print(json.dumps(values),flush=True)
    app=None;record(weight_source=args.wall_source,wall_fixture={'center':[2.9,0.,4.],'dimensions':[.2,8.,8.],'restitution':.8},transfer_scope='Native Iris/PRT Set+Hover unchanged weights and observations; added physical wall. HCSP Receive actor and hover on single native Iris; Receive observation layout, constructed same-side incoming serves. Optional wall rearm, no physical-state writes. No AeroWall legal-cap scoring.')
    try:
        sys.path.insert(0,str(HCSP))
        import numpy as np
        import torch
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        from hcsp import init_simulation_app
        mode=f'{args.skill}_hover' if args.hover else args.skill
        if args.skill!='attack':assert args.hover, 'Reference skill requires the trained hover chain'
        source=HCSP/'scripts/shell'/f'{mode}.sh'
        tokens=shlex.split(source.read_text().replace('\\\n',' '),comments=True)
        overrides=tokens[tokens.index(f'../train_{mode}.py')+1:]
        replacement={'task.env.num_envs':'16','headless':'true','wandb.mode':'disabled','only_eval':'true'}
        if args.default_pass_state:
            assert args.skill=='pass'
            replacement['task.use_trained_state']='false'
        record(initial_state_distribution='upstream default reset (diagnostic change)' if args.default_pass_state else 'original shell configuration')
        overrides=[v for v in overrides if v.split('=',1)[0] not in replacement]
        overrides += [f'{k}={v}' for k,v in replacement.items()]
        OmegaConf.register_new_resolver('eval',eval,replace=True)
        with initialize_config_dir(version_base=None,config_dir=str(HCSP/'cfg')):
            cfg=compose(config_name=f'train_{mode}',overrides=overrides)
        OmegaConf.resolve(cfg);OmegaConf.set_struct(cfg,False)
        if args.same_side:
            cfg.algo=OmegaConf.load(HCSP/'cfg/algo/mappo_receive_hover.yaml')
            cfg.task.initial.ball_pos_dist.low=[3.5,-.2,2.8]
            cfg.task.initial.ball_pos_dist.high=[3.7,.2,3.0]
            cfg.task.initial.ball_vel_dist.low=[2.,-.1,-1.]
            cfg.task.initial.ball_vel_dist.high=[2.5,.1,-.5]
            cfg.task.initial.SecPass_pos_dist.low=[4.5,0.,1.5]
            cfg.task.initial.SecPass_pos_dist.high=[4.5,0.,1.5]
            cfg.task.initial.SecPass_rpy_dist.low=[-.2,-.2,-.2]
            cfg.task.initial.SecPass_rpy_dist.high=[.2,.2,.2]
            cfg.task.initial.SecPass_lin_vel_dist.low=[0.,0.,0.]
            cfg.task.initial.SecPass_lin_vel_dist.high=[0.,0.,0.]
            cfg.task.initial.SecPass_ang_vel_dist.low=[0.,0.,0.]
            cfg.task.initial.SecPass_ang_vel_dist.high=[0.,0.,0.]
            cfg.task.SecPass_hover_pos_after_hit=[4.5,0.,1.5]
            record(initial_state_distribution='Constructed single-receiver incoming serves on x>3 wall side; Receive drone pose support; no opposing server; added ball-ground termination')
        OmegaConf.save(cfg,args.output.with_suffix('.yaml'))
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')]
        app=init_simulation_app(cfg)
        from hcsp.envs import IsaacEnv
        from hcsp.learning import MAPPOPolicy_Attack, MAPPOPolicy_Attack_hover, MAPPOPolicy_Set_hover, MAPPOPolicy_Receive_hover, MAPPOPolicy_Pass_hover
        from hcsp.utils.torch import quat_rotate
        from torchrl.envs.transforms import TransformedEnv,Compose,InitTracker
        from check_upstream_contacts import contact_reporting_before_initialization
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        from pxr import PhysicsSchemaTools
        import carb.settings
        parent=IsaacEnv.REGISTRY[cfg.task.name]
        class WallSetReference(parent):
            def _design_scene(self):
                paths=super()._design_scene()
                from omni.isaac.core.objects import FixedCuboid
                from omni.isaac.core.materials import PhysicsMaterial
                material=PhysicsMaterial('/World/Physics_Materials/reference_wall',restitution=.8)
                FixedCuboid('/World/envs/env_0/reference_wall',translation=np.array([2.9,0.,4.]),scale=np.array([.2,8.,8.]),size=1.,physics_material=material)
                return paths
            def _compute_state_and_obs(self):
                td=super()._compute_state_and_obs()
                # Receive uses the raw root state; Set canonicalizes quaternion sign.
                td['agents','SecPass_observation'][...,:23]=self.root_state
                expected=torch.cat([self.root_state,self.ball_pos,self.rpos,self.ball_vel,td['agents','SecPass_observation'][...,-2:]],dim=-1)
                assert torch.equal(expected,td['agents','SecPass_observation'])
                return td
            def _compute_reward_and_done(self):
                result=super()._compute_reward_and_done()
                if args.same_side:
                    ground=self.ball_pos[...,2] <= self.ball_radius
                    result['terminated'] |= ground
                    result['done'] |= ground
                if hasattr(self,'wall_transfer_views'):
                    for i,view in enumerate(self.wall_transfer_views):
                        # Same physical readback used in the saved contact evidence.
                        force=view.get_contact_force_matrix(dt=1.)[0,-1]
                        touching=bool(force.norm()>1e-8)
                        if touching and not self.wall_transfer_touching[i]:
                            self.SecPass_hit[i]=0.
                            self.SecPass_turn[i]=1
                            self.stats['SecPass_hit'][i]=0.
                            self.wall_transfer_rearms+=1
                        self.wall_transfer_touching[i]=touching
                return result
        with contact_reporting_before_initialization():
            base=WallSetReference(cfg,headless=True)
        env=TransformedEnv(base,Compose(InitTracker())).eval()
        players=['SecPass','SecPass_hover','Att_goto','Att']
        if args.hover:players.append('Att_hover')
        policy_type=MAPPOPolicy_Attack_hover if args.hover else MAPPOPolicy_Attack
        names=['checkpoint_secpass.pt','checkpoint_secpass_hover.pt','checkpoint_goto.pt','checkpoint_att.pt']
        if args.hover:names.append('checkpoint_att_hover.pt')
        if args.skill=='set':
            players=['SecPass','SecPass_hover']
            names=['checkpoint_firstpass_receive.pt','checkpoint_firstpass_receive_hover.pt']
            policy_type=MAPPOPolicy_Set_hover
        elif args.skill=='receive':
            players=['Opp_Server','Opp_Server_hover','FirstPass_goto','FirstPass','FirstPass_hover']
            names=['checkpoint_serve.pt','checkpoint_serve_hover.pt','checkpoint_goto.pt','checkpoint_firstpass_receive.pt','checkpoint_firstpass_receive_hover.pt']
            policy_type=MAPPOPolicy_Receive_hover
        elif args.skill=='pass':
            players=['Opp_SecPass','Opp_SecPass_hover','Opp_Att_goto','Opp_Att','Opp_Att_hover','FirstPass_goto','FirstPass','FirstPass_hover']
            names=['checkpoint_secpass.pt','checkpoint_secpass_hover.pt','checkpoint_goto.pt','checkpoint_att.pt','checkpoint_att_hover.pt','checkpoint_goto.pt','checkpoint_firstpass.pt','checkpoint_firstpass_hover.pt']
            policy_type=MAPPOPolicy_Pass_hover
        policy=policy_type(cfg.algo,agent_spec_dict={k:env.agent_spec[k] for k in players},device=base.device)
        agent_count=base.drone.n
        phase_keys={'set':['SecPass_hit'],'attack':['SecPass_hit','Att_hit'],
                    'receive':['Opp_Server_hit','FirstPass_hit'],
                    'pass':['Opp_SecPass_hit','Opp_Att_hit','FirstPass_hit']}[args.skill]
        checkpoints=[]
        audit={r['file']:r for r in json.loads((ROOT/'docs/hcsp-checkpoint-audit.json').read_text())}
        for player,name in zip(players,names):
            path=weight_root/name
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==audit[name]['sha256'],f'Checkpoint differs from pinned audit: {name}'
            policy.load_state_dict(torch.load(path,map_location=base.device),player=player)
            checkpoints.append({'player':player,'path':str(path),'sha256':digest})
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
        interface=get_physx_simulation_interface()
        callback_reports=[]
        callback_step=[-1]
        def contact_callback(headers,data):
            for h in headers:
                callback_reports.append({'step':callback_step[0],'type':str(h.type),
                    'actor0':str(PhysicsSchemaTools.intToSdfPath(h.actor0)),
                    'actor1':str(PhysicsSchemaTools.intToSdfPath(h.actor1)),
                    'offset':int(h.contact_data_offset),'count':int(h.num_contact_data),
                    'points':[{'position':list(data[h.contact_data_offset+j].position),
                               'normal':list(data[h.contact_data_offset+j].normal),
                               'impulse':list(data[h.contact_data_offset+j].impulse)} for j in range(h.num_contact_data)]})
        callback_subscription=interface.subscribe_contact_report_events(contact_callback)
        tensor_views=[];tensor_samples=[]
        if args.tensor_contacts:
            from omni.isaac.core.prims.rigid_contact_view import RigidContactView
            for index in range(16):
                view=RigidContactView(f'/World/envs/env_{index}/ball',
                    [f'/World/envs/env_{index}/Iris_{j}/base_link' for j in range(agent_count)]+[f'/World/envs/env_{index}/reference_wall'],
                    name=f'reference_ball_contacts_{index}',prepare_contact_sensors=False,
                    disable_stablization=False,apply_rigid_body_api=False,max_contact_count=32)
                view.initialize()
                assert view.num_shapes==1 and view.num_filters==agent_count+1
                tensor_views.append(view)
        base.wall_transfer_views=tensor_views if args.rearm else []
        base.wall_transfer_touching=[False]*16
        base.wall_transfer_rearms=0
        record(wall_rearm_enabled=args.rearm,observation_adapter='Receive 34-feature raw root/ball/relative ball/ball velocity/turn; role keys renamed only; Receive hover target [4.5,0,1.5]')
        env.set_seed(20260921)
        with torch.no_grad():td=env.reset()
        initial_ball=base.ball.get_world_poses()[0][:,0]-base.envs_positions
        initial_drone=base.drone.get_world_poses()[0][:,0]-base.envs_positions
        if args.same_side:
            assert bool((initial_ball[:,0]-base.ball_radius>3.).all())
            assert bool((initial_drone[:,0]-.5>3.).all())
        record(initial_ball_position=initial_ball.cpu().tolist(),initial_drone_position=initial_drone.cpu().tolist(),initial_same_side_clearance_checked=args.same_side)
        body_com_local=base.drone.base_link.get_coms()[0].reshape(16,agent_count,3)
        frames=[];events=[];finished=torch.zeros(16,dtype=torch.bool,device=base.device);outcomes=[None]*16
        record(status='replaying',env_origins=base.envs_positions.cpu().tolist(),task=cfg.task.name,checkpoints=checkpoints,num_envs=16,agent_count=agent_count,phase_keys=phase_keys,drone_model=cfg.task.drone_model,
               dt=base.dt,action_keys=[list(env.agent_spec[k].action_key) for k in players])
        with torch.no_grad():
            for step in range(base.max_episode_length):
                active=~finished.clone()
                policy(td,deterministic=True)
                action_data={k:td[env.agent_spec[k].action_key].cpu().numpy().copy() for k in players}
                phase_before={k:td['stats',k].cpu().numpy().copy() for k in phase_keys}
                callback_step[0]=step
                nxt=env.step(td)['next']
                drone_pos,drone_quat=base.drone.get_world_poses()
                ball_pos,ball_quat=base.ball.get_world_poses()
                body_pos,body_quat=base.drone.base_link.get_world_poses()
                body_vel=base.drone.base_link.get_velocities()
                body_com=body_pos+quat_rotate(body_quat,body_com_local)
                body_normal=quat_rotate(body_quat,torch.tensor([0.,0.,1.],device=base.device).expand(16,agent_count,3))
                values={'body_com_world':body_com.clone(),'body_velocity':body_vel.clone(),'body_normal':body_normal.clone(),'drone_position':drone_pos.clone(),'drone_quaternion_wxyz':drone_quat.clone(),
                        'drone_velocity':base.drone.get_velocities().clone(),'ball_position':ball_pos.clone(),
                        'ball_velocity':base.ball.get_velocities().clone()}
                assert all(torch.isfinite(v).all() for v in values.values()),'Nonfinite HCSP state'
                frames.append({**{k:v.cpu().numpy() for k,v in values.items()},**{f'action_{k}':v for k,v in action_data.items()},**{f'phase_before_{k}':v for k,v in phase_before.items()},'active_before':active.cpu().numpy()})
                for env_id,view in enumerate(tensor_views):
                    impulses,positions,normals,separations,counts,starts=view.get_contact_force_data(dt=1.0)
                    assert impulses.device.type=='cuda' and positions.device.type=='cuda'
                    assert int(counts.sum())<32, 'Tensor contact buffer may be saturated'
                    for agent_id in range(agent_count+1):
                        count=int(counts[0,agent_id]);start=int(starts[0,agent_id])
                        for contact in range(start,start+count):
                            magnitude=float(impulses[contact].item())
                            if abs(magnitude)<1e-8:continue
                            assert torch.isfinite(positions[contact]).all() and .9<float(normals[contact].norm())<1.1
                            if agent_id==agent_count:
                                tensor_samples.append({'kind':'wall','step':step,'env_id':env_id,'first_episode_active':bool(active[env_id]),'normal_impulse':magnitude,'position':positions[contact].cpu().tolist(),'normal':normals[contact].cpu().tolist(),'separation':float(separations[contact].item())})
                                continue
                            rotation=torch.cross(body_vel[env_id,agent_id,3:],positions[contact]-body_com[env_id,agent_id],dim=0)
                            tensor_samples.append({'step':step,'env_id':env_id,'agent_id':agent_id,
                                'first_episode_active':bool(active[env_id]),'normal_impulse':magnitude,
                                'position':positions[contact].cpu().tolist(),'normal':normals[contact].cpu().tolist(),
                                'separation':float(separations[contact].item()),
                                'body_normal':body_normal[env_id,agent_id].cpu().tolist(),
                                'translation_velocity':body_vel[env_id,agent_id,:3].cpu().tolist(),
                                'rotational_velocity':rotation.cpu().tolist(),
                                'contact_point_velocity':(body_vel[env_id,agent_id,:3]+rotation).cpu().tolist()})
                headers,data=interface.get_contact_report()
                for h in headers:
                    row={'step':step,'time':(step+1)*base.dt,'type':str(h.type),
                         **{k:str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ['actor0','actor1','collider0','collider1']}}
                    row['points']=[{'position':list(data[h.contact_data_offset+j].position),
                                    'normal':list(data[h.contact_data_offset+j].normal),
                                    'impulse':list(data[h.contact_data_offset+j].impulse)} for j in range(h.num_contact_data)]
                    import re
                    matches=[re.search(r'/env_(\d+)/Iris_(\d+)/base_link$',row[k]) for k in ['actor0','actor1']]
                    match=next((m for m in matches if m),None)
                    if match:
                        env_id,agent_id=map(int,match.groups())
                        row.update(env_id=env_id,agent_id=agent_id,first_episode_active=bool(active[env_id]))
                        for point in row['points']:
                            valid=sum(x*x for x in point['normal'])>.5 and sum(x*x for x in point['impulse'])>1e-12
                            point['valid_impulse_sample']=valid
                            if valid:
                                location=torch.tensor(point['position'],device=base.device)
                                angular=torch.cross(body_vel[env_id,agent_id,3:],location-body_com[env_id,agent_id],dim=0)
                                point.update(body_normal=body_normal[env_id,agent_id].cpu().tolist(),
                                             translation_velocity=body_vel[env_id,agent_id,:3].cpu().tolist(),
                                             rotational_velocity=angular.cpu().tolist(),
                                             contact_point_velocity=(body_vel[env_id,agent_id,:3]+angular).cpu().tolist())
                    events.append(row)
                done=nxt['done'].flatten()
                for index in range(16):
                    if bool(active[index]) and (bool(done[index]) or step+1==base.max_episode_length):
                        outcomes[index]={'scenario_id':index,'steps':step+1,'stats':{k:v[index].cpu().tolist() for k,v in nxt['stats'].items()}}
                        finished[index]=True
                if finished.all():break
                if done.any():nxt.set('_reset',nxt['done']);td=env.reset(nxt)
                else:td=nxt
                if (step+1)%100==0:record(completed_steps=step+1,completed_scenarios=int(finished.sum()))
        trajectory=args.output.with_suffix('.trajectory.npz')
        np.savez_compressed(trajectory,**{k:np.stack([v[k] for v in frames]) for k in frames[0]},dt=base.dt)
        args.output.with_suffix('.events.json').write_text(json.dumps(events,indent=2)+'\n')
        args.output.with_suffix('.tensor-contacts.json').write_text(json.dumps(tensor_samples,indent=2)+'\n')
        record(wall_rearm_count=base.wall_transfer_rearms,tensor_contact_points=len(tensor_samples),wall_positive_impulse_scenarios=sorted(set(x['env_id'] for x in tensor_samples if x.get('kind')=='wall' and x['first_episode_active'])),tensor_readback_enabled=args.tensor_contacts)
        args.output.with_suffix('.callback-events.json').write_text(json.dumps(callback_reports,indent=2)+'\n')
        record(callback_event_count=len(callback_reports),callback_positive_impulse_points=sum(sum(x*x for x in p['impulse'])>1e-12 for e in callback_reports for p in e['points']))
        record(status='passed',outcomes=outcomes,completed_scenarios=int(finished.sum()),steps=len(frames),
               trajectory=str(trajectory),trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
               contact_event_count=len(events),note='Replay completion only; no flip, recovery or task-success claim without trajectory analysis')
        return 0
    except Exception as error:
        record(status='failed',error=repr(error),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()


if __name__=='__main__':raise SystemExit(main())
