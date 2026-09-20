"""Replay pinned HCSP Attack checkpoints in their original PRT environment.

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
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','pid':os.getpid(),'scope':'HCSP Attack reference replay; original PRT, not CTBR',
            'source_commit':subprocess.check_output(['git','-C',str(HCSP),'rev-parse','HEAD'],text=True).strip(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def record(**values):
        report.update(values);tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
        print(json.dumps(values),flush=True)
    app=None;record()
    try:
        sys.path.insert(0,str(HCSP))
        import numpy as np
        import torch
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        from hcsp import init_simulation_app
        source=HCSP/'scripts/shell/attack.sh'
        tokens=shlex.split(source.read_text().replace('\\\n',' '),comments=True)
        overrides=tokens[tokens.index('../train_attack.py')+1:]
        replacement={'task.env.num_envs':'16','headless':'true','wandb.mode':'disabled','only_eval':'true'}
        overrides=[v for v in overrides if v.split('=',1)[0] not in replacement]
        overrides += [f'{k}={v}' for k,v in replacement.items()]
        OmegaConf.register_new_resolver('eval',eval,replace=True)
        with initialize_config_dir(version_base=None,config_dir=str(HCSP/'cfg')):
            cfg=compose(config_name='train_attack',overrides=overrides)
        OmegaConf.resolve(cfg);OmegaConf.set_struct(cfg,False)
        OmegaConf.save(cfg,args.output.with_suffix('.yaml'))
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')]
        app=init_simulation_app(cfg)
        from hcsp.envs import IsaacEnv
        from hcsp.learning import MAPPOPolicy_Attack
        from torchrl.envs.transforms import TransformedEnv,Compose,InitTracker
        from check_upstream_contacts import contact_reporting_before_initialization
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        from pxr import PhysicsSchemaTools
        import carb.settings
        with contact_reporting_before_initialization():
            base=IsaacEnv.REGISTRY[cfg.task.name](cfg,headless=True)
        env=TransformedEnv(base,Compose(InitTracker())).eval()
        players=['SecPass','SecPass_hover','Att_goto','Att']
        policy=MAPPOPolicy_Attack(cfg.algo,agent_spec_dict={k:env.agent_spec[k] for k in players},device=base.device)
        names=['checkpoint_secpass.pt','checkpoint_secpass_hover.pt','checkpoint_goto.pt','checkpoint_att.pt']
        checkpoints=[]
        audit={r['file']:r for r in json.loads((ROOT/'docs/hcsp-checkpoint-audit.json').read_text())}
        for player,name in zip(players,names):
            path=HCSP/'scripts/shell/checkpoint'/name
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==audit[name]['sha256'],f'Checkpoint differs from pinned audit: {name}'
            policy.load_state_dict(torch.load(path,map_location=base.device),player=player)
            checkpoints.append({'player':player,'path':str(path),'sha256':digest})
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
        interface=get_physx_simulation_interface()
        env.set_seed(20260921)
        with torch.no_grad():td=env.reset()
        frames=[];events=[];finished=torch.zeros(16,dtype=torch.bool,device=base.device);outcomes=[None]*16
        record(status='replaying',checkpoints=checkpoints,num_envs=16,drone_model=cfg.task.drone_model,
               dt=base.dt,action_keys=[list(env.agent_spec[k].action_key) for k in players])
        with torch.no_grad():
            for step in range(base.max_episode_length):
                active=~finished.clone()
                policy(td,deterministic=True)
                action_data={k:td[env.agent_spec[k].action_key].cpu().numpy().copy() for k in players}
                nxt=env.step(td)['next']
                drone_pos,drone_quat=base.drone.get_world_poses()
                ball_pos,ball_quat=base.ball.get_world_poses()
                values={'drone_position':drone_pos.clone(),'drone_quaternion_wxyz':drone_quat.clone(),
                        'drone_velocity':base.drone.get_velocities().clone(),'ball_position':ball_pos.clone(),
                        'ball_velocity':base.ball.get_velocities().clone()}
                assert all(torch.isfinite(v).all() for v in values.values()),'Nonfinite HCSP state'
                frames.append({**{k:v.cpu().numpy() for k,v in values.items()},**{f'action_{k}':v for k,v in action_data.items()},'active_before':active.cpu().numpy()})
                headers,data=interface.get_contact_report()
                for h in headers:
                    row={'step':step,'time':(step+1)*base.dt,'type':str(h.type),
                         **{k:str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ['actor0','actor1','collider0','collider1']}}
                    row['points']=[{'position':list(data[h.contact_data_offset+j].position),
                                    'normal':list(data[h.contact_data_offset+j].normal),
                                    'impulse':list(data[h.contact_data_offset+j].impulse)} for j in range(h.num_contact_data)]
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
        record(status='passed',outcomes=outcomes,completed_scenarios=int(finished.sum()),steps=len(frames),
               trajectory=str(trajectory),trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
               contact_event_count=len(events),note='Replay completion only; no flip, recovery or task-success claim without trajectory analysis')
        return 0
    except Exception as error:
        record(status='failed',error=repr(error),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()


if __name__=='__main__':raise SystemExit(main())
