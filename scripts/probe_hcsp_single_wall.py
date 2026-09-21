"""Single physical drone with full frozen HCSP skill library and explicit role arbitration."""
import argparse, hashlib, json, os, re, subprocess, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HCSP=ROOT/'third_party/HCSP'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--contacts',action='store_true')
    p.add_argument('--dt',type=float,default=0.01)
    p.add_argument('--priority',choices=['firstpass','set','attack'],default='firstpass')
    a=p.parse_args();a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','pid':os.getpid(),'scope':'Single physical drone; HCSP role replication and arbitration adapter; physical wall','seed':a.seed}
    def record(**kw):
        report.update(kw);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(kw),flush=True)
    app=None;record()
    try:
        import numpy as np
        import torch
        import wandb
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        sys.path.insert(0,str(HCSP))
        from hcsp import init_simulation_app
        OmegaConf.register_new_resolver('eval',eval,replace=True)
        overrides=['headless=true','wandb.mode=disabled',f'task.env.num_envs={a.num_envs}',
                   'task.drone_model=IrisTest',f'task.sim.dt={a.dt}','task.sim.substeps=1',
                   'task.random_turn=true','task.use_trained_state=true',f'seed={a.seed}']
        with initialize_config_dir(version_base=None,config_dir=str(HCSP/'cfg')):
            cfg=compose(config_name='train_coselfplay_phase_one',overrides=overrides)
        OmegaConf.resolve(cfg);OmegaConf.set_struct(cfg,False)
        cfg.single_priority=a.priority
        OmegaConf.save(cfg,a.output.with_suffix('.yaml'))
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')]
        app=init_simulation_app(cfg)
        from hcsp.envs import IsaacEnv
        from hcsp.learning import PSROPolicy_coselfplay_phase_one
        from torchrl.envs.transforms import TransformedEnv,Compose,InitTracker
        from check_upstream_contacts import contact_reporting_before_initialization
        wandb.init(mode='disabled',dir=str(a.output.parent))
        torch.manual_seed(a.seed);np.random.seed(a.seed)
        from hcsp_single_wall_env import HCSPSingleWall
        with contact_reporting_before_initialization():
            base=HCSPSingleWall(cfg,headless=True)
        assert base.physical_drone.n==1
        assert tuple(base.physical_drone.shape)==(a.num_envs,1)
        record(priority=a.priority,physical_drone_shape=list(base.physical_drone.shape),adapter_sha256=hashlib.sha256((ROOT/'scripts/hcsp_single_wall_env.py').read_bytes()).hexdigest())
        env=TransformedEnv(base,Compose(InitTracker())).eval()
        policy=PSROPolicy_coselfplay_phase_one(cfg.algo,agent_spec_dict=env.agent_spec,device=base.device,num_envs=a.num_envs)
        checkpoints=[]
        def load(path,role):
            path=HCSP/'scripts/shell'/path
            checkpoints.append({'role':role,'file':str(path.relative_to(HCSP)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            return torch.load(path,map_location=base.device)
        shell=(HCSP/'scripts/shell/coselfplay_phase_one.sh').read_text()
        for role,path in re.findall(r'(\w+)_policy_checkpoint_path="([^"]+)"',shell):
            if role=='high_level':continue
            if role=='SecPass':path='checkpoint/coselfplay/checkpoint_final_SecPass.pt'
            policy.load_state_dict(load(path,role),player=role)
        high=policy.policy_high_level
        state=load('checkpoint/coselfplay/policy_high_level_final.pt','high_level_full')
        for side in (0,1):
            high.init_population_from_checkpoint({'actor_params':state[f'actor_params_{side}']},side)
        high.set_actor_params_with_latest_policy(both_populations=True)
        high.critic.load_state_dict(state['critic'])
        high.value_normalizer.load_state_dict(state['value_normalizer'])
        high.set_latest_strategy();high.eval_payoff=True
        record(status='running',checkpoints=checkpoints,num_drones=base.physical_drone.n,virtual_role_slots=base.drone.n,
               source_commit=subprocess.check_output(['git','-C',str(HCSP),'rev-parse','HEAD'],text=True).strip(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        views=[];contacts=[]
        if a.contacts:
            from omni.isaac.core.prims.rigid_contact_view import RigidContactView
            import carb.settings
            from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
            carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
            for idx in range(a.num_envs):
                view=RigidContactView(f'/World/envs/env_{idx}/ball',
                    [f'/World/envs/env_{idx}/{base.drone.name}_{j}/base_link' for j in range(base.physical_drone.n)],
                    name=f'full_hcsp_contacts_{idx}',prepare_contact_sensors=False,
                    disable_stablization=False,apply_rigid_body_api=False,max_contact_count=64)
                view.initialize();assert view.num_shapes==1 and view.num_filters==base.physical_drone.n
                views.append(view)
        record(deterministic_low_level=a.deterministic,deterministic_high_level=True,dt=base.dt,
               contact_filters_drone_name=base.drone.name,independent_contacts=a.contacts)
        active=torch.ones(a.num_envs,dtype=torch.bool,device=base.device)
        outcomes=[];trace={'ball':[],'drone':[],'high':[],'active':[],'hits':[],'ball_velocity':[],'executed_high':[],'physical_drone':[],'executed_skill':[],'executed_role':[],'executed_action':[],'contact_impulse':[],'contact_entry':[],'wall_returns':[]}
        with torch.no_grad():
            td=env.reset()
            record(initial_ball=(base.ball.get_world_poses()[0]-base.envs_positions[:,None,:]).cpu().tolist(),initial_drone=base.physical_drone.get_state().cpu().tolist(),wall={'center':[0,0,4],'size':[0.2,8,8]})
            for step in range(a.steps):
                td=policy(td,deterministic=a.deterministic)
                trace['high'].append(td['agents','high_level_action'].cpu().numpy().copy())
                trace['active'].append(active.cpu().numpy().copy())
                nxt=env.step(td)['next']
                bp=base.ball.get_world_poses()[0]-base.envs_positions.unsqueeze(1)
                ds=base.drone.get_state()
                assert torch.isfinite(bp).all() and torch.isfinite(ds).all()
                trace['ball'].append(bp.cpu().numpy().copy());trace['drone'].append(ds.cpu().numpy().copy())
                trace['hits'].append(base.racket_hit_ball.cpu().numpy().copy())
                trace['ball_velocity'].append(base.ball.get_velocities().cpu().numpy().copy())
                trace['executed_high'].append(base.info['high_level_action'].cpu().numpy().copy())
                for idx,view in enumerate(views):
                    if not bool(active[idx]):continue
                    impulses,positions,normals,separations,counts,starts=view.get_contact_force_data(dt=1.0)
                    assert impulses.device.type=='cuda' and int(counts.sum())<64
                    for agent in range(base.physical_drone.n):
                        start=int(starts[0,agent]);count=int(counts[0,agent])
                        for ci in range(start,start+count):
                            impulse=float(impulses[ci].item())
                            if abs(impulse)<=1e-8:continue
                            contacts.append({'env':idx,'step':step,'agent':agent,'impulse':impulse,
                                'position':positions[ci].cpu().tolist(),'normal':normals[ci].cpu().tolist()})
                for key,value in [('physical_drone',base.physical_drone.get_state()),('executed_skill',base.executed_skill),('executed_role',base.executed_role),('executed_action',base.executed_action),('contact_impulse',base.contact_impulse),('contact_entry',base.contact_entry),('wall_returns',base.wall_returns)]:
                    trace[key].append(value.cpu().numpy().copy())
                done=nxt['done'].flatten()
                for idx in torch.nonzero(active & (done | (step+1==a.steps))).flatten().tolist():
                    outcomes.append({'env':idx,'steps':step+1,'done':bool(done[idx]),'wall_returns':int(base.wall_returns[idx]),'stats':{k:v[idx].cpu().tolist() for k,v in base.stats.items()}})
                    active[idx]=False
                if not active.any():break
                if done.any():nxt.set('_reset',nxt['done']);td=env.reset(nxt)
                else:td=nxt
                if step%100==0:print(f'step={step} active={active.sum().item()}',flush=True)
        trajectory=a.output.with_suffix('.npz');np.savez_compressed(trajectory,**{k:np.stack(v) for k,v in trace.items()})
        record(status='passed',contact_points=base.contact_points,env_origins=base.envs_positions.cpu().tolist(),contacts=contacts,outcomes=outcomes,trajectory=str(trajectory),trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),steps=step+1)
    except BaseException:
        record(status='failed',error=traceback.format_exc());raise
    finally:
        if app is not None:app.close()
if __name__=='__main__':main()
