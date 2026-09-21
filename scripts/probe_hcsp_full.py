"""Frozen full HCSP phase-one replay, original six-drone environment."""
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
    a=p.parse_args();a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','pid':os.getpid(),'scope':'Full phase-one HCSP; original six-drone volleyball baseline','seed':a.seed}
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
                   'task.drone_model=IrisTest','task.sim.dt=0.01','task.sim.substeps=1',
                   'task.random_turn=true','task.use_trained_state=true',f'seed={a.seed}']
        with initialize_config_dir(version_base=None,config_dir=str(HCSP/'cfg')):
            cfg=compose(config_name='train_coselfplay_phase_one',overrides=overrides)
        OmegaConf.resolve(cfg);OmegaConf.set_struct(cfg,False)
        OmegaConf.save(cfg,a.output.with_suffix('.yaml'))
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')]
        app=init_simulation_app(cfg)
        from hcsp.envs import IsaacEnv
        from hcsp.learning import PSROPolicy_coselfplay_phase_one
        from torchrl.envs.transforms import TransformedEnv,Compose,InitTracker
        from check_upstream_contacts import contact_reporting_before_initialization
        wandb.init(mode='disabled',dir=str(a.output.parent))
        torch.manual_seed(a.seed);np.random.seed(a.seed)
        with contact_reporting_before_initialization():
            base=IsaacEnv.REGISTRY[cfg.task.name](cfg,headless=True)
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
        record(status='running',checkpoints=checkpoints,num_drones=base.drone.n,
               source_commit=subprocess.check_output(['git','-C',str(HCSP),'rev-parse','HEAD'],text=True).strip(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        active=torch.ones(a.num_envs,dtype=torch.bool,device=base.device)
        outcomes=[];trace={'ball':[],'drone':[],'high':[],'active':[],'hits':[]}
        with torch.no_grad():
            td=env.reset()
            for step in range(a.steps):
                td=policy(td,deterministic=False)
                trace['high'].append(td['agents','high_level_action'].cpu().numpy().copy())
                trace['active'].append(active.cpu().numpy().copy())
                nxt=env.step(td)['next']
                bp=base.ball.get_world_poses()[0]-base.envs_positions.unsqueeze(1)
                ds=base.drone.get_state()
                assert torch.isfinite(bp).all() and torch.isfinite(ds).all()
                trace['ball'].append(bp.cpu().numpy().copy());trace['drone'].append(ds.cpu().numpy().copy())
                trace['hits'].append(base.racket_hit_ball.cpu().numpy().copy())
                done=nxt['done'].flatten()
                for idx in torch.nonzero(active & (done | (step+1==a.steps))).flatten().tolist():
                    outcomes.append({'env':idx,'steps':step+1,'done':bool(done[idx]),'stats':{k:v[idx].cpu().tolist() for k,v in base.stats.items()}})
                    active[idx]=False
                if not active.any():break
                if done.any():nxt.set('_reset',nxt['done']);td=env.reset(nxt)
                else:td=nxt
                if step%100==0:print(f'step={step} active={active.sum().item()}',flush=True)
        trajectory=a.output.with_suffix('.npz');np.savez_compressed(trajectory,**{k:np.stack(v) for k,v in trace.items()})
        record(status='passed',outcomes=outcomes,trajectory=str(trajectory),trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),steps=step+1)
    except BaseException:
        record(status='failed',error=traceback.format_exc());raise
    finally:
        if app is not None:app.close()
if __name__=='__main__':main()
