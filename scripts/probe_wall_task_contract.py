"""WallRally interface/target/reset fixtures; no policy-success trajectory."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report={'status':'initializing','pid':os.getpid(),'scope':'explicit state/clock initialization fixtures; not policy success'}
    def record(**values):
        report.update(values);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output);print(json.dumps(values),flush=True)
    app=None;record()
    try:
        import torch
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        cfg=OmegaConf.load(ROOT/'runs/singlejuggle-dev-001.yaml');OmegaConf.set_struct(cfg,False)
        w=OmegaConf.load(ROOT/'configs/wall_single_return.yaml');cfg.wall_fixture=w.wall_fixture;cfg.wall_task=w.wall_task
        cfg.wall_task.targets_yz=[[0.,1.8],[.4,2.1]]
        cfg.env.num_envs=cfg.task.env.num_envs=16;cfg.sim.dt=.0025;cfg.sim.substeps=8;cfg.headless=True
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')];app=init_simulation_app(cfg)
        from aerowall.envs.wall_rally import WallRally
        from check_upstream_contacts import contact_reporting_before_initialization
        from omni_drones.controllers import PID_controller_flightmare
        from runtime_adapters import ResetSafePIDRateController
        from torchrl.envs.transforms import TransformedEnv,Compose,InitTracker
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        import carb.settings
        with contact_reporting_before_initialization(enable_body_collisions=True):base=WallRally(cfg,headless=True)
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
        env=TransformedEnv(base,Compose(InitTracker(),ResetSafePIDRateController(PID_controller_flightmare(.02,base.drone.params,base.device).to(base.device))))
        env.set_seed(20260921);td=env.reset()
        assert td['agents','observation'].shape==(16,1,43) and td['agents','state'].shape==(16,47)
        assert not base.info['prev_action'].any()
        old=base.current_restitution.clone();original=base._compute_state_and_obs()['agents','observation'].clone()
        base.current_restitution[:]=.123;cfg.wall_fixture.restitution=.456
        changed=base._compute_state_and_obs()['agents','observation'].clone()
        assert torch.equal(original,changed),'Actor reads hidden restitution configuration'
        base.current_restitution.copy_(old);cfg.wall_fixture.restitution=.8
        bp=base.envs_positions[:,None,:].clone();bp[...,0]+=base.wall_front-.12;bp[...,2]+=1.8
        bv=torch.zeros(16,1,6,device=base.device);bv[...,0]=2.
        base.ball.set_world_poses(bp);base.ball.set_velocities(bv);td.update(base._compute_state_and_obs())
        def step(td):
            action=torch.zeros(16,1,4,device=base.device);action[...,3]=torch.atanh(torch.tensor(2*9.81/15-1,device=base.device))
            td['agents','action']=action
            return env.step(td)['next']
        for _ in range(8):td=step(td)
        assert all(s.wall_hits==1 and s.target_index==1 and s.rallies==0 and not s.terminated for s in base.router.batch.states)
        expected=torch.tensor([base.wall_front,.4,2.1],device=base.device)
        assert torch.allclose(td['agents','observation'][:,0,33:36],expected.expand(16,3))
        base.progress_buf[:]=base.max_episode_length-1;td=step(td)
        assert td['truncated'].all() and not td['terminated'].any()
        other_target=base.targets[1].clone();other_action=base.info['prev_action'][1].clone();other_state=vars(base.router.batch.states[1]).copy()
        mask=torch.zeros(16,1,dtype=torch.bool,device=base.device);mask[0]=True;td['_reset']=mask;td=env.reset(td)
        assert torch.equal(base.targets[1],other_target) and torch.equal(base.info['prev_action'][1],other_action)
        assert vars(base.router.batch.states[1])==other_state
        assert not base.info['prev_action'][0].any() and base.router.batch.states[0].target_index==0
        assert not base.router.batch.states[0].terminated and not base.router.batch.states[0].truncated
        assert torch.allclose(base.targets[0]-base.envs_positions[0],torch.tensor([base.wall_front,0.,1.8],device=base.device))
        record(status='passed',observation_shape=[16,1,43],state_shape=[16,47],actual_wall_hits=16,false_rallies=0,
               target_publication_passed=True,actor_restitution_read_isolation=True,timeout_separate=True,selective_reset_passed=True,
               caveat='Hidden-parameter test checks observation dependency, not randomization of actual physics materials.')
        return 0
    except Exception as e:record(status='failed',error=repr(e),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()
if __name__=='__main__':raise SystemExit(main())
