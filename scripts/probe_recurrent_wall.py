"""Native short GRU-PPO data-flow probe; not learned wall-return evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    files=['scripts/probe_recurrent_wall.py','scripts/source_evidence.py','scripts/runtime_adapters.py',
           'aerowall/learning/recurrent_policy.py','aerowall/learning/recurrent_protocol.py','aerowall/learning/wall_policy.py',
           'aerowall/envs/wall_rally.py','aerowall/envs/aligned_juggle.py','aerowall/contact_router.py','aerowall/collider_bounds.py','aerowall/rally_events.py']
    report={'status':'initializing','pid':os.getpid(),'scope':'16-env recurrent PPO integration; fresh random policy, not task success',
            'source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}}
    def record(**kw):
        report.update(kw);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output);print(json.dumps(kw),flush=True)
    app=None;record()
    try:
        from source_evidence import snapshot_sources,verify_sources
        record(source_snapshot=snapshot_sources(ROOT,report['source_hashes'],a.output.with_suffix('.sources')))
        import torch
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        cfg=OmegaConf.load(ROOT/'runs/wall-outbound-dev-001.yaml');OmegaConf.set_struct(cfg,False)
        cfg.env.num_envs=cfg.task.env.num_envs=16
        width=int(cfg.algo.actor.hidden_units[-1])
        cfg.algo.actor.rnn={'cls':'gru','kwargs':{'hidden_size':width},'train_seq_len':16}
        cfg.headless=True;cfg.seed=20260921
        a.output.with_suffix('.yaml').write_text(OmegaConf.to_yaml(cfg))
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')];app=init_simulation_app(cfg)
        from aerowall.envs.wall_rally import WallRally
        from aerowall.learning.recurrent_policy import RecurrentWallPolicy
        from check_upstream_contacts import contact_reporting_before_initialization
        from runtime_adapters import ResetSafePIDRateController
        from omni_drones.controllers import PID_controller_flightmare
        from omni_drones.utils.torchrl import SyncDataCollector
        from torchrl.envs.transforms import TransformedEnv,Compose,InitTracker,TensorDictPrimer
        from torchrl.data import UnboundedContinuousTensorSpec
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        import carb.settings
        with contact_reporting_before_initialization(enable_body_collisions=True):base=WallRally(cfg,headless=True)
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
        key='drone.actor_rnn_state'
        env=TransformedEnv(base,Compose(InitTracker(),
            TensorDictPrimer(primers={key:UnboundedContinuousTensorSpec((16,1,width),device=base.device)},default_value=0.),
            ResetSafePIDRateController(PID_controller_flightmare(.02,base.drone.params,base.device).to(base.device))))
        env.set_seed(cfg.seed);policy=RecurrentWallPolicy(cfg.algo,env.agent_spec['drone'],base.device)
        assert policy.hidden_key==key
        verify_sources(ROOT,report['source_hashes']);record(status='collecting',initialization_source_consistency=True)
        collector=SyncDataCollector(env,policy=policy,frames_per_batch=1024,total_frames=3072,device=base.device,return_same_td=True)
        audits=[]
        for j,data in enumerate(collector):
            hidden=data[key];future=data['next',key];initial=data['is_init'].squeeze(-1)
            assert not hidden[initial].any(),'Reset hidden state is nonzero'
            continuation=~initial[:,1:]
            assert torch.equal(hidden[:,1:][continuation],future[:,:-1][continuation]),'Hidden carry differs'
            with torch.no_grad():
                replay=policy.sequence_actor_output(data)[policy.act_logps_name]
                error=float((replay-data[policy.act_logps_name].squeeze(2)).abs().max())
            assert error<2e-4,('Rollout/sequence log probabilities differ',error)
            assert data['next','agents','state'].shape==(16,64,47)
            selected=data.select(*policy.train_in_keys,strict=False)
            assert selected['next','agents','state'].shape==(16,64,47)
            metrics=policy.train_op(data)
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            audits.append({'batch':j+1,'logp_replay_max_error':error,'reset_steps':int(initial.sum()),'metrics':metrics})
            record(batch_audits=audits)
            if j==1:break
        assert len(audits)==2
        directory=ROOT/'checkpoints'/a.output.stem;directory.mkdir(parents=True,exist_ok=False)
        checkpoint=directory/'frames-000000002048.pt'
        torch.save({'policy':policy.state_dict(),'actor_optimizer':policy.actor_opt.state_dict(),
                    'critic_optimizer':policy.critic_opt.state_dict(),'n_updates':policy.n_updates,
                    'environment_frames':2048,'source_hashes':report['source_hashes']},checkpoint)
        restored=RecurrentWallPolicy(cfg.algo,env.agent_spec['drone'],base.device)
        restored.load_state_dict(torch.load(checkpoint,map_location=base.device)['policy'])
        td=env.reset()
        with torch.no_grad():
            left=policy(td.clone(),deterministic=True);right=restored(td.clone(),deterministic=True)
        assert torch.equal(left['agents','action'],right['agents','action'])
        assert torch.equal(left['next',key],right['next',key])
        td=env.reset();td[key]=torch.randn_like(td[key]);before=td[key].clone()
        mask=torch.zeros(16,1,dtype=torch.bool,device=base.device);mask[0]=True;td['_reset']=mask
        reset=env.reset(td);assert not reset[key][0].any() and torch.equal(reset[key][1:],before[1:])
        record(status='passed',frames=2048,updates=policy.n_updates,checkpoint=str(checkpoint),
               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               hidden_carry_passed=True,selective_hidden_reset_passed=True,reload_action_and_hidden_exact=True,
               wall_totals=base.wall_totals,formal_comparison_ready=False)
        return 0
    except Exception as e:record(status='failed',error=repr(e),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()


if __name__=='__main__':raise SystemExit(main())
