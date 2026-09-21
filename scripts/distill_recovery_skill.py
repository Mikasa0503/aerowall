"""Collect fresh frozen-teacher rollouts and fit a single actor; no PPO or critic training.

Teacher weights are frozen. All newly collected transitions count toward the
budget. Whole environment IDs are held out; fixed evaluation states are not used.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM = ROOT / 'third_party/JuggleRL_train'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batches',type=int,default=8,help='Fresh teacher rollout batches; not PPO updates')
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--minibatch',type=int,default=512)
    parser.add_argument('--physics-dt',type=float,default=.0025)
    parser.add_argument('--wall-config',type=Path,default=ROOT/'configs/wall_single_return.yaml')
    args = parser.parse_args()
    args.wall_config=args.wall_config.resolve()
    assert args.epochs>0 and args.minibatch>0
    assert args.seed != 20260921, 'Frozen development roster seed is not training data'
    assert args.physics_dt in (.02,.01,.005,.0025,.00125)
    if args.batches < 1 or not 16 <= args.num_envs <= 512:
        parser.error('Require positive rollout batch count and 16–512 environments')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'pid': os.getpid(), 'phase': 'single_actor_skill_distillation',
              'seed': args.seed, 'collection_batches_requested': args.batches, 'performance_claim': False,
              'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / 'scripts/runtime_adapters.py', ROOT / 'scripts/source_evidence.py', ROOT / 'scripts/python.sh', ROOT / 'aerowall/envs/aligned_juggle.py', ROOT / 'aerowall/contact_router.py', ROOT / 'aerowall/rally_events.py', ROOT / 'aerowall/envs/wall_rally.py', ROOT / 'aerowall/learning/wall_policy.py', ROOT / 'aerowall/learning/phase_recovery.py', ROOT / 'aerowall/learning/return_reference.py', ROOT / 'aerowall/learning/recovery_skill_policy.py', ROOT / 'aerowall/collider_bounds.py', ROOT / 'aerowall/learning/skill_distillation.py', args.wall_config]}}
    def record(**values):
        report.update(values)
        tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(report, indent=2) + '\n'); tmp.replace(args.output)
        print(json.dumps(values), flush=True)
    app = None
    record()
    try:
        from source_evidence import snapshot_sources, verify_sources
        record(source_snapshot=snapshot_sources(ROOT, report['source_hashes'], args.output.with_suffix('.sources')))
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        tokens = shlex.split((UPSTREAM / 'scripts/shell/singlejuggle_sim2real.sh').read_text().replace('\\\n', ' '), comments=True)
        overrides = tokens[tokens.index('../train.py') + 1:]
        replacements = {'task.env.num_envs': str(args.num_envs), 'wandb.mode': 'disabled', 'headless': 'true', 'seed': str(args.seed)}
        overrides = [v for v in overrides if v.split('=', 1)[0] not in replacements]
        overrides += [f'{k}={v}' for k, v in replacements.items()]
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(UPSTREAM / 'cfg')):
            cfg = compose(config_name='train', overrides=overrides)
        OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
        cfg.sim.dt=args.physics_dt;cfg.sim.substeps=round(.02/args.physics_dt)
        wall_cfg=OmegaConf.load(args.wall_config)
        cfg.wall_fixture=wall_cfg.wall_fixture;cfg.wall_task=wall_cfg.wall_task
        # This run's actual budget overrides the author shell's two-billion cap.
        frames_per_batch = args.num_envs * int(cfg.algo.train_every)
        cfg.total_frames = args.batches * frames_per_batch
        cfg.algo.actor.output_dist_params=True
        config_text = OmegaConf.to_yaml(cfg)
        args.output.with_suffix('.yaml').write_text(config_text)
        learning_config = OmegaConf.to_container(cfg, resolve=True)
        for key in ['total_frames', 'save_interval', 'eval_interval', 'max_iters']:
            learning_config.pop(key, None)
        config_hash = hashlib.sha256(json.dumps(learning_config, sort_keys=True).encode()).hexdigest()
        sys.argv = [sys.argv[0], '--portable', '--portable-root', str(ROOT / '.cache/kit')]
        app = init_simulation_app(cfg)
        from omni_drones.envs import IsaacEnv
        from omni_drones.controllers import PID_controller_flightmare
        from omni_drones.learning import MAPPOPolicy
        from omni_drones.utils.torchrl import SyncDataCollector
        from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
        from runtime_adapters import ResetSafePIDRateController
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        from check_upstream_contacts import contact_reporting_before_initialization
        from aerowall.envs.wall_rally import WallRally
        from aerowall.learning.wall_policy import WallMAPPOPolicy
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        import carb.settings
        with contact_reporting_before_initialization(enable_body_collisions=True):
            base = WallRally(cfg, headless=True)
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
        controller = PID_controller_flightmare(.02, base.drone.params, base.device).to(base.device)
        controller_calls = 0
        def finite_output(module, inputs, output):
            nonlocal controller_calls
            controller_calls += 1
            assert torch.isfinite(output).all(), 'Non-finite controller output before upstream cleanup'
        controller.register_forward_hook(finite_output)
        env = TransformedEnv(base, Compose(InitTracker(), ResetSafePIDRateController(controller))).train()
        env.set_seed(args.seed)
        from aerowall.learning.recovery_skill_policy import RecoverySkillPolicy
        from aerowall.learning.skill_distillation import gaussian_kl,balanced_branch_loss
        from tensordict import TensorDict
        teacher_payload=torch.load(args.teacher,map_location=base.device)
        teacher=RecoverySkillPolicy(cfg.algo,agent_spec=env.agent_spec['drone'],device=base.device)
        teacher.load_state_dict(teacher_payload['policy']);teacher.eval()
        teacher_actor_before=teacher.actor_params.clone();launch_before=teacher.frozen_launch_params.clone()
        student=WallMAPPOPolicy(cfg.algo,agent_spec=env.agent_spec['drone'],device=base.device)
        launch_payload={'policy':{'actor_params':teacher.frozen_launch_params}}
        student.initialize_wall_actor(launch_payload)
        record(teacher=str(args.teacher),teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
               initial_launch_actor_audit=student.audit_wall_actor(launch_payload),
               policy_mode='single_actor_distillation',prior_environment_frames=teacher_payload['environment_frames'],
               budget_this_run=args.batches*frames_per_batch,critic_training=False,
               split='Last quarter of environment IDs held out throughout collection; no frozen development roster loaded')
        verify_sources(ROOT,report['source_hashes']);record(initialization_source_consistency=True)
        collector=SyncDataCollector(env,policy=teacher,frames_per_batch=frames_per_batch,
            total_frames=(args.batches+1)*frames_per_batch,device=base.device,return_same_td=True)
        train_parts=[];held_parts=[];held_envs=max(1,args.num_envs//4)
        def extract(data):
            flat=data.reshape(-1)
            return {name:flat[key].detach().cpu().clone() for name,key in [
                ('observation',teacher.obs_name),('loc',('debug','action_loc')),
                ('scale',('debug','action_scale')),('recovery',teacher.mask_key)]}
        record(status='collecting',held_out_env_ids=list(range(args.num_envs-held_envs,args.num_envs)))
        for index,data in enumerate(collector):
            audit=teacher.audit_rollout(data.to_tensordict())
            train_parts.append(extract(data[:-held_envs]));held_parts.append(extract(data[-held_envs:]))
            record(collected_batches=index+1,collected_frames=int(collector._frames),teacher_rollout_audit=audit)
            if index+1==args.batches:break
        assert int(collector._frames)==args.batches*frames_per_batch
        assert all(torch.equal(v,teacher.actor_params[k]) for k,v in teacher_actor_before.items(True,True))
        assert all(torch.equal(v,teacher.frozen_launch_params[k]) for k,v in launch_before.items(True,True))
        train={k:torch.cat([part[k] for part in train_parts]) for k in train_parts[0]}
        held={k:torch.cat([part[k] for part in held_parts]) for k in held_parts[0]}
        for split in [train,held]:
            assert all(torch.isfinite(v).all() for v in split.values())
            assert (split['scale']>0).all()
        dataset=args.output.with_suffix('.teacher-data.pt');assert not dataset.exists()
        torch.save({'train':train,'held_out':held,'teacher_sha256':report['teacher_sha256'],
            'seed':args.seed,'environment_frames':int(collector._frames)},dataset)
        record(dataset=str(dataset),dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
               training_rows=len(train['observation']),held_out_rows=len(held['observation']),teacher_parameters_unchanged=True)
        def distributions(observation):
            td=TensorDict({student.obs_name:observation},batch_size=list(observation.shape[:-1]),device=base.device)
            out=student.actor(td,student.actor_params,deterministic=True)
            return out['debug','action_loc'],out['debug','action_scale']
        @torch.no_grad()
        def assess(split):
            totals={'launch':[0.,0.,0],'recovery':[0.,0.,0]}
            for start in range(0,len(split['observation']),args.minibatch):
                batch={k:v[start:start+args.minibatch].to(base.device) for k,v in split.items()}
                loc,scale=distributions(batch['observation']);kl=gaussian_kl(batch['loc'],batch['scale'],loc,scale).reshape(-1)
                error=(loc-batch['loc']).square().mean(-1).reshape(-1);mask=batch['recovery'].reshape(-1).bool()
                for name,m in [('launch',~mask),('recovery',mask)]:
                    totals[name][0]+=float(kl[m].sum());totals[name][1]+=float(error[m].sum());totals[name][2]+=int(m.sum())
            return {name:{'rows':n,'mean_kl':k/n if n else None,'action_mean_rmse':math.sqrt(e/n) if n else None} for name,(k,e,n) in totals.items()}
        record(status='fitting',held_out_before=assess(held),training_before=assess(train))
        history=[]
        for epoch in range(args.epochs):
            order=torch.randperm(len(train['observation']));loss_sum=0.;batches=0
            for indices in order.split(args.minibatch):
                batch={k:v[indices].to(base.device) for k,v in train.items()}
                loc,scale=distributions(batch['observation']);kl=gaussian_kl(batch['loc'],batch['scale'],loc,scale)
                loss=balanced_branch_loss(kl,batch['recovery']);assert torch.isfinite(loss)
                student.actor_opt.zero_grad();loss.backward()
                grads=[v.grad for v in student.actor_params.values(True,True) if v.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                torch.nn.utils.clip_grad_norm_(list(student.actor_params.values(True,True)),5.)
                student.actor_opt.step();loss_sum+=float(loss.detach());batches+=1
            history.append({'epoch':epoch+1,'mean_minibatch_balanced_kl':loss_sum/batches,'held_out':assess(held)})
            record(fitting_progress=history[-1])
        assert all(torch.equal(v,teacher.actor_params[k]) for k,v in teacher_actor_before.items(True,True))
        assert all(torch.equal(v,teacher.frozen_launch_params[k]) for k,v in launch_before.items(True,True))
        checkpoint_dir=ROOT/'checkpoints'/args.output.stem;checkpoint_dir.mkdir(parents=True,exist_ok=False)
        frames=teacher_payload['environment_frames']+int(collector._frames)
        checkpoint=checkpoint_dir/f'frames-{frames:012d}.pt'
        payload={'policy':student.state_dict(),'environment_frames':frames,'n_updates':0,
          'source_hashes':report['source_hashes'],'teacher_sha256':report['teacher_sha256'],
          'requires_actor_only_initialization':True,'scope':'Single actor fitted to teacher distributions; critic untrained. PPO continuation must use initialize-policy, not resume.'}
        assert 'frozen_launch_params' not in payload['policy'];torch.save(payload,checkpoint)
        restored=WallMAPPOPolicy(cfg.algo,agent_spec=env.agent_spec['drone'],device=base.device)
        restored.load_state_dict(torch.load(checkpoint,map_location=base.device)['policy'])
        sample=train['observation'][:64].to(base.device)
        with torch.no_grad():
            td=TensorDict({student.obs_name:sample},batch_size=list(sample.shape[:-1]),device=base.device)
            actual=student.actor(td.clone(),student.actor_params,deterministic=True)[student.act_name]
            reloaded=restored.actor(td.clone(),restored.actor_params,deterministic=True)[restored.act_name]
        assert torch.equal(actual,reloaded)
        record(status='passed',checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            frames_this_run=int(collector._frames),environment_frames=frames,history=history,
            held_out_after=assess(held),training_after=assess(train),single_actor_checkpoint_actions_exact=True,
            scope='Distillation execution and checkpoint integrity only; teacher and student physical task success require fixed-scene evaluation')
        return 0
    except Exception as error:
        record(status='failed',error=repr(error),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()

if __name__=='__main__':raise SystemExit(main())
