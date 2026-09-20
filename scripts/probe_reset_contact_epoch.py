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
    report={'status':'initializing','pid':os.getpid(),'scope':'physical reset-contact epoch fixtures only',
            'file_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/probe_reset_contact_epoch.py','aerowall/envs/wall_scene.py','aerowall/contact_router.py','aerowall/rally_events.py','aerowall/gpu_contacts.py','aerowall/collider_bounds.py']}}
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
        device=base.device;n=base.num_envs
        quat=torch.zeros(n,1,4,device=device);quat[...,0]=1
        zero=torch.zeros(n,1,6,device=device);results=[];all_events=[]
        base.reset()
        for name,height in [('ground_first',.058),('teleport_clear',1.2),('ground_again',.058)]:
            dp=base.envs_positions[:,None,:].clone();dp[...,2]+=height
            bp=base.envs_positions[:,None,:].clone();bp[...,0]-=1.;bp[...,2]+=3.
            base.drone.set_world_poses(dp,quat);base.drone.set_velocities(zero)
            base.ball.set_world_poses(bp,quat);base.ball.set_velocities(zero)
            router.reset(range(n))
            base.sim.step(render=False)
            impacts,events=router.read()
            ground=[i for i,row in enumerate(impacts) if any(v.kind.value=='drone_ground' for v in row)]
            discarded=sum(len(e['discarded_reset_points']) for e in events)
            for event in events:event['fixture']=name
            all_events.extend(events)
            passed=not ground if name=='teleport_clear' else len(ground)==n
            results.append({'fixture':name,'passed':passed,'ground_contact_envs':ground,'discarded_stale_points':discarded,
                            'drone_heights':base.drone.get_world_poses()[0][:,0,2].cpu().tolist()})
            record(status='checking',results=results)
        a.output.with_suffix('.events.json').write_text(json.dumps(all_events,indent=2)+'\n')
        passed=all(x['passed'] for x in results)
        record(status='passed' if passed else 'failed',results=results,source_collider_radii=router.owner_radii,
               note='First-step true ground contacts must survive geometric validation; teleport-separated old contacts must not score. Initialization placements are fixtures, not policy trajectories.')
        return 0 if passed else 1
    except Exception as e:record(status='failed',error=repr(e),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()
if __name__=='__main__':raise SystemExit(main())
