"""Paired momentum reaction gate for the body-enabled, visually aligned bat."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--dt',type=float,choices=[.02,.01],default=.02);a=p.parse_args()
    report={'status':'initializing','pid':os.getpid(),'dt':a.dt,'scope':'paired centered linear reaction; not policy or angular reaction',
            'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/probe_aligned_reaction.py','scripts/check_upstream_reaction.py','aerowall/envs/wall_scene.py']}}
    def record(**values):
        report.update(values);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output);print(json.dumps(values),flush=True)
    app=None;a.output.parent.mkdir(parents=True,exist_ok=True);record()
    try:
        import torch
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        cfg=OmegaConf.load(ROOT/'runs/singlejuggle-dev-001.yaml');OmegaConf.set_struct(cfg,False)
        cfg.env.num_envs=cfg.task.env.num_envs=16;cfg.headless=True;cfg.wandb.mode='disabled';cfg.sim.dt=cfg.task.sim.dt=a.dt
        cfg.wall_fixture={'center':[2.5,0.,3.],'dimensions':[.2,4.,6.],'restitution':.8,'align_cap_to_visual_top':True}
        OmegaConf.save(cfg,a.output.with_suffix('.yaml'))
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')];app=init_simulation_app(cfg)
        from aerowall.envs.wall_scene import WallContactScene
        from check_upstream_contacts import contact_reporting_before_initialization
        from check_upstream_reaction import check_reaction
        with contact_reporting_before_initialization(enable_body_collisions=True):base=WallContactScene(cfg,headless=True)
        base.set_seed(20260921)
        with torch.no_grad():
            result=check_reaction(base,base,record)
        record(status='passed' if result['passed'] else 'failed',bat_overlay=base.bat_overlay)
        return 0 if result['passed'] else 1
    except Exception as e:record(status='failed',error=repr(e),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()
if __name__=='__main__':raise SystemExit(main())
