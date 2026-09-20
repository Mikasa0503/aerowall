"""Native GPU wall-point and contact-lifecycle fixture; not a learned wall rally."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report={'status':'initializing','pid':os.getpid(),'scope':'16 independent one-wall physical fixtures, no bat or policy',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def record(**values):
        report.update(values);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
        print(json.dumps(values),flush=True)
    app=None;a.output.parent.mkdir(parents=True,exist_ok=True);record()
    try:
        from omni.isaac.kit import SimulationApp
        sys.argv=[sys.argv[0],'--portable','--portable-root',str(ROOT/'.cache/kit')]
        app=SimulationApp({'headless':True,'anti_aliasing':0},experience=str(Path(os.environ['EXP_PATH'])/'omni.isaac.sim.python.kit'))
        import numpy as np
        import torch
        import omni.usd
        from omegaconf import OmegaConf
        from omni.isaac.core import SimulationContext
        from omni.isaac.core.objects import DynamicSphere,FixedCuboid
        from omni.isaac.core.materials import PhysicsMaterial
        from omni.isaac.core.prims import RigidPrimView
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema,PhysicsSchemaTools
        from aerowall.gpu_contacts import GPUContactReadback
        from aerowall.rally_events import RallyBatch,Kind
        cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'third_party/JuggleRL_train/cfg/base/sim_base.yaml').sim,resolve=True)
        cfg['dt']=.02
        sim=SimulationContext(stage_units_in_meters=1.,physics_dt=.02,rendering_dt=.02,backend='torch',device='cuda:0',sim_params=cfg)
        material=PhysicsMaterial('/World/material',restitution=.8)
        for i in range(16):
            path=f'/World/cases/case_{i}'
            FixedCuboid(path+'/wall',translation=np.array([0.,8.*i,2.]),scale=np.array([.2,4.,4.]),size=1.,physics_material=material)
            ball=DynamicSphere(path+'/ball',translation=np.array([-.45,8.*i,1.5]),radius=.04,mass=.0472,physics_material=material)
            PhysxSchema.PhysxContactReportAPI.Apply(ball.prim).CreateThresholdAttr().Set(0.)
        balls=RigidPrimView('/World/cases/case_*/ball',name='wall_probe_balls')
        sim.reset();balls.initialize()
        stage=omni.usd.get_context().get_stage()
        assert PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(sim.get_physics_context().prim_path)).GetEnableGPUDynamicsAttr().Get()
        ids=[int(p.split('/case_')[1].split('/')[0]) for p in balls.prim_paths]
        rows={env:row for row,env in enumerate(ids)}
        sensors=[GPUContactReadback(f'/World/cases/case_{i}/ball',[f'/World/cases/case_{i}/wall'],f'wall_points_{i}') for i in range(16)]
        initial=torch.zeros(16,6,device='cuda:0')
        for row,i in enumerate(ids):initial[row,0]=1.+7.*i/15
        balls.set_velocities(initial)
        batch=RallyBatch(16);events=[];velocities=[];interface=get_physx_simulation_interface()
        credits=[0]*16;gpu_samples=[0]*16;cpu_invalid_points=0
        record(status='stepping',gpu=torch.cuda.get_device_name(0),sim_config=cfg,view_order=ids)
        for step in range(50):
            before=balls.get_velocities().clone();sim.step(render=False);after=balls.get_velocities().clone()
            assert torch.isfinite(after).all()
            samples=[s.read()[0] for s in sensors]
            headers,data=interface.get_contact_report();impacts=[[] for _ in range(16)]
            for h in headers:
                paths=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ['actor0','actor1']]
                if not any(path.endswith('/ball') for path in paths):continue
                ball_path=next(path for path in paths if path.endswith('/ball'));i=int(ball_path.split('/case_')[1].split('/')[0])
                expected={ball_path,f'/World/cases/case_{i}/wall'}
                assert set(paths)==expected,paths
                edge=str(h.type).rsplit('_',1)[-1].lower()
                gpu=samples[i]
                if len(gpu)>1:raise RuntimeError('Sphere-wall fixture has multiple positive points; inspect rather than choosing arbitrarily')
                magnitude=sum(v['impulse'] for v in gpu);point=gpu[0]['point'] if gpu else None
                for j in range(h.num_contact_data):
                    normal=list(data[h.contact_data_offset+j].normal)
                    if sum(v*v for v in normal)<.5:cpu_invalid_points+=1
                if gpu:
                    gpu_samples[i]+=1
                    assert abs(point[0]+.1)<.05,(i,point)
                    assert abs(gpu[0]['normal'][0])>.99
                    assert abs(point[1]-i*8.)<.01
                impact=batch.ledgers[i].observe(tuple(paths),edge,Kind.WALL,magnitude,point)
                if impact is not None:impacts[i].append(impact);credits[i]+=1
                events.append({'step':step,'env_id':i,'edge':edge,'gpu':gpu,'credited':impact is not None,
                               'before_v':before[rows[i]].cpu().tolist(),'after_v':after[rows[i]].cpu().tolist()})
            for i in range(16):batch.states[i].advance(impacts[i],(-.1,i*8.,1.5))
            velocities.append(after.cpu().tolist())
        a.output.with_suffix('.events.json').write_text(json.dumps(events,indent=2)+'\n')
        assert credits==[1]*16,credits
        assert all(s.rallies==0 and s.wall_hits==1 and s.target_index==1 for s in batch.states)
        assert all(not l.active for l in batch.ledgers), 'All reflected balls must depart their wall contacts'
        for i in range(16):
            measured=[e for e in events if e['env_id']==i and e['credited']]
            assert len(measured)==1 and measured[0]['before_v'][0]>0 and measured[0]['after_v'][0]<0
        record(status='passed',positive_wall_credits=credits,gpu_sample_counts=gpu_samples,
               cpu_invalid_point_count=cpu_invalid_points,states=[asdict(s) for s in batch.states],
               all_contacts_departed=True,ball_mass=.0472,ball_radius=.04,
               point_checks='front face x=-0.1 within 5 cm, unit normal along x, local y within 1 cm',
               limitation='Wall-only fixture must produce zero rallies; full bat-wall-bat integration not established')
        return 0
    except Exception as e:record(status='failed',error=repr(e),traceback=traceback.format_exc());return 1
    finally:
        if app is not None:app.close()
if __name__=='__main__':raise SystemExit(main())
