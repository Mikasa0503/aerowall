"""Exercise actual WallRally contact reward method without starting Isaac.

Mock only rigid-body IO; use the real event state machine and staged method.
This checks scoring semantics, not physics or policy performance.
"""
import argparse, ast, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aerowall.rally_events import Impact, Kind, RallyState
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
tree=ast.parse(a.source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WallRally')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_after_physics_contacts')
namespace={'torch':torch,'Kind':Kind};exec(compile(ast.Module(body=[method],type_ignores=[]),str(a.source),'exec'),namespace)
step=namespace['_after_physics_contacts'];cfg=OmegaConf.load(a.config)
def body():
    return NS(get_world_poses=lambda:(torch.tensor([[[0.,0.,1.5]]]),None),get_velocities=lambda:torch.zeros(1,1,6))
def fixture(started=True):
    state=RallyState()
    if started:state.advance([Impact(Kind.CAP,('ball','bat'),1.)],[.9,0.,2.8],.65)
    spec=OmegaConf.create(OmegaConf.to_container(cfg.wall_task));spec.launch_reward=0.
    obj=NS(ball=body(),drone=body(),envs_positions=torch.zeros(1,3),device='cpu',wall_spec=spec,
        targets=torch.tensor([[.9,0.,2.8]]),target_sequence=torch.tensor([[0.,2.8]]),progress_buf=torch.ones(1),substeps=8,max_episode_length=500,
        router=NS(batch=NS(states=[state])),event_reward=torch.zeros(1,1),wall_cap_counts=torch.zeros(1,1),reset_boundary_failures=[],wall_front=.9,
        wall_totals=dict(rallies=0,joint_rallies=0,outbound_legs=0,wall_hits=0),launch_stats=dict(contacts=0,positive=0,negative=0,shaping_sum=0.,forward_velocity_sum=0.))
    return obj,state
wall=lambda z=2.8,front=True:Impact(Kind.WALL,('ball','wall'),1.,(.9,0.,z),front)
cases=[('high_front',[wall()],True,15.),('low_front',[wall(1.2)],True,0.),('side_or_back',[wall(front=False)],True,0.),('bare_wall',[wall()],False,0.),('double_wall',[wall(),wall()],True,0.),('illegal_same_step',[wall(),Impact(Kind.BALL_BODY,('ball','body'),1.)],True,-20.)]
rows=[]
for name,impacts,started,expected in cases:
    obj,state=fixture(started);step(obj,[impacts],[],0);actual=float(obj.event_reward)
    assert actual==expected,(name,actual,expected)
    assert state.rallies==0, 'A first wall hit is not a completed rally'
    rows.append(dict(case=name,reward=actual,rallies=state.rallies,passed=True))
obj,state=fixture();step(obj,[[wall()]],[],0);obj.event_reward.zero_();step(obj,[[Impact(Kind.CAP,('ball','bat'),1.)]],[],1)
assert state.rallies==1 and float(obj.event_reward)==100.
rows.append(dict(case='subsequent_real_cap',reward=float(obj.event_reward),rallies=state.rallies,passed=True))
a.output.write_text(json.dumps(dict(status='passed',scope=__doc__,source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),cases=rows),indent=2)+'\n');print(json.dumps(rows))
