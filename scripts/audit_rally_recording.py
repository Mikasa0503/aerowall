"""Replay actual SingleJuggle contact lifecycles through the WallRally counter.

A negative control: thousands of vertical bat impacts must never become wall
rallies. This preserves prior geometric labels and audits event qualification,
not the geometric classifier or the yet-unimplemented WallRally environment.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aerowall.rally_events import Kind,RallyBatch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.report.read_text());assert r['status']=='passed'
event_path=a.report.with_suffix('.events.jsonl');events=[json.loads(s) for s in event_path.read_text().splitlines()]
batch=RallyBatch(len(r['outcomes']));group=defaultdict(list)
for e in events:group[e['step'],e['scenario_id']].append(e)
credits=[0]*len(batch.states);expected=[0]*len(batch.states);total=0
for step in range(max(o['steps'] for o in r['outcomes'])):
 for i in range(len(batch.states)):
  impacts=[]
  for e in group[step,i]:
   paths=(e['actor0'],e['actor1']);pair=(e['collider0'],e['collider1'])
   if any(v.endswith('/bat') for v in paths):kind=Kind.CAP if e['provisional_top_contact'] else Kind.NON_CAP
   elif any('GroundPlane' in v for v in paths):kind=Kind.BALL_GROUND
   elif any('/Air_0/' in v for v in paths):kind=Kind.BALL_BODY
   else:raise RuntimeError(f'Unrecognized actual contact pair {pair}')
   edge=e['type'].rsplit('_',1)[-1].lower()
   impact=batch.ledgers[i].observe(pair,edge,kind,e.get('impulse_norm',0.),e.get('position'))
   if impact is not None:
    impacts.append(impact)
    credits[i]+=int(impact.kind==Kind.CAP)
   expected[i]+=int(e.get('credited_top_impact',False));total+=1
  batch.states[i].advance(impacts,(0.,0.,0.))
assert credits==expected,[(i,x,y) for i,(x,y) in enumerate(zip(credits,expected)) if x!=y]
assert all(s.rallies==0 and s.wall_hits==0 for s in batch.states)
summary={'status':'passed','source_report':str(a.report),'source_report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),
         'source_events_sha256':hashlib.sha256(event_path.read_bytes()).hexdigest(),
         'rally_module_sha256':hashlib.sha256((ROOT/'aerowall/rally_events.py').read_bytes()).hexdigest(),
         'event_count':total,'cap_credits':sum(credits),'per_scenario_cap_credits':credits,
         'all_prior_cap_credits_reproduced':True,'false_wall_rallies':sum(s.rallies for s in batch.states),
         'states':[asdict(s) for s in batch.states],
         'scope':'Actual recorded SingleJuggle negative control; no synthetic walls or full-rally success claim'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['states','per_scenario_cap_credits']}))
