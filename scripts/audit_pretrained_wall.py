"""Reconstruct native body-wall-body chains from saved GPU impulses, not hit proxies."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.report.read_text());assert r['status']=='passed'
assert r['initial_same_side_clearance_checked']
s=json.loads(a.report.with_suffix('.tensor-contacts.json').read_text())
z=np.load(a.report.with_suffix('.trajectory.npz'));assert hashlib.sha256(a.report.with_suffix('.trajectory.npz').read_bytes()).hexdigest()==r['trajectory_sha256']
initial_ball=np.array(r['initial_ball_position']);initial_drone=np.array(r['initial_drone_position'])
fixture=r['wall_fixture'];axis=int(np.argmin(fixture['dimensions']))
center=float(fixture['center'][axis]);half=float(fixture['dimensions'][axis])/2
faces=[center-half,center+half]
side=np.sign(initial_drone[:,axis]-center)
assert ((initial_drone[:,axis]-center)*side>.5+half).all()
assert ((initial_ball[:,axis]-center)*side>.1+half).all()
origins=np.array(r['env_origins']);rows=[];bad=[]
for outcome in r['outcomes']:
 i=outcome['scenario_id'];groups={}
 for x in s:
  if x['env_id']!=i or not x['first_episode_active']:continue
  # Exclude terminal held-action sample; upstream termination may be an invalid hit.
  if x['step']>=outcome['steps']-1:continue
  kind=x.get('kind','body')
  if kind=='wall':
   point=np.array(x['position'])-origins[i]
   if min(abs(point[axis]-f) for f in faces)>.15 or abs(x['normal'][axis])<.8:
    bad.append({'scenario':i,'sample':x});continue
  assert abs(x['normal_impulse'])>1e-8
  groups.setdefault(x['step'],set()).add(kind)
 phase='wait';previous=set();body=walls=returns=ambiguous=0;chain=[]
 for step in range(outcome['steps']-1):
  contacts=groups.get(step,set());entered=contacts-previous;previous=contacts
  if len(entered)>1:phase='wait';ambiguous+=1;continue
  for kind in entered:
   chain.append({'step':step,'time':(step+1)*r['dt'],'kind':kind})
   if kind=='body':
    body+=1
    if phase=='return':returns+=1
    phase='outbound'
   elif kind=='wall':
    walls+=1;phase='return' if phase=='outbound' else 'wait'
 rows.append({'scenario_id':i,'body_entries':body,'wall_entries':walls,'returns':returns,'ambiguous_intervals':ambiguous,'chain':chain})
assert not bad,bad[:1]
result={'status':'passed','audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'report':str(a.report),'report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),
 'scope':'16 native Iris/PRT first episodes; GPU base-link contacts, not AeroWall legal cap. Policy-step entry debounce; terminal intervals excluded, substep ordering unavailable. Invalid initial overlaps excluded by assertions.',
 'scenarios':len(rows),'body_contact_scenarios':sum(x['body_entries']>0 for x in rows),
 'wall_contact_scenarios':sum(x['wall_entries']>0 for x in rows),'return_scenarios':sum(x['returns']>0 for x in rows),'total_returns':sum(x['returns'] for x in rows),'rows':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
