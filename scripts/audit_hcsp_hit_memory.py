"""Check every active step's role-hit memory against recorded physical events."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.report.read_text());assert r['status']=='passed' and r['hit_memory']
z=np.load(r['trajectory']);assert hashlib.sha256(Path(r['trajectory']).read_bytes()).hexdigest()==r['trajectory_sha256']
checks=0;updates=0;clears=0
for env in range(z['active'].shape[1]):
 expected=np.zeros(3,bool);last=True
 for step in range(z['active'].shape[0]):
  if not z['active'][step,env]:break
  body,wall=z['contact_entry'][step,env]
  if body:expected[int(z['executed_role'][step,env])-2]=True;last=False;updates+=1
  if wall:expected[:]=False;last=True;clears+=1
  assert np.array_equal(expected,z['already_hit'][step,env]),(step,env)
  assert last==bool(z['last_hit_side'][step,env]),(step,env)
  checks+=1
out={'status':'passed','report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),
 'checked_active_steps':checks,'body_event_updates':updates,'wall_event_rearms':clears,
 'scope':'Discrete role memory and last hitter side match physical contact events; not contact legality'}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
