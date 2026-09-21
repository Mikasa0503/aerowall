"""Independent first-episode impulse-chain audit for the one-body HCSP adapter."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.report.read_text());assert r['status']=='passed'
assert r['num_drones']==1 and r['physical_drone_shape']==[len(r['outcomes']),1]
assert len(r['checkpoints'])==27
assert len({x['role'] for x in r['checkpoints']})==27
root=Path(__file__).resolve().parents[1]
for c in r['checkpoints']:
 assert hashlib.sha256((root/'third_party/HCSP'/c['file']).read_bytes()).hexdigest()==c['sha256']
t=Path(r['trajectory']);assert hashlib.sha256(t.read_bytes()).hexdigest()==r['trajectory_sha256']
z=np.load(t);active=z['active'];imp=z['contact_impulse'];entry=z['contact_entry']
assert imp.shape==(*active.shape,2) and np.isfinite(imp).all()
assert z['physical_drone'].shape[2]==1 and z['executed_action'].shape[2:]==(1,4)
assert np.isfinite(z['physical_drone']).all() and np.isfinite(z['executed_action']).all()
assert set(np.unique(z['executed_role'])).issubset({2,3,4})
assert np.array(r['initial_ball'])[:,:,0].min()>.2
assert np.array(r['initial_drone'])[:,:,0].min()>.6
outcomes=[]
for env in range(active.shape[1]):
 phase=0;returns=0;body_count=0;wall_count=0;last=np.zeros(2,bool);events=[]
 for step in range(active.shape[0]):
  if not active[step,env]:break
  live=imp[step,env]>1e-8;new=live&~last;last=live
  assert np.array_equal(new,entry[step,env])
  body,wall=new
  body_count+=int(body);wall_count+=int(wall)
  if body or wall:events.append({'step':step,'body':bool(body),'wall':bool(wall),'role':int(z['executed_role'][step,env])})
  if body and wall:phase=0;continue
  if body:
   if phase==2:returns+=1
   phase=1
  elif wall:
   phase=2 if phase==1 else 0
 match=next(x for x in r['outcomes'] if x['env']==env)
 assert match['wall_returns']==returns
 outcomes.append({'env':env,'body_entries':body_count,'wall_entries':wall_count,'returns':returns,'events':events})
result={'status':'passed','scope':'Single physical drone, full checkpoint library with explicit role/observation adapter; body-wall-body only',
 'report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),'episodes':len(outcomes),
 'episodes_body_contact':sum(x['body_entries']>0 for x in outcomes),
 'episodes_wall_contact':sum(x['wall_entries']>0 for x in outcomes),
 'episodes_wall_return':sum(x['returns']>0 for x in outcomes),'maximum_returns':max(x['returns'] for x in outcomes),
 'executed_role_steps':{str(i):int(((z['executed_role']==i)&active).sum()) for i in (2,3,4)},
 'outcomes':outcomes,'limitations':['Body contact is not yet classified as legal racket contact.',
 'Single-body observation replication and role arbitration differ from original six-drone HCSP.',
 'Only first episodes from this initial-state distribution; not a general capability claim.']}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='outcomes'},indent=2))
