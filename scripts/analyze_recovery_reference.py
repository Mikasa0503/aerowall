"""Replay receiving-reference coverage on actual scored actor observations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aerowall.learning.return_reference import predict_return_reference,recovery_target_from_observation
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
assert r['actor_observation_record']['ball_velocity_delay_steps']==0
z=np.load(a.evaluation.with_suffix('.trajectory.npz'))
assert hashlib.sha256(a.evaluation.with_suffix('.trajectory.npz').read_bytes()).hexdigest()==r['trajectory_sha256']
obs=torch.from_numpy(z['actor_observation_before']);active=torch.from_numpy(z['active_before'].astype(bool))
phase=obs[...,40:43].argmax(-1)
gate=active&(((phase==1)&(obs[...,21]>.5))|(phase==2))
counts=Counter();rows=[];task=r['wall_task'];home=task.get('recovery_home_position',[0,0,1])
for i in range(obs.shape[1]):
 o=obs[:,i][gate[:,i]]
 if not len(o):continue
 target,t,valid,reflect=predict_return_reference(o,contact_height=home[2]+.123)
 reference,accepted=recovery_target_from_observation(o,home,task['bounds_low'],task['bounds_high'])
 postwall=(o[:,40:43].argmax(-1)==2)|reflect
 categories={'invalid_ballistic_prediction':~valid,'height_crossing_before_wall':valid&~postwall,
             'outside_reference_bounds':valid&postwall&~accepted,'accepted_reference':accepted}
 assert sum(int(m.sum()) for m in categories.values())==len(o)
 row={'scenario_id':i,'recovery_decision_steps':len(o),**{k:int(m.sum()) for k,m in categories.items()}}
 counts.update({k:v for k,v in row.items() if k!='scenario_id'})
 if accepted.any():
  distance=torch.linalg.vector_norm(reference[accepted]-o[accepted,:3],dim=-1)
  row['accepted_reference_distance_median']=float(distance.median())
  row['predicted_time_median_seconds']=float(t[accepted].median())
 rows.append(row)
result={'evaluation':str(a.evaluation),'evaluation_sha256':hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'aerowall/learning/return_reference.py').read_bytes()).hexdigest(),
        'scope':'Offline fixed-prior 0.8 reference coverage at pre-action decisions; not post-action reward counts, reachability proof or policy evaluation',
        'counts':dict(counts),'rows':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
