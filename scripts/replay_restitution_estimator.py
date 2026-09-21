"""Replay estimator on exact saved actor inputs; simulator events are audit-only.

The estimator receives no event rows or material labels. Independent comparison
below is possible only for unambiguous saved wall intervals with kinematics.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aerowall.learning.restitution_estimator import RestitutionEstimator
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
z=np.load(a.evaluation.with_suffix('.trajectory.npz'))
obs=z['actor_observation_before'];active=z['active_before'].astype(bool)
delay=r['actor_observation_record']['ball_velocity_delay_steps']
e=RestitutionEstimator(obs.shape[1],'cpu',velocity_delay_steps=delay)
rows=[];counts=Counter()
for step in range(len(obs)):
    features=e.update(torch.from_numpy(obs[step]),active=torch.from_numpy(active[step]))
    for i in torch.nonzero(e.reason).flatten().tolist():
        reason=e.REASONS[int(e.reason[i])];counts[reason]+=1
        ratio=float(e.last_ratio[i])
        rows.append({'scenario_id':i,'observation_step':step,'reason':reason,
                     'ratio':ratio if np.isfinite(ratio) else None,
                     'features':features[i].tolist(),
                     'collision_policy_step':step-delay-1})
# Only now load simulator events for comparison; none enter estimator.update.
events=[json.loads(line) for line in a.evaluation.with_suffix('.events.jsonl').read_text().splitlines()]
by_interval={}
for event in events:
    if event['credited']:
        by_interval.setdefault((event['env_id'],event['policy_step']),[]).append(event)
errors=[];uncompared=Counter()
for row in rows:
    if row['reason']!='accepted':continue
    es=by_interval.get((row['scenario_id'],row['collision_policy_step']),[])
    if len(es)!=1 or es[0]['kind']!='wall':
        uncompared['not_single_wall_only_interval']+=1;continue
    wall=es[0]
    if 'ball_velocity_before' not in wall or 'ball_velocity_after' not in wall:
        uncompared['missing_wall_kinematics']+=1;continue
    vin=wall['ball_velocity_before'][0];vout=wall['ball_velocity_after'][0]
    if vin<=0 or vout>=0:
        uncompared['wall_kinematics_wrong_sign']+=1;continue
    reference=-vout/vin;error=abs(row['ratio']-reference)
    row['audit_only_microstep_ratio']=reference;row['absolute_ratio_error']=error;errors.append(error)
result={'status':'passed','evaluation':str(a.evaluation),
        'evaluation_sha256':hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
        'estimator_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'aerowall/learning/restitution_estimator.py').read_bytes()).hexdigest(),
        'scope':'Offline observable-history estimator replay; no learned estimator policy or randomized-wall validation',
        'velocity_delay_steps':delay,'scored_observations':int(active.sum()),
        'diagnostic_counts':dict(counts),'episodes_with_estimate':int((e.samples>0).sum()),
        'sample_support_caveat':'count/(count+1) is heuristic support, not a calibrated probability',
        'microstep_comparisons':len(errors),'uncompared':dict(uncompared),
        'maximum_absolute_ratio_error':max(errors) if errors else None,
        'rows':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
