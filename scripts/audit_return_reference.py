"""Compare observable ballistic predictions with later recorded ball crossings.

Future samples and contacts are used only for this offline error measurement.
All wall episodes remain in the denominator, including unavailable crossings.
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
from aerowall.learning.return_reference import predict_return_reference
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--scenarios',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
assert r['actor_observation_record']['ball_velocity_delay_steps']==0,'Delay requires separate prediction state alignment'
assert hashlib.sha256(a.scenarios.read_bytes()).hexdigest()==r['scenario_sha256']
roster=json.loads(a.scenarios.read_text());origins=np.asarray(roster['env_origins'])
ball_radius=float(roster['config_task']['ball_radius']);height=1.+.083+ball_radius
z=np.load(a.evaluation.with_suffix('.trajectory.npz'));obs=z['actor_observation_before'];dt=float(z['dt'])
events=[json.loads(l) for l in a.evaluation.with_suffix('.events.jsonl').read_text().splitlines()]
rows=[]
for outcome in r['outcomes']:
    if not outcome['wall_hits']:continue
    i=outcome['scenario_id'];terminal=outcome.get('physics_steps',outcome['steps'])*r['physics_dt']
    es=[e for e in events if e['env_id']==i and e['credited']]
    walls=[e for e in es if e['kind']=='wall' and e['time']<terminal-1e-8 and not e.get('physics_step_illegal_priority',False)]
    if not walls:continue
    wall=walls[0]
    next_contact=min([e['time'] for e in es if e['time']>wall['time'] and e['kind'] in ('cap','wall','ball_body','ball_ground','non_cap')]+[terminal])
    for mode,s in [('pre_wall',wall['policy_step']),('post_wall',wall['policy_step']+1)]:
        row={'scenario_id':i,'mode':mode,'observation_step':s,'wall_time':wall['time']};rows.append(row)
        if s>=len(obs) or not z['active_before'][s,i]:row['status']='no_active_observation';continue
        target,t,valid,reflected=predict_return_reference(torch.from_numpy(obs[s,i:i+1]),ball_radius=ball_radius,contact_height=height)
        if not bool(valid[0]):row['status']='invalid_prediction';continue
        if mode=='pre_wall' and not bool(reflected[0]):row['status']='height_crossing_predicted_before_wall';continue
        row['predicted_crossing_time']=s*dt+float(t[0]);row['predicted_point']=target[0].tolist()
        previous=obs[s,i,18:21];previous_time=s*dt;crossing=None
        for k in range(s,len(obs)):
            now=(k+1)*dt
            if now>=next_contact-1e-8:break
            current=z['ball_position'][k,i]-origins[i]
            if previous[2]>=height and current[2]<=height and now>wall['time']:
                alpha=(previous[2]-height)/(previous[2]-current[2])
                crossing_time=previous_time+alpha*(now-previous_time)
                if crossing_time>wall['time']:
                    crossing=(crossing_time,previous+alpha*(current-previous));break
            previous=current;previous_time=now
        if crossing is None:row['status']='no_uninterrupted_observed_crossing';continue
        time,point=crossing
        row.update(status='compared',observed_crossing_time=float(time),observed_point=point.tolist(),
                   xy_error_m=float(np.linalg.norm(np.asarray(row['predicted_point'])[:2]-point[:2])),
                   time_error_s=abs(row['predicted_crossing_time']-float(time)))
summary={}
for mode in ('pre_wall','post_wall'):
    subset=[x for x in rows if x['mode']==mode];errors=[x['xy_error_m'] for x in subset if x['status']=='compared']
    summary[mode]={'wall_scenarios':len(subset),'statuses':dict(Counter(x['status'] for x in subset)),
                   'xy_error_quantiles_50_90_max_m':np.quantile(errors,[.5,.9,1.]).tolist() if errors else None}
result={'status':'passed','scope':'Offline prediction errors conditional on uninterrupted height crossings; not interception success or reachability',
        'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'crossing_interpolation':'Linear interpolation between 20 ms samples; brackets ending at or after the next contact/termination excluded',
        'evaluation':str(a.evaluation),'evaluation_sha256':hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
        'predictor_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'aerowall/learning/return_reference.py').read_bytes()).hexdigest(),
        'ball_center_reference_height':height,'restitution_prior':.8,'all_episodes':len(r['outcomes']),
        'summary':summary,'rows':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
