"""Describe post-wall approach from real policy-rate trajectory samples.

Distances use ball and drone rigid-body origins, not the bat collision surface.
They diagnose gross interception error; they do not classify legal contacts or
prove that a substep collision was absent. Excludes terminal-impact samples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
z=np.load(a.evaluation.with_suffix('.trajectory.npz'))
events=[json.loads(l) for l in a.evaluation.with_suffix('.events.jsonl').read_text().splitlines()]
t=(np.arange(len(z['active_before']))+1)*float(z['dt']);rows=[];missing=[]
for o in r['outcomes']:
    i=o['scenario_id'];terminal=o.get('physics_steps',o['steps'])*r['physics_dt']
    walls=[e for e in events if e['env_id']==i and e['credited'] and e['kind']=='wall' and e['time']<terminal]
    if not walls:continue
    wall=walls[0]
    caps=[e for e in events if e['env_id']==i and e.get('wall_cap_credit') and wall['time']<e['time']<terminal]
    stop=min(terminal,caps[0]['time'] if caps else terminal)
    mask=z['active_before'][:,i].astype(bool)&(t>wall['time'])&(t<stop-1e-8)
    indices=np.flatnonzero(mask)
    if not len(indices):
        missing.append(i);continue
    delta=z['ball_position'][indices,i]-z['drone_position'][indices,i]
    distance=np.linalg.norm(delta,axis=-1);j=int(np.argmin(distance));k=int(indices[j])
    relative_velocity=z['ball_velocity'][k,i,:3]-z['drone_velocity'][k,i,:3]
    radial=float(np.dot(relative_velocity,delta[j])/max(float(distance[j]),1e-9))
    rows.append({'scenario_id':i,'reason':o['reason'],'rallies':o['rallies'],
                 'wall_time':wall['time'],'terminal_time':terminal,
                 'next_legal_cap_time':caps[0]['time'] if caps else None,
                 'precontact_sample_count':len(indices),'minimum_origin_distance_m':float(distance[j]),
                 'closest_sample_time':float(t[k]),'relative_ball_position_at_closest_m':delta[j].tolist(),
                 'relative_ball_velocity_at_closest_m_s':relative_velocity.tolist(),
                 'radial_relative_speed_at_closest_m_s':radial,
                 'closest_is_first_sample':j==0,'closest_is_last_sample':j==len(indices)-1,
                 'vertical_offset_min_max_m':[float(delta[:,2].min()),float(delta[:,2].max())]})
d=[x['minimum_origin_distance_m'] for x in rows]
result={'evaluation':str(a.evaluation),'report_sha256':hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Sampled ball-to-drone-origin approach before next cap or termination; not bat surface geometry or legal-contact verification',
        'analyzed_scenarios':len(rows),'wall_scenarios_without_preterminal_sample':missing,
        'minimum_origin_distance_quantiles_10_50_90_m':np.quantile(d,[.1,.5,.9]).tolist() if d else None,
        'rows':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
