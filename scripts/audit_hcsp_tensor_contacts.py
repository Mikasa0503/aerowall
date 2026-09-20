"""Compare single GPU normal-contact samples with measured ball momentum change."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import yaml
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
z=np.load(a.report.with_suffix('.trajectory.npz'));cfg=yaml.safe_load(a.report.with_suffix('.yaml').read_text())
contacts=json.loads(a.report.with_suffix('.tensor-contacts.json').read_text());groups=collections.defaultdict(list)
events=json.loads(a.report.with_suffix('.events.json').read_text())
for c in contacts:
 if c['first_episode_active']:groups[c['step'],c['env_id']].append(c)
rows=[];mass=float(cfg['task']['ball_mass']);dt=float(z['dt']);gravity=np.array(cfg['sim']['gravity'])*dt
for (step,env),items in groups.items():
 if len(items)!=1 or step<1:continue
 c=items[0];v0=z['ball_velocity'][step-1,env,0,:3];v1=z['ball_velocity'][step,env,0,:3]
 projected=abs(float((mass*(v1-v0-gravity))@np.array(c['normal'])));impulse=abs(c['normal_impulse'])
 ball_path=f'/World/envs/env_{env}/ball';body_path=f'/World/envs/env_{env}/Iris_{c["agent_id"]}/base_link'
 other_pairs=sorted({tuple(sorted([e['actor0'],e['actor1']])) for e in events
                     if e['step']==step and ball_path in [e['actor0'],e['actor1']]
                     and body_path not in [e['actor0'],e['actor1']]
                     and 'CONTACT_LOST' not in e['type'] and e['points']})
 distance=min(np.linalg.norm(np.array(c['position'])-z['ball_position'][k,env,0]) for k in [step-1,step])
 rows.append({'step':step,'env_id':env,'agent_id':c['agent_id'],'normal_impulse':impulse,
              'momentum_normal_component':projected,'relative_error':abs(projected-impulse)/impulse,
              'closest_endpoint_ball_center_distance':float(distance),
              'other_reported_ball_pairs':other_pairs,'single_pair_candidate':not other_pairs})
single=[x for x in rows if x['single_pair_candidate']]
r={'single_pair_candidate_count':len(single),
   'max_relative_normal_momentum_error_single_pair_candidates':max((x['relative_error'] for x in single),default=None),
   'sample_count':len(rows),'max_relative_normal_momentum_error':max((x['relative_error'] for x in rows),default=None),
   'samples':rows,'scope':'Raw error retains all steps with one GPU base-link sample. Candidate subset excludes other simultaneous ball pairs reported by CPU headers; missing headers and tangential forces remain outside this audit.'}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='samples'}))
