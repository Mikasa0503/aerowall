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
for c in contacts:
 if c['first_episode_active']:groups[c['step'],c['env_id']].append(c)
rows=[];mass=float(cfg['task']['ball_mass']);dt=float(z['dt']);gravity=np.array(cfg['sim']['gravity'])*dt
for (step,env),items in groups.items():
 if len(items)!=1 or step<1:continue
 c=items[0];v0=z['ball_velocity'][step-1,env,0,:3];v1=z['ball_velocity'][step,env,0,:3]
 projected=abs(float((mass*(v1-v0-gravity))@np.array(c['normal'])));impulse=abs(c['normal_impulse'])
 distance=min(np.linalg.norm(np.array(c['position'])-z['ball_position'][k,env,0]) for k in [step-1,step])
 rows.append({'step':step,'env_id':env,'agent_id':c['agent_id'],'normal_impulse':impulse,
              'momentum_normal_component':projected,'relative_error':abs(projected-impulse)/impulse,
              'closest_endpoint_ball_center_distance':float(distance)})
r={'sample_count':len(rows),'max_relative_normal_momentum_error':max(x['relative_error'] for x in rows),
   'samples':rows,'scope':'single normal-contact sample at active first-episode step; no claim to include tangential or unfiltered simultaneous contacts'}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='samples'}))
