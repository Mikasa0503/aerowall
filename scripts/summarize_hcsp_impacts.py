"""Export measured HCSP role impacts, keeping sample timing and body-contact scope explicit."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
r=json.loads(a.report.read_text())
assert r['status']=='passed'
z=np.load(a.report.with_suffix('.trajectory.npz'))
role,agent={'Attack_hover':('Att',1),'Set_hover':('SecPass',0),
            'Receive_hover':('FirstPass',1),'Pass_hover':('FirstPass',2)}[r['task']]
dt=float(z['dt']);rows=[]
for c in json.loads(a.report.with_suffix('.tensor-contacts.json').read_text()):
 if not c['first_episode_active'] or c['agent_id']!=agent:continue
 k=c['step'];i=c['env_id']
 linear=np.array(c['translation_velocity']);angular=np.array(c['rotational_velocity'])
 velocity=np.array(c['contact_point_velocity'])
 assert np.allclose(linear+angular,velocity,atol=1e-5)
 assert np.isfinite(velocity).all()
 rows.append({**c,'sample_before_time_s':k*dt if k else None,'sample_after_time_s':(k+1)*dt,
              'ball_velocity_before':z['ball_velocity'][k-1,i,0,:3].tolist() if k else None,
              'ball_velocity_after':z['ball_velocity'][k,i,0,:3].tolist(),
              'body_angular_velocity_after':z['body_velocity'][k,i,agent,3:].tolist(),
              'body_com_after':z['body_com_world'][k,i,agent].tolist(),
              'linear_speed':float(np.linalg.norm(linear)),
              'rotation_contribution_speed':float(np.linalg.norm(angular)),
              'contact_point_speed':float(np.linalg.norm(velocity))})
summary={'initial_state_distribution':r.get('initial_state_distribution','original shell configuration'), 'task':r['task'],'role':role,'agent_index':agent,'dt':dt,'source_report':str(a.report),
         'source_commit':r['source_commit'],'trajectory_sha256':r['trajectory_sha256'],
         'sample_count':len(rows),'scenario_count':len({x['env_id'] for x in rows}),
         'scope':'Positive normal-impulse GPU ball/base-link samples in first episodes. HCSP visual bat is not a separate physical bat; these are not AeroWall legal-cap credits.',
         'timing':'Ball velocities bracket one simulation step, not instantaneous collision limits; body contact-point velocity uses the sampled post-step COM and angular velocity.',
         'samples':rows}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='samples'}))
