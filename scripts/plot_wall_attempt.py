"""Plot one actual failed/successful wall attempt without stitching resets."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--scenarios',type=Path,required=True)
p.add_argument('--scenario',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
i=a.scenario;outcome=r['outcomes'][i]
z=np.load(a.evaluation.with_suffix('.trajectory.npz'))
origin=np.array(json.loads(a.scenarios.read_text())['env_origins'][i])
events=[json.loads(x) for x in a.evaluation.with_suffix('.events.jsonl').read_text().splitlines()]
events=[e for e in events if e['env_id']==i]
wall=next(e for e in events if e['kind']=='wall' and e['credited'])
caps=[e for e in events if e.get('wall_cap_credit')]
launch=max((e for e in caps if e['time']<wall['time']),key=lambda e:e['time'])
returned=next((e for e in caps if e['time']>wall['time']),None)
active=z['active_before'][:,i].astype(bool)
t=(np.arange(len(active))+1)*float(z['dt'])
keep=active & (t>=max(0,launch['time']-.3))
t=t[keep];ball=z['ball_position'][keep,i]-origin;drone=z['drone_position'][keep,i]-origin
bv=z['ball_velocity'][keep,i];dv=z['drone_velocity'][keep,i];q=z['drone_quaternion_wxyz'][keep,i]
w,x,y,zz=q.T
normal=np.stack([2*(x*zz+w*y),2*(y*zz-w*x),1-2*(x*x+y*y)],axis=-1)
fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
ax[0,0].plot(ball[:,0],ball[:,2],label='ball');ax[0,0].plot(drone[:,0],drone[:,2],label='drone')
front=r['wall_fixture']['center'][0]-r['wall_fixture']['dimensions'][0]/2
ax[0,0].axvline(front,color='black',ls='--',label='wall front')
for event,label,color in [(launch,'legal cap','green'),(wall,'wall','purple')]:
    weights=np.array([s['impulse'] for s in event['points']]);points=np.array([s['point'] for s in event['points']])
    point=np.average(points,axis=0,weights=weights)-origin
    ax[0,0].scatter(point[0],point[2],color=color);ax[0,0].annotate(label,(point[0],point[2]))
ax[0,0].set(xlabel='local x (m)',ylabel='height (m)',title='Actual trajectory from one episode');ax[0,0].legend()
for j,label in enumerate(['ball vx','ball vy','ball vz']):ax[0,1].plot(t,bv[:,j],label=label)
ax[0,1].set(ylabel='m/s',title='Outgoing and reflected ball velocity');ax[0,1].legend()
for j,label in enumerate(['normal x','normal y','normal z']):ax[1,0].plot(t,normal[:,j],label=label)
ax[1,0].set(ylabel='unit vector',title='Bat normal from rigid body orientation');ax[1,0].legend()
ax[1,1].plot(t,np.linalg.norm(dv[:,3:],axis=-1),label='angular speed (rad/s)')
ax[1,1].plot(t,np.linalg.norm(dv[:,:3],axis=-1),label='linear speed (m/s)')
ax[1,1].set(title='Body recovery diagnostics');ax[1,1].legend()
for axis in [ax[0,1],ax[1,0],ax[1,1]]:
    axis.axvline(launch['time'],color='green',ls='--');axis.axvline(wall['time'],color='purple',ls='--')
    axis.set_xlabel('episode time (s)');axis.grid(alpha=.2)
fig.suptitle(f"Scenario {i}: cap -> wall -> {outcome['reason']}; scored rallies = {outcome['rallies']}")
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);plt.close(fig)
summary={'scenario_id':i,'launch_time':launch['time'],'wall_time':wall['time'],
         'next_legal_cap_time':None if returned is None else returned['time'],'outcome':outcome,
         'launch_ball_velocity_before':launch.get('ball_velocity_before'),
         'launch_ball_velocity_after':launch.get('ball_velocity_after'),
         'launch_bat_normal':launch.get('bat_normal'),
         'launch_bat_contact_point_velocity':launch.get('bat_contact_point_velocity'),
         'last_ball_position_local':ball[-1].tolist(),'last_drone_position_local':drone[-1].tolist(),
         'scope':'Actual first episode; absence of next legal cap is not successful recovery; no injected states'}
a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
