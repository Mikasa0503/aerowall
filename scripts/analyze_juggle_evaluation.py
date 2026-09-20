"""Summarize fixed-development evaluation and plot a median-duration episode."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
r=json.loads(a.report.read_text())
events=[json.loads(s) for s in a.report.with_suffix('.events.jsonl').read_text().splitlines()]
z=np.load(a.report.with_suffix('.trajectory.npz'))
outcomes=sorted(r['outcomes'],key=lambda row:(row['steps'],row['scenario_id']))
selected=outcomes[len(outcomes)//2]; index=selected['scenario_id']; n=selected['steps'];dt=float(z['dt'])
hits=[e for e in events if e['scenario_id']==index and e.get('credited_top_impact',e['new_entry'] and e['provisional_top_contact'])]
first=hits[0]['step'] if hits else None
next_hit=hits[1] if len(hits)>1 else None
if next_hit is not None:n=min(n,next_hit['step']+3)
t=np.arange(1,n+1)*dt
cycle_label='next cap impact observed' if next_hit is not None else 'no return observed before episode ended'
first_departure=next((e for e in events if e['scenario_id']==index and 'CONTACT_LOST' in e['type'] and first is not None and e['time']>hits[0]['time'] and next_hit is not None and e['time']<next_hit['time'] and {e['actor0'],e['actor1']}=={hits[0]['actor0'],hits[0]['actor1']}),None)
ball=z['ball_position'][:n,index]; drone=z['drone_position'][:n,index]
vz=z['ball_velocity'][:n,index,2]
apex=next((k for k in range(first+1,n) if vz[k]<=0),n-1) if first is not None else n-1
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
axes[0,0].plot(t,ball[:,2],label='Ball height');axes[0,0].plot(t,drone[:,2],label='Drone height');axes[0,0].set_ylabel('World height (m)')
axes[0,1].plot(t,vz,label='Ball vertical velocity');axes[0,1].plot(t,z['bat_velocity'][:n,index,2],label='Bat COM vertical velocity');axes[0,1].set_ylabel('Velocity (m/s)')
for hit in hits:
 if hit['step']>=n:continue
 axes[0,1].scatter(hit['time'],hit['bat_contact_point_velocity'][2],marker='x',color='black')
for j,name in enumerate('xyz'):
 axes[1,0].plot(t,z['bat_normal'][:n,index,j],label=f'Bat normal {name}')
 axes[1,1].plot(t,z['drone_velocity'][:n,index,j+3],label=f'Drone angular velocity {name}')
axes[1,0].set_ylabel('Unit normal');axes[1,1].set_ylabel('Angular velocity (rad/s)')
for ax in axes.flat:
 if first is not None:
  contact_time=hits[0]['time']
  departure_time=first_departure['time'] if first_departure else contact_time+r.get('physics_dt',dt)
  for left,right,color in [(0,contact_time,'#edf2fa'),(contact_time,departure_time,'#ffb56b'),(departure_time,(apex+1)*dt,'#e6f4df'),((apex+1)*dt,n*dt,'#f7e8ed')]:
   ax.axvspan(left,right,color=color,alpha=.7)
  for hit in hits:
   if hit['step']<n:ax.axvline(hit['time'],color='#b6591a',ls=':',lw=.8)
 ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best');ax.set_xlabel('Time (s)')
fig.suptitle(f'Development checkpoint: {r["trained_environment_frames"]:,} transitions | median-duration scenario {index}\nApproach / first contact / outgoing-to-apex / return observation ({cycle_label})',fontsize=11)
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output.with_suffix('.png'),dpi=170);plt.close(fig)
reasons=collections.Counter()
for row in outcomes:
 for k,v in row['upstream_stats'].items():
  if k not in ['num_true_hits','wrong_hit','truncated'] and v:reasons[k]+=1
summary={'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],'frames':r['trained_environment_frames'],
         'scenario_sha256':r['scenario_sha256'],'scenario_count':100,'selected_scenario':selected,
         'selection':'upper median episode duration; tie-break scenario ID',
         'physical_ball_bat_entry_distribution':dict(collections.Counter(r['ball_bat_entry_distribution'])),
         'provisional_top_entry_distribution':dict(collections.Counter(r['provisional_top_entry_distribution'])),
         'five_physical_ball_bat_rate':sum(v>=5 for v in r['ball_bat_entry_distribution'])/100,
         'failure_flags':dict(reasons),'selected_top_events':hits,
         'policy_sample_dt':dt,'physics_dt':r.get('physics_dt',dt),
         'first_contact_time':None if not hits else hits[0]['time'],
         'first_departure_time':None if first_departure is None else first_departure['time'],
         'five_legal_cap_rate':sum(v>=5 for v in r['provisional_top_entry_distribution'])/100,
         'first_contact_step':first,'first_post_contact_apex_step':apex,
         'episode_time_limit':bool(selected['upstream_stats'].get('truncated',0)),
         'contact_to_next_contact_seconds':None if next_hit is None else next_hit['time']-hits[0]['time'],
         'leave_to_next_contact_seconds':None if first_departure is None else next_hit['time']-first_departure['time'],
         'hover_recovery_time':'not established; no frozen hover-convergence criterion',
         'scope':'development first-cycle observation; not complete HCSP/wall-task action analysis'}
a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='selected_top_events'},indent=2))
