"""Contact-bounded motion and declared hover-recovery diagnostic sensitivity.

These thresholds describe post-contact hover only. They are not source-paper
criteria, WallRally rewards, global attitude constraints, or flip termination.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CRITERIA={
 'strict':{'distance_m':.10,'speed_m_s':.25,'angular_speed_rad_s':.5,'up_z':.98,'hold_s':.3},
 'nominal':{'distance_m':.25,'speed_m_s':.5,'angular_speed_rad_s':1.,'up_z':.9,'hold_s':.3},
 'relaxed':{'distance_m':.50,'speed_m_s':1.,'angular_speed_rad_s':2.,'up_z':.8,'hold_s':.3}}

def stable_interval(mask,dt,hold,start):
    count=math.ceil(hold/dt)+1 # elapsed time between first and last sampled point
    for first in range(start,len(mask)-count+1):
        if mask[first:first+count].all():return first,first+count-1
    return None

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True);runs=[]
for path in a.reports:
 r=json.loads(path.read_text());assert r['status']=='passed'
 z=np.load(path.with_suffix('.trajectory.npz'));dt=float(z['dt']);cfg=yaml.safe_load(path.with_suffix('.yaml').read_text())
 role,agent={'Set_hover':('SecPass',0),'Receive_hover':('FirstPass',1),'Pass_hover':('FirstPass',2),'Attack_hover':('Att',1)}[r['task']]
 target=np.array(cfg['task'][f'{role}_hover_pos_after_hit']);origins=np.array(r['env_origins'])
 contacts=json.loads(path.with_suffix('.tensor-contacts.json').read_text());events=json.loads(path.with_suffix('.events.json').read_text());rows=[];cache={}
 for outcome in r['outcomes']:
  i=outcome['scenario_id'];n=outcome['steps'];times=(np.arange(n)+1)*dt
  cs=[c for c in contacts if c['env_id']==i and c['agent_id']==agent and c['first_episode_active'] and abs(c['normal_impulse'])>1e-6]
  pos=z['drone_position'][:n,i,agent]-origins[i];dist=np.linalg.norm(pos-target,axis=-1)
  speed=np.linalg.norm(z['body_velocity'][:n,i,agent,:3],axis=-1);omega=np.linalg.norm(z['body_velocity'][:n,i,agent,3:],axis=-1);up=z['body_normal'][:n,i,agent,2]
  hit=min((c['step'] for c in cs),default=None)
  pair={f'/World/envs/env_{i}/ball',f'/World/envs/env_{i}/Iris_{agent}/base_link'}
  departure=min((e['step'] for e in events if hit is not None and e['step']>=hit and 'CONTACT_LOST' in e['type'] and {e['actor0'],e['actor1']}==pair),default=None)
  recovery={}
  for name,c in CRITERIA.items():
   interval=None
   if departure is not None:
    mask=(dist<=c['distance_m'])&(speed<=c['speed_m_s'])&(omega<=c['angular_speed_rad_s'])&(up>=c['up_z'])
    interval=stable_interval(mask,dt,c['hold_s'],departure)
   recovery[name]={'recovered':interval is not None,'stable_interval_steps':interval,
                   'time_from_contact_departure_s':(interval[1]-departure)*dt if interval else None,
                   'censored_at_observation_end':interval is None and departure is not None,
                   'observed_after_departure_s':max(0,(n-1-departure)*dt) if departure is not None else None}
  row={'scenario_id':i,'steps':n,'physical_role_contact':hit is not None,'first_positive_contact_step':hit,
       'contact_departure_step':departure,'contact_point_samples':len(cs),'recovery':recovery,
       'end_stats':outcome['stats']}
  rows.append(row);cache[i]=(dist,speed,omega,up,cs)
 candidates=sorted([row for row in rows if row['physical_role_contact']],key=lambda row:(row['steps'],row['scenario_id']))
 selected=candidates[len(candidates)//2] if candidates else rows[len(rows)//2]
 i=selected['scenario_id'];n=selected['steps'];t=(np.arange(n)+1)*dt;hit=selected['first_positive_contact_step'];leave=selected['contact_departure_step'];dist,speed,omega,up,cs=cache[i]
 fig,axs=plt.subplots(3,2,figsize=(13,10));time_axes=[axs[0,0],axs[0,1],axs[1,0],axs[1,1],axs[2,1]]
 axs[0,0].plot(t,z['ball_position'][:n,i,0,2],label='Ball z');axs[0,0].plot(t,z['drone_position'][:n,i,agent,2],label=f'{role} z');axs[0,0].set_ylabel('World height (m)')
 for j,label in enumerate('xyz'):
  axs[0,1].plot(t,z['ball_velocity'][:n,i,0,j],label=f'Ball v{label}')
  axs[1,0].plot(t,z['body_normal'][:n,i,agent,j],label=f'Body up {label}')
  axs[1,1].plot(t,z['body_velocity'][:n,i,agent,j+3],label=f'World omega {label}')
 axs[0,1].set_ylabel('Ball velocity (m/s)');axs[1,0].set_ylabel('Unit vector');axs[1,1].set_ylabel('Angular velocity (rad/s)')
 if hit is not None:axs[0,1].set_xlim(max(0,hit*dt-.16),min(n*dt,(hit+1)*dt+.2))
 if cs:
  contact=min(cs,key=lambda c:c['step']);x=np.arange(3)
  for delta,key,label in [(-.25,'translation_velocity','v COM'),(0,'rotational_velocity','omega cross r'),(.25,'contact_point_velocity','v contact')]:
   axs[2,0].bar(x+delta,contact[key],width=.25,label=label)
  axs[2,0].set_xticks(x,['x','y','z']);axs[2,0].set_ylabel('First contact-point velocity (m/s)')
  relative=np.array(contact['position'])-origins[i]
  axs[2,0].set_title('Contact point relative to env origin: '+', '.join(f'{v:.3f}' for v in relative)+' m',fontsize=9)
 else:axs[2,0].text(.1,.5,'No measured role contact',transform=axs[2,0].transAxes)
 axs[2,0].legend(fontsize=8);axs[2,0].grid(alpha=.2)
 axs[2,1].plot(t,dist,label='Distance to hover target (m)');axs[2,1].plot(t,speed,label='Body speed (m/s)')
 axs[2,1].axhline(CRITERIA['nominal']['distance_m'],ls=':',color='C0');axs[2,1].axhline(CRITERIA['nominal']['speed_m_s'],ls=':',color='C1')
 interval=selected['recovery']['nominal']['stable_interval_steps']
 for ax in time_axes:
  if hit is not None:
   ax.axvspan(0,(hit+1)*dt,color='#dfeafa',alpha=.45)
   ax.axvline((hit+1)*dt,color='#a95318',ls='-',label='Positive contact')
  if leave is not None:
   ax.axvspan((hit+1)*dt,(leave+1)*dt,color='#f6cb95',alpha=.5)
   ax.axvline((leave+1)*dt,color='#676767',ls=':',label='Contact departed')
  if interval:ax.axvspan((interval[0]+1)*dt,(interval[1]+1)*dt,color='#84ba8a',alpha=.35)
  ax.set_xlabel('Time (s)');ax.grid(alpha=.2);ax.legend(fontsize=7)
 reset_label='default-reset diagnostic' if 'default' in path.name else 'original shell reset'
 fig.suptitle(f'{r["task"]} | {reset_label} | scenario {i}\nApproach / contact / outgoing ball and body recovery (the last two overlap in time)',fontsize=13)
 fig.tight_layout(rect=[0,0,1,.95]);figure=a.output.parent/(path.stem+'-motion.png');fig.savefig(figure,dpi=150);plt.close(fig)
 runs.append({'report':str(path),'task':r['task'],'scenario_count':len(rows),'actual_role_contact_count':len(candidates),
              'recovery_counts_all_scenarios':{key:sum(row['recovery'][key]['recovered'] for row in rows) for key in CRITERIA},
              'rows':rows,'selected_scenario':selected,'figure':str(figure),'initial_state_distribution':r.get('initial_state_distribution','original shell configuration')})
result={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'criteria_version':'hover-diagnostic-v1','criteria':CRITERIA,'runs':runs,
        'criteria_provenance':'Declared project diagnostics, not paper success criteria. Nominal is not tuned per skill; strict/relaxed sensitivity retained.',
        'scope':'Measured base-link contacts in original HCSP Iris/PRT. Hover recovery is separate from a learned WallRally catch. No-contact and censored episodes remain in all-scenario denominators.',
        'stage_semantics':'Approach precedes positive impact; contact departure is a measured LOST edge. Outgoing ball flight and drone recovery are concurrent observations, not a forced sequential controller.'}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in run.items() if k not in ['rows','selected_scenario']} for run in runs],indent=2))
