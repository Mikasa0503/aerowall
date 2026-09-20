"""Separate observed HCSP motion, upstream hit flags, and valid GPU contacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.report.read_text());d=np.load(a.report.with_suffix('.trajectory.npz'))
config=yaml.safe_load(a.report.with_suffix('.yaml').read_text())
target=np.array(config['task']['Att_hover_pos_after_hit']);origins=np.array(r['env_origins']);dt=float(d['dt'])
contacts=json.loads(a.report.with_suffix('.tensor-contacts.json').read_text());rows=[]
for outcome in r['outcomes']:
 i=outcome['scenario_id'];n=outcome['steps'];ph=d['phase_before_Att_hit'][:n,i].reshape(-1).astype(bool)
 pos=d['drone_position'][:n,i,1]-origins[i];dist=np.linalg.norm(pos-target,axis=-1);speed=np.linalg.norm(d['body_velocity'][:n,i,1,:3],axis=-1)
 samples=[v for v in contacts if v['env_id']==i and v['agent_id']==1 and v['first_episode_active']]
 rows.append({'scenario_id':i,'steps':n,'upstream_Att_hit':bool(outcome['stats']['Att_hit'][0]),
              'min_attacker_up_z':float(d['body_normal'][:n,i,1,2].min()),
              'inverted_sample_count':int((d['body_normal'][:n,i,1,2]<-.8).sum()),
              'valid_attacker_contact_samples':len(samples),'post_hit_observed_seconds':float(ph.sum()*dt),
              'post_hit_min_target_distance':float(dist[ph].min()) if ph.any() else None,
              'post_hit_final_target_distance':float(dist[-1]) if ph.any() else None,
              'post_hit_final_speed':float(speed[-1]) if ph.any() else None})
hits=sorted([v for v in rows if v['upstream_Att_hit']],key=lambda x:(x['steps'],x['scenario_id']))
selected=hits[len(hits)//2] if hits else rows[len(rows)//2];i=selected['scenario_id'];n=selected['steps'];t=(np.arange(n)+1)*dt
phase=d['phase_before_SecPass_hit'][:n,i].reshape(-1).astype(int)+d['phase_before_Att_hit'][:n,i].reshape(-1).astype(int)
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
axes[0,0].plot(t,d['ball_position'][:n,i,0,2],label='Ball z');axes[0,0].plot(t,d['drone_position'][:n,i,1,2],label='Attacker z');axes[0,0].axhline(target[2],ls=':',color='gray',label='Hover target z');axes[0,0].set_ylabel('Height (m)')
axes[0,1].plot(t,d['body_normal'][:n,i,1,2],label='Attacker up z');axes[0,1].axhline(-.8,ls=':',color='gray',label='Inverted-sample diagnostic');axes[0,1].set_ylabel('Unit-vector component')
for j,name in enumerate('xyz'):axes[1,0].plot(t,d['body_velocity'][:n,i,1,j+3],label=f'World angular velocity {name}')
axes[1,0].set_ylabel('rad/s')
distance=np.linalg.norm(d['drone_position'][:n,i,1]-origins[i]-target,axis=-1)
axes[1,1].plot(t,distance,label='Distance to hover target');axes[1,1].plot(t,np.linalg.norm(d['body_velocity'][:n,i,1,:3],axis=-1),label='Attacker speed');axes[1,1].set_ylabel('m / m/s')
colors={0:'#e9eef9',1:'#fcead5',2:'#e9f3df'}
for ax in axes.flat:
 start=0
 for k in range(1,n+1):
  if k==n or phase[k]!=phase[start]:ax.axvspan(start*dt,k*dt,color=colors[int(phase[start])],alpha=.65);start=k
 ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Time (s)')
fig.suptitle(f'HCSP Attack + Attack_hover | original Iris / PRT | scenario {i}\nBackground: Goto / Attack / Hover, from upstream policy-switch flags',fontsize=12)
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output.with_suffix('.png'),dpi=170);plt.close(fig)
summary={'task':r['task'],'source_commit':r['source_commit'],'checkpoints':r['checkpoints'],
         'scenario_count':16,'upstream_hit_count':sum(x['upstream_Att_hit'] for x in rows),
         'scenarios_with_valid_attacker_contact_samples':sum(x['valid_attacker_contact_samples']>0 for x in rows),
         'scenarios_with_inverted_samples':sum(x['inverted_sample_count']>0 for x in rows),
         'selected_scenario':selected,'selection':'median episode duration among scenarios with upstream Att_hit flag',
         'rows':rows,'recovery_success':'not claimed; position/speed curves are descriptive, no frozen recovery threshold',
         'full_front_flip_reproduction':'not claimed from hit flags or replay completion'}
a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['checkpoints','rows']},indent=2))
