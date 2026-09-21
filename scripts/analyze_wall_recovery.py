"""Measure real post-launch motion, excluding terminal collision samples.

These are diagnostics, not a new success rule or an imposed hover trajectory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--scenarios',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
assert hashlib.sha256(a.scenarios.read_bytes()).hexdigest()==r['scenario_sha256']
z=np.load(a.evaluation.with_suffix('.trajectory.npz'))
origins=np.array(json.loads(a.scenarios.read_text())['env_origins'])
events=[json.loads(line) for line in a.evaluation.with_suffix('.events.jsonl').read_text().splitlines()]
times=(np.arange(len(z['active_before']))+1)*float(z['dt']);rows=[]
for o in r['outcomes']:
    if not o['wall_hits']:continue
    i=o['scenario_id'];es=[e for e in events if e['env_id']==i]
    wall=next(e for e in es if e['kind']=='wall' and e['credited'])
    caps=[e for e in es if e.get('wall_cap_credit') and e['time']<wall['time']]
    if not caps:continue
    cap=caps[-1];terminal=o.get('physics_steps',o['steps'])*r['physics_dt']
    # Strictly before terminal microstep: the final held-action sample can
    # contain ground-impact angular velocity and must not diagnose control.
    mask=z['active_before'][:,i].astype(bool)&(times>=cap['time'])&(times<terminal-1e-8)
    if not mask.any():continue
    indices=np.flatnonzero(mask);dp=z['drone_position'][mask,i]-origins[i]
    dv=z['drone_velocity'][mask,i];q=z['drone_quaternion_wxyz'][mask,i]
    nz=1-2*(q[:,1]**2+q[:,2]**2)
    angular=np.linalg.norm(dv[:,3:],axis=-1)
    prewall=indices[times[indices]<wall['time']]
    atwall=int(prewall[-1]) if len(prewall) else int(indices[0])
    wall_z=float(z['drone_position'][atwall,i,2]-origins[i,2])
    rows.append({'scenario_id':i,'reason':o['reason'],'rallies':o['rallies'],
                 'cap_time':cap['time'],'wall_time':wall['time'],'terminal_time':terminal,
                 'wall_to_terminal_seconds':terminal-wall['time'],
                 'sample_count':len(indices),'last_preterminal_sample_time':float(times[indices[-1]]),
                 'height_first_postcap_sample':float(dp[0,2]),'height_last_prewall_sample':wall_z,
                 'height_last_preterminal_sample':float(dp[-1,2]),'minimum_height':float(dp[:,2].min()),
                 'minimum_up_z':float(nz.min()),'maximum_tilt_degrees':float(np.degrees(np.arccos(np.clip(nz,-1,1))).max()),
                 'maximum_angular_speed_rad_s':float(angular.max()),
                 'last_preterminal_vertical_speed':float(dv[-1,2]),
                 'launch_ball_velocity':cap['ball_velocity_after'][:3]})
result={'evaluation':str(a.evaluation),'report_sha256':hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
        'scope':'Actual cap-to-terminal intervals for first wall encounters; policy-rate body states, microstep event times; no recovery success threshold',
        'analyzed_wall_scenarios':len(rows),'ending_counts':dict(Counter(x['reason'] for x in rows)),
        'postlaunch_height_loss_over_0_3m':sum(x['height_first_postcap_sample']-x['height_last_preterminal_sample']>.3 for x in rows),
        'rows':rows}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
