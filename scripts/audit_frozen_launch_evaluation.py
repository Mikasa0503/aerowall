"""Compare actual evaluated trajectories before the first recovery handoff."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--skill',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();reports=[json.loads(f.read_text()) for f in [a.baseline,a.skill]]
assert all(r['status']=='passed' for r in reports)
assert reports[1]['controller_mode']=='hierarchical_learned_recovery_skill'
for k in ['scenario_sha256','physics_dt','policy_dt','episode_seconds','wall_fixture']:
 assert reports[0][k]==reports[1][k],k
zs=[]
for f,r in zip([a.baseline,a.skill],reports):
 path=f.with_suffix('.trajectory.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==r['trajectory_sha256'];zs.append(np.load(path))
obs=zs[1]['actor_observation_before'];phase=obs[...,40:43].argmax(-1)
route=((phase==1)&(obs[...,21]>.5))|(phase==2)
keys=[k for k in ['actor_observation_before','action','drone_position','ball_position'] if all(k in z.files for z in zs)]
rows=[]
for i in range(obs.shape[1]):
 active=zs[1]['active_before'][:,i].astype(bool)
 switches=np.flatnonzero(route[:,i]&active)
 stop=int(switches[0]) if len(switches) else int(active.sum())
 n=min(stop,len(zs[0]['active_before']))
 assert zs[0]['active_before'][:n,i].all()
 errors={k:float(np.max(np.abs(zs[0][k][:n,i]-zs[1][k][:n,i]))) if n else 0.
         for k in keys}
 rows.append({'scenario_id':i,'first_recovery_step':int(switches[0]) if len(switches) else None,'compared_steps':n,'maximum_errors':errors})
maximum={k:max(r['maximum_errors'][k] for r in rows) for k in rows[0]['maximum_errors']}
result={'baseline':str(a.baseline),'skill':str(a.skill),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'report_hashes':[hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.baseline,a.skill]],
 'scope':'Actual fixed-scene deterministic evaluation before first recovery decision; excludes handoff step and later samples',
 'compared_keys':keys,'actor_input_available_in_both':'actor_observation_before' in keys,
 'all_pre_handoff_values_exact':all(v==0 for v in maximum.values()),'maximum_errors':maximum,'rows':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
