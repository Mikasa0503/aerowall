"""Attribute scored episode outcomes to before/after observable recovery handoff."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.evaluation.read_text());assert r['status']=='passed'
assert r['controller_mode']=='hierarchical_learned_recovery_skill'
path=a.evaluation.with_suffix('.trajectory.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==r['trajectory_sha256']
z=np.load(path);obs=z['actor_observation_before'];active=z['active_before'].astype(bool)
phase=obs[...,40:43].argmax(-1);gate=((phase==1)&(obs[...,21]>.5))|(phase==2)
recorded=z['recovery_controlled'].reshape(gate.shape).astype(bool)
assert np.array_equal(gate[active],recorded[active]),'Recorded routing differs from observable rule'
never=np.flatnonzero(~(gate&active).any(0)).tolist();outcomes={o['scenario_id']:o for o in r['outcomes']}
assert all(outcomes[i]['rallies']==0 for i in never)
result={'evaluation':str(a.evaluation),'report_sha256':hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'routing_matches':True,
 'scope':'Actual scored first episodes. Episodes ending before any recovery action cannot be repaired by the recovery branch on these recorded trajectories; not a universal reachability bound.',
 'scenarios':len(outcomes),'no_recovery_handoff_scenarios':never,'no_handoff_reasons':dict(Counter(outcomes[i]['reason'] for i in never)),
 'scenarios_with_recovery_handoff':len(outcomes)-len(never)}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
