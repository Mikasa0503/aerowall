"""Check recorded actor inputs against independently saved policy-boundary states.

This checks configured sample-delay alignment, not universal information isolation.
Only the scored first episode is inspected; reset slots are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation', type=Path, required=True)
p.add_argument('--scenarios', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
r = json.loads(a.evaluation.read_text())
assert r['status'] == 'passed'
assert hashlib.sha256(a.scenarios.read_bytes()).hexdigest() == r['scenario_sha256']
roster = json.loads(a.scenarios.read_text())
z = np.load(a.evaluation.with_suffix('.trajectory.npz'))
obs = z['actor_observation_before']
active = z['active_before'].astype(bool)
assert obs.shape == (*active.shape, 43)
assert np.isfinite(obs[active]).all()
origin = np.asarray(roster['env_origins'])
# Observation precedes the action; saved simulator states follow the action.
def before(key):
    return np.concatenate([np.asarray(roster['initial_state'][key])[None], z[key][:-1]], axis=0)

dp = before('drone_position') - origin
bp = before('ball_position') - origin
checks = {'drone_position': (obs[..., :3], dp, active),
          'ball_position': (obs[..., 18:21], bp, active),
          'relative_ball_position': (obs[..., 15:18], bp-dp, active)}
metadata = r['actor_observation_record']
skipped = []
for name, slot, key, delay in [
    ('drone_velocity', slice(3,6), 'drone_velocity', metadata['drone_velocity_delay_steps']),
    ('ball_velocity', slice(21,24), 'ball_velocity', metadata['ball_velocity_delay_steps'])]:
    assert delay in (0,1,2)
    expected=before(key)[..., :3]
    mask=active.copy()
    if delay:
        expected=np.concatenate([np.repeat(expected[:1],delay,axis=0),expected[:-delay]],axis=0)
        mask[:delay]=False
        skipped.append(name + ': first '+str(delay)+' warmup samples have no recorded prehistory')
    checks[name]=(obs[..., slot],expected,mask)
errors = {name: float(np.max(np.abs(actual[mask]-expected[mask]))) if mask.any() else None
          for name, (actual, expected,mask) in checks.items()}
mismatches={}
for name,(actual,expected,mask) in checks.items():
    bad=(np.max(np.abs(actual-expected),axis=-1)>1e-5)&mask
    rows=np.argwhere(bad)
    if len(rows):mismatches[name]={'count':len(rows),'first_indices_step_env':rows[:10].tolist()}
result = {'status': 'passed' if all(v is not None and v < 1e-5 for v in errors.values()) else 'failed',
          'evaluation': str(a.evaluation),
          'report_sha256': hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
          'scored_observation_count': int(active.sum()), 'shape': list(obs.shape),
          'maximum_absolute_errors': errors, 'mismatches':mismatches,'skipped': skipped,
          'scope': 'Recorded actor position and configured velocity-delay timing; does not prove every feature excludes privileged information'}
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
assert result['status'] == 'passed'
