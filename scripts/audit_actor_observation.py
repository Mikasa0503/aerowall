"""Check recorded actor inputs against independently saved policy-boundary states.

This checks zero-delay trajectory alignment, not universal information isolation.
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
checks = {'drone_position': (obs[..., :3], dp),
          'ball_position': (obs[..., 18:21], bp),
          'relative_ball_position': (obs[..., 15:18], bp-dp)}
metadata = r['actor_observation_record']
skipped = []
for name, slot, key, delay in [
    ('drone_velocity', slice(3,6), 'drone_velocity', metadata['drone_velocity_delay_steps']),
    ('ball_velocity', slice(21,24), 'ball_velocity', metadata['ball_velocity_delay_steps'])]:
    if delay == 0:
        checks[name] = (obs[..., slot], before(key)[..., :3])
    else:
        skipped.append(name + ': nonzero delay requires separate history alignment')
errors = {name: float(np.max(np.abs(actual[active]-expected[active])))
          for name, (actual, expected) in checks.items()}
result = {'status': 'passed' if all(v < 1e-5 for v in errors.values()) else 'failed',
          'evaluation': str(a.evaluation),
          'report_sha256': hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
          'scored_observation_count': int(active.sum()), 'shape': list(obs.shape),
          'maximum_absolute_errors': errors, 'skipped': skipped,
          'scope': 'Recorded actor position and zero-delay velocity timing; does not prove every feature excludes privileged information'}
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
assert result['status'] == 'passed'
