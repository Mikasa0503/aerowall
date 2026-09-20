"""Compare frozen eager/lazy evaluations, including every stored trajectory array."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--eager', type=Path, required=True)
p.add_argument('--lazy', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
x, y = (json.loads(v.read_text()) for v in (a.eager, a.lazy))
paths = [v.with_suffix('.trajectory.npz') for v in (a.eager, a.lazy)]
u, v = [np.load(path) for path in paths]
arrays = {k: k in v.files and np.array_equal(u[k], v[k], equal_nan=True) for k in u.files}
checks = {'both_passed': x['status'] == y['status'] == 'passed',
          'same_checkpoint': x['checkpoint_sha256'] == y['checkpoint_sha256'],
          'same_scenarios': x['scenario_sha256'] == y['scenario_sha256'],
          'same_outcomes': x['outcomes'] == y['outcomes'],
          'same_array_keys': set(u.files) == set(v.files), 'arrays': arrays}
r = {'status': 'passed' if all(v for k,v in checks.items() if k != 'arrays') and all(arrays.values()) else 'failed',
     'checks': checks, 'reports': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.eager,a.lazy)},
     'lazy_elapsed_seconds': y['evaluation_elapsed_seconds'], 'lazy_gpu_queries': y['gpu_contact_buffer_queries'],
     'scope': '100 frozen scenarios; equality of saved trajectories and outcomes, not a universal determinism claim'}
a.output.write_text(json.dumps(r, indent=2)+'\n')
assert r['status'] == 'passed', r
print(json.dumps(r))
