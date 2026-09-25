"""Freeze deterministic, direction-balanced ball-force scenarios."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def make_cases(kind, seed, light=None, medium=None):
    rng = random.Random(seed)
    cases = []
    if kind == 'calibration':
        levels = [('candidate_' + str(value).replace('.', 'p'), value, 16)
                  for value in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)]
    else:
        assert light is not None and medium is not None and 0 < light < medium
        count = 10 if kind in ('development', 'challenge') else 25
        levels = [('light', light, count), ('medium', medium, count)]
    for difficulty, magnitude, count in levels:
        for direction, sign in (('left', -1), ('right', 1)):
            for repeat in range(count):
                cases.append({
                    'case_id': f'{kind}-{difficulty}-{direction}-{repeat:02d}',
                    'difficulty': difficulty,
                    'direction': direction,
                    'delta_vy': sign * magnitude,
                    'delay_seconds': round(rng.uniform(0.12, 0.2), 7),
                })
    rng.shuffle(cases)
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['calibration', 'development', 'formal', 'challenge'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--light', type=float)
    p.add_argument('--medium', type=float)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': 1, 'kind': a.kind, 'seed': a.seed,
               'cases': make_cases(a.kind, a.seed, a.light, a.medium)}
    content = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    if a.output.exists():
        assert a.output.read_text() == content, 'Frozen case bank differs; choose a new path'
    else:
        a.output.write_text(content)
    print(json.dumps({'path': str(a.output), 'cases': len(payload['cases']),
                      'sha256': hashlib.sha256(content.encode()).hexdigest()}))


if __name__ == '__main__':
    main()
