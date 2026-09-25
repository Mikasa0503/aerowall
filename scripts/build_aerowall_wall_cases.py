"""Create immutable AeroWall wall-task initial-state banks for paired evaluation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowall.wall_rl.curriculum import CASE_VERSION, make_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['fixed', 'train', 'heldout'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = {'schema_version': CASE_VERSION, 'kind': args.kind, 'seed': args.seed,
               'cases': make_cases(args.kind, args.seed, args.count)}
    content = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != content:
        raise ValueError('Frozen bank differs; use a new output path')
    args.output.write_text(content)
    print(json.dumps({'output': str(args.output), 'count': args.count,
                      'sha256': hashlib.sha256(content.encode()).hexdigest()}))


if __name__ == '__main__':
    main()
