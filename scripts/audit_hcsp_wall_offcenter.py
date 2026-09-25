"""Recalculate pre-recovery off-center evidence without editing raw reports."""

import hashlib
import json
from pathlib import Path

from run_hcsp_wall_recenter_experiment import RUNS


def main():
    for path in sorted(RUNS.glob('*-recovery-dev.json')):
        report = json.loads(path.read_text())
        assert report['status'] == 'passed' and report['protocol'] == 'recovery'
        displaced = []
        for row in report['outcomes']:
            candidate = row['recovery_wall_ordinal']
            prior_wall_y = row['wall_y_sequence'][2: candidate - 1 if candidate else 5]
            if any(abs(y) > 1.0 for y in prior_wall_y):
                displaced.append(row)
        summary = {'source_report': str(path),
                   'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'definition': 'wall contact outside |y|<=1 after impulse and before first centered recovery wall',
                   'count': len(displaced),
                   'recovery_success': sum(row['recovery_success'] for row in displaced),
                   'direction_counts': {direction: sum(row['direction'] == direction for row in displaced)
                                        for direction in ('left', 'right')},
                   'deprecated_later_off_center_count': report['summary'].get('out_of_band_wall_subset', {}).get('count')}
        output = path.with_suffix('.offcenter-audit.json')
        output.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'report': path.name, 'off_center_before_recovery': len(displaced),
                          'later_off_center_was': summary['deprecated_later_off_center_count']}))


if __name__ == '__main__':
    main()
