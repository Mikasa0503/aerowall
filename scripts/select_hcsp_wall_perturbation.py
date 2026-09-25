"""Select frozen light/medium impulse sizes from the A-only calibration bank."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration-report', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    report = json.loads(a.calibration_report.read_text())
    assert report['status'] == 'passed' and report['protocol'] == 'recovery'
    assert not report['contact_audit']['callback_errors']
    case_index = {case['case_id']: case for case in
                  json.loads(Path(report['case_bank_path']).read_text())['cases']}
    grouped = defaultdict(list)
    for row in report['outcomes']:
        assert row['contact_audit_passed'] and row['perturbation_applied'] and not row['perturbation_missed'], row['case_id']
        magnitude = round(abs(case_index[row['case_id']]['delta_vy']), 6)
        grouped[(magnitude, row['direction'])].append(row)
    candidates = sorted({level for level, _ in grouped})
    assert len(candidates) == 6
    for level in candidates:
        assert len(grouped[(level, 'left')]) == len(grouped[(level, 'right')]) == 16
    rate = {level: sum(sum(row['recovery_success'] for row in grouped[(level, sign)]) for sign in ('left', 'right')) / 32
            for level in candidates}
    viable = [(lo, hi) for lo in candidates for hi in candidates if lo < hi]
    light, medium = min(viable, key=lambda pair: (abs(rate[pair[0]] - 0.8) + abs(rate[pair[1]] - 0.4), pair))
    selection_method = 'baseline_recovery_bracket'
    if abs(rate[light] - 0.8) > 0.25 or abs(rate[medium] - 0.4) > 0.25:
        # A has a deterministic rightward drift; if no candidate can meet the
        # recovery contract, bracket by physical severity without weakening it.
        viable_levels = []
        for level in candidates:
            mean_rallies = {direction: sum(row['rallies'] for row in grouped[(level, direction)]) / 16
                            for direction in ('left', 'right')}
            if min(mean_rallies.values()) >= 5:
                viable_levels.append(level)
        if len(viable_levels) < 2:
            raise SystemExit('Calibration found fewer than two safe, trainable impulse levels')
        light, medium = viable_levels[:2]
        selection_method = 'physical_severity_fallback'
    result = {'light_delta_vy': light, 'medium_delta_vy': medium, 'rates': rate,
              'selection_method': selection_method,
              'calibration_report_sha256': hashlib.sha256(a.calibration_report.read_bytes()).hexdigest()}
    result['status'] = 'selected'
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    config = json.loads(a.config.read_text())
    config['perturbation']['light_delta_vy'] = light
    config['perturbation']['medium_delta_vy'] = medium
    a.config.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
