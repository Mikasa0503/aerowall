"""Check that a recorded free-flight force produced the requested ball impulse."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--case-bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.report.read_text())
    assert report['status'] == 'passed' and report['num_envs'] == 1
    outcome = report['outcomes'][0]
    assert outcome['perturbation_applied'] and outcome['contact_audit_passed']
    assert report['contact_audit']['corroborated_events'] == report['contact_audit']['legal_events']
    assert report['contact_audit']['corroborated_wall_events'] == report['contact_audit']['wall_events']
    cases = json.loads(a.case_bank.read_text())['cases']
    assert len(cases) == 1
    expected = float(cases[0]['delta_vy'])
    forces = [json.loads(line) for line in a.report.with_suffix('.forces.jsonl').read_text().splitlines()]
    assert len(forces) == 40 and all(row['env'] == 0 for row in forces)
    integrated = sum(row['known_delta_vy'] for row in forces)
    start = outcome['perturbation_start_at']
    end = outcome['perturbation_end_at']
    assert end - start == 39
    events = [json.loads(line) for line in a.report.with_suffix('.events.jsonl').read_text().splitlines()]
    assert not any(start <= row['policy_step'] * 8 + row['substep'] <= end for row in events)
    velocity = np.load(report['trajectory'])['ball_velocity'][:, 0, 1]
    observed = float(velocity[end // 8 + 1] - velocity[start // 8 - 1])
    assert abs(integrated - expected) < 0.02
    assert abs(observed - expected) < 0.02
    assert max(abs(float(velocity[index + 1] - velocity[index]))
               for index in range(end // 8 + 1, end // 8 + 4)) < 0.02
    result = {'status': 'passed', 'expected_delta_vy': expected,
              'integrated_delta_vy': integrated, 'observed_delta_vy': observed,
              'force_substeps': len(forces), 'physical_contacts_during_force': 0,
              'contact_audit_passed': True, 'residual_acceleration_absent': True}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
