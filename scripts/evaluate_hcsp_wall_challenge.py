"""Run a frozen post-selection strong-impulse challenge, outside formal gates."""

import json
from pathlib import Path

from finalize_hcsp_wall_recenter_experiment import evaluate, atomic_json, digest
from run_hcsp_wall_recenter_experiment import ROOT, RUNS


def main():
    formal = json.loads((RUNS / 'formal-summary.json').read_text())
    assert formal['status'] == 'passed'
    selection = formal['selection']
    bank = ROOT / 'configs/recenter_case_banks/challenge-cases.json'
    assert len(json.loads(bank.read_text())['cases']) == 40
    budget = json.loads((RUNS / 'formal-runtime.json').read_text())
    results = {}
    for group in ('A', 'B', 'C'):
        checkpoint = selection['groups'][group]['checkpoint']
        name = f'{group.lower()}-challenge-40'
        report = evaluate(name, checkpoint, 'recovery', 40, 7001, budget, bank)
        displaced = report['summary']['off_center_recovery_subset']
        results[group] = {
            'checkpoint_sha256': digest(checkpoint),
            'all_recovery_success': report['summary']['recovery_success_count'],
            'left_right': report['summary']['recovery_by_direction'],
            'actually_off_center_before_recovery': displaced,
            'protocol_missed': report['summary']['perturbation_missed_count'],
            'contact_audit_passed': all(row['contact_audit_passed'] for row in report['outcomes']),
        }
    output = RUNS / 'challenge-summary.json'
    atomic_json(output, {'status': 'passed', 'scope': 'post-selection stress test; not used for model selection or acceptance',
                         'case_bank_sha256': digest(bank), 'results': results,
                         'total_gpu_seconds_through_challenge': budget['prior_total_seconds'] + budget['formal_seconds']})
    print(json.dumps({'report': str(output), 'results': results}))


if __name__ == '__main__':
    main()
