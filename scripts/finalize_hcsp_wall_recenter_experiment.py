"""Freeze development selections and perform untouched formal evaluations."""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from run_hcsp_wall_recenter_experiment import (
    ROOT, RUNS, PYTHON, EVAL, CONFIG, SOURCE, LAUNCH, ACTOR, dev_score,
)
from project_paths import runtime_environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def post_report_close_crash(log_text):
    """Recognize Isaac Sim crashing only while closing after a saved report."""
    return ('Fatal Python error: Segmentation fault' in log_text
            and 'simulation_app.py' in log_text and 'in close' in log_text)


def evaluate(name, checkpoint, protocol, count, seed, budget, case_bank=None):
    output = RUNS / (name + '.json')
    if output.exists():
        report = json.loads(output.read_text())
        if (report.get('status') == 'passed' and report['checkpoint_sha256'] == digest(checkpoint)
                and all(row['contact_audit_passed'] for row in report['outcomes'])):
            if not any(job['name'] == name for job in budget['jobs']):
                log_text = (RUNS / (name + '.log')).read_text(errors='replace')
                durations = re.findall(r'\[([0-9.]+)s\] Simulation App Shutting Down',
                                       log_text)
                close_crash = not durations and post_report_close_crash(log_text)
                if not durations and not close_crash:
                    raise RuntimeError(f'Cannot account completed evaluation time: {name}')
                # The evaluator starts its timer after Kit initialization.
                # Charge an extra minute for startup/close on recovered crashes;
                # normal jobs use Kit's complete process time instead.
                elapsed = (float(durations[-1]) if durations
                           else float(report['elapsed_seconds']) + 60.0)
                budget['formal_seconds'] += elapsed
                budget['jobs'].append({'name': name, 'seconds': elapsed,
                                       'recovered_completed_job': True,
                                       'post_report_close_crash': close_crash,
                                       'time_estimated_conservatively': close_crash})
                atomic_json(RUNS / 'formal-runtime.json', budget)
            return report
        raise RuntimeError(f'Existing invalid or incomplete evaluation: {output}')
    if budget['formal_seconds'] >= 14400 - 600 or budget['prior_total_seconds'] + budget['formal_seconds'] >= 86400 - 600:
        raise RuntimeError('Formal evaluation GPU budget reached')
    command = [str(PYTHON), str(EVAL), '--output', str(output), '--checkpoint', str(checkpoint),
               '--launch-checkpoint', str(LAUNCH), '--stage', 'RALLY', '--num-envs', str(count),
               '--seed', str(seed), '--experiment-config', str(CONFIG), '--protocol', protocol]
    if case_bank:
        command += ['--case-bank', str(case_bank)]
    environment = runtime_environment()
    environment['OMP_NUM_THREADS'] = environment['MKL_NUM_THREADS'] = '4'
    started = time.monotonic()
    with (RUNS / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads(output.read_text()) if output.exists() else {}
    if report.get('status') != 'passed':
        raise RuntimeError(f'{name} failed: {report.get("error")}')
    if not all(row['contact_audit_passed'] for row in report['outcomes']):
        raise RuntimeError(f'{name} physical contact audit failed')
    close_crash = result.returncode != 0 and post_report_close_crash(
        (RUNS / (name + '.log')).read_text(errors='replace'))
    if result.returncode and not close_crash:
        raise RuntimeError(f'{name} exited {result.returncode} after reporting success; inspect log')
    elapsed = time.monotonic() - started
    budget['formal_seconds'] += elapsed
    budget['jobs'].append({'name': name, 'seconds': elapsed,
                           'exit_code': result.returncode,
                           'post_report_close_crash': close_crash})
    atomic_json(RUNS / 'formal-runtime.json', budget)
    print(json.dumps({'name': name, 'seconds': elapsed,
                      'gate': report['gate']}), flush=True)
    return report


def choose_group_checkpoint(group, completed_pairs):
    best = None
    for update in completed_pairs:
        if group == 'C' and update == 50:
            continue  # Independent C50 is a pilot; official C branches at B50.
        prefix = f'{group.lower()}-u{update:03d}'
        natural = json.loads((RUNS / (prefix + '-natural-dev.json')).read_text())
        recovery = json.loads((RUNS / (prefix + '-recovery-dev.json')).read_text())
        candidate = {'group': group, 'update': update,
                     'checkpoint': str(RUNS / (prefix + '.pt')),
                     'score': dev_score(natural, recovery, update)}
        if best is None or candidate['score'] > best['score']:
            best = candidate
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=RUNS / 'formal-summary.json')
    a = p.parse_args()
    state = json.loads((RUNS / 'execution-state.json').read_text())
    runtime_path = RUNS / 'formal-runtime.json'
    budget = (json.loads(runtime_path.read_text()) if runtime_path.exists() else
              {'formal_seconds': 0.0, 'prior_total_seconds': state['total_gpu_seconds'], 'jobs': []})
    completed = state['completed_pairs']
    assert completed and state.get('stop_reason'), 'Training must have stopped before formal evaluation'
    selected = {group: choose_group_checkpoint(group, completed) for group in ('B', 'C')}
    assert selected['B'] and selected['C']
    leader = max(selected.values(), key=lambda row: row['score'])
    selection_path = RUNS / 'formal-selection.json'
    selection = {
        'selection_basis': 'development-only predeclared ranking',
        'training_stop_reason': state['stop_reason'],
        'experiment_sha256': digest(CONFIG),
        'formal_case_bank_sha256': digest(ROOT / 'configs/recenter_case_banks/formal-cases.json'),
        'launch_sha256': digest(LAUNCH),
        'groups': {
            'A': {'checkpoint': str(ACTOR), 'sha256': digest(ACTOR), 'update': 0},
            **{group: {**row, 'sha256': digest(row['checkpoint'])}
               for group, row in selected.items()},
        },
        'selected_demo_group': leader['group'],
    }
    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == json.loads(json.dumps(selection))
    else:
        atomic_json(selection_path, selection)
    formal_bank = ROOT / 'configs/recenter_case_banks/formal-cases.json'
    results = {}
    for group in ('A', 'B', 'C'):
        checkpoint = selection['groups'][group]['checkpoint']
        natural = evaluate(f'{group.lower()}-formal-natural-128', checkpoint, 'natural', 128, 3001, budget)
        recovery = evaluate(f'{group.lower()}-formal-recovery-100', checkpoint, 'recovery', 100, 3001, budget, formal_bank)
        results[group] = {
            'checkpoint_sha256': selection['groups'][group]['sha256'],
            'natural_safe10_count': natural['summary']['safe10_count'],
            'natural_safe15_count': natural['summary']['safe15_count'],
            'natural_gate_passed': natural['gate']['passed'],
            'recovery_success_count': recovery['summary']['recovery_success_count'],
            'recovery_by_direction': recovery['summary']['recovery_by_direction'],
            'recovery_off_center_subset': recovery['summary']['off_center_recovery_subset'],
            'recovery_gate_passed': recovery['gate']['passed'],
        }
        atomic_json(a.output, {'status': 'running', 'selection': selection, 'results': results})
    repeat = []
    demo_group = leader['group']
    demo_checkpoint = selection['groups'][demo_group]['checkpoint']
    for index in range(20):
        report = evaluate(f'{demo_group.lower()}-formal-repeat-{index:02d}', demo_checkpoint,
                          'natural', 1, 4001 + index, budget)
        repeat.append(bool(report['outcomes'][0]['safe10'] and report['outcomes'][0]['contact_audit_passed']))
        atomic_json(a.output, {'status': 'running', 'selection': selection,
                               'results': results, 'repeat': repeat})
    result = {'status': 'passed', 'selection': selection, 'results': results,
              'repeat_safe10_count': sum(repeat), 'repeat_required': 16,
              'repeat_passed': sum(repeat) >= 16,
              'formal_gpu_seconds': budget['formal_seconds'],
              'total_gpu_seconds_through_formal': budget['prior_total_seconds'] + budget['formal_seconds'],
              'complete_target_passed': (results[demo_group]['natural_gate_passed']
                                         and results[demo_group]['recovery_gate_passed']
                                         and sum(repeat) >= 16)}
    atomic_json(a.output, result)
    print(json.dumps({'status': result['status'], 'selected_demo_group': demo_group,
                      'complete_target_passed': result['complete_target_passed']}), flush=True)


if __name__ == '__main__':
    main()
