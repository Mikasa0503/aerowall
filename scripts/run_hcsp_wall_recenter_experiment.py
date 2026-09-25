"""Run matched B/C PPO batches and development-only model selection."""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from project_paths import CHECKPOINT_DIR, ROOT, RUNS, PYTHON, runtime_environment


TRAIN = ROOT / 'scripts/train_aerowall_wall_rl.py'
EVAL = ROOT / 'scripts/evaluate_aerowall_wall_rl.py'
CONFIG = ROOT / 'configs/recenter_recovery.json'
SOURCE = CHECKPOINT_DIR
ACTOR = SOURCE / 'hcsp-wall-goal3-round3-rally-train.pt'
LAUNCH = SOURCE / 'hcsp-wall-goal2-wall-train-01.pt'


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def run_job(name, command, state):
    output = RUNS / (name + '.json')
    if output.exists():
        payload = json.loads(output.read_text())
        if payload.get('status') == 'passed':
            if not any(row['name'] == name for row in state['jobs']):
                log_text = (RUNS / (name + '.log')).read_text()
                durations = re.findall(r'\[([0-9.]+)s\] Simulation App Shutting Down', log_text)
                assert durations, f'Missing elapsed Kit time for {name}'
                elapsed = float(durations[-1])
                state['jobs'].append({'name': name, 'seconds': elapsed, 'exit_code': 0,
                                      'recovered_completed_job': True})
                state['total_gpu_seconds'] += elapsed
                atomic_json(RUNS / 'execution-state.json', state)
            return payload
        raise RuntimeError(f'Existing unfinished/failed report requires investigation: {output}')
    env = runtime_environment()
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '4'
    started = time.monotonic()
    log = RUNS / (name + '.log')
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    state['jobs'].append({'name': name, 'seconds': elapsed, 'exit_code': result.returncode})
    state['total_gpu_seconds'] += elapsed
    atomic_json(RUNS / 'execution-state.json', state)
    payload = json.loads(output.read_text()) if output.exists() else {}
    if result.returncode or payload.get('status') not in ('passed', 'budget_stopped'):
        raise RuntimeError(f'{name} failed; inspect {log}: {payload.get("error")}')
    return payload


def dev_score(natural, recovery, update):
    n = natural['summary']['safe10_count']
    r = recovery['summary']['recovery_success_count']
    prefix = natural['summary']['mean_centered_prefix_rallies']
    return ((1, r, n, prefix, -update) if n >= 26
            else (0, n, prefix, r, -update))


def recovery_gate(report):
    side = report['summary']['recovery_by_direction']
    return (report['summary']['recovery_success_count'] >= 28
            and side['left']['success'] >= 12 and side['right']['success'] >= 12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prior-gpu-seconds', type=float, required=True)
    p.add_argument('--calendar-start-utc', required=True)
    a = p.parse_args()
    cfg = json.loads(CONFIG.read_text())
    assert cfg['perturbation']['light_delta_vy'] == 0.25
    assert cfg['perturbation']['medium_delta_vy'] == 0.5
    RUNS.mkdir(parents=True, exist_ok=True)
    state_path = RUNS / 'execution-state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {'schema_version': 1, 'jobs': [], 'prior_gpu_seconds': a.prior_gpu_seconds,
                 'total_gpu_seconds': a.prior_gpu_seconds,
                 'calendar_start_utc': a.calendar_start_utc, 'pair_metrics': {},
                 'completed_pairs': [], 'best': None}
        atomic_json(state_path, state)
    assert state['calendar_start_utc'] == a.calendar_start_utc
    assert state['prior_gpu_seconds'] == a.prior_gpu_seconds
    last_state = {'B': None, 'C': None}
    best_metrics = {g: {'natural': -1, 'recovery': -1, 'prefix': -1.0} for g in ('B', 'C')}
    no_improvement = 0
    consecutive_dev_pass = {'B': 0, 'C': 0}
    for prior_target in sorted(state['completed_pairs']):
        improved_prior_pair = False
        for prior_group in ('B', 'C'):
            if prior_target == 50 and prior_group == 'C':
                continue  # Independent C50 was a nondeterminism pilot.
            prior = state['pair_metrics'][str(prior_target)][prior_group]
            earlier = best_metrics[prior_group]
            improved_prior_pair |= (prior['natural'] > earlier['natural']
                                    or prior['recovery'] > earlier['recovery']
                                    or prior['prefix'] >= earlier['prefix'] + 0.5)
            for key in earlier:
                earlier[key] = max(earlier[key], prior[key])
            consecutive_dev_pass[prior_group] = (
                consecutive_dev_pass[prior_group] + 1 if prior['eligible'] else 0)
        no_improvement = 0 if improved_prior_pair else no_improvement + 1
    for target in range(50, cfg['training']['max_updates_per_group'] + 1, 50):
        if target in state['completed_pairs']:
            for group in ('B', 'C'):
                last_state[group] = RUNS / f'{group.lower()}-u{target:03d}.state.pt'
            continue
        elapsed_days = (datetime.now(timezone.utc) - datetime.fromisoformat(a.calendar_start_utc)).total_seconds()
        if state['total_gpu_seconds'] >= cfg['training']['max_total_gpu_seconds'] - 900 or elapsed_days >= 3 * 86400:
            state['stop_reason'] = 'total_budget'
            break
        pair_results = {}
        for group in ('B', 'C'):
            group_seconds = sum(job['seconds'] for job in state['jobs']
                                if job['name'].startswith(group.lower() + '-'))
            if group_seconds >= 28800 - 900:
                state['stop_reason'] = f'{group}_eight_hour_budget'
                atomic_json(state_path, state)
                return
            name = f'{group.lower()}-u{target:03d}'
            output = RUNS / (name + '.json')
            command = [str(PYTHON), str(TRAIN), '--output', str(output), '--updates', str(target),
                       '--num-envs', '128', '--stage', 'RALLY', '--seed', '42',
                       '--launch-checkpoint', str(LAUNCH), '--experiment-config', str(CONFIG),
                       '--group', group, '--save-every', '25', '--max-gpu-seconds', '28800']
            if last_state[group]:
                if group == 'C' and target == 100:
                    command += ['--branch-from-training-state', str(RUNS / 'b-u050.state.pt')]
                else:
                    command += ['--training-state', str(last_state[group])]
            else:
                command += ['--actor-warmstart', str(ACTOR)]
            result = run_job(name, command, state)
            if result['updates_completed'] != target or result['status'] != 'passed':
                state['stop_reason'] = f'{group}_budget_stopped'
                atomic_json(state_path, state)
                return
            last_state[group] = Path(result['training_state'])
            pair_results[group] = result
            if group == 'C' and target == 100:
                b50 = json.loads((RUNS / 'b-u050.json').read_text())
                assert result['start_update'] == 50
                assert result['initial_policy_sha256'] == b50['final_policy_sha256']
        if target == 50:
            assert pair_results['B']['initial_policy_sha256'] == pair_results['C']['initial_policy_sha256']
        improved_pair = False
        pair_dev = {}
        pair_metrics = {}
        for group in ('B', 'C'):
            for protocol, count in (('natural', 32), ('recovery', 40)):
                name = f'{group.lower()}-u{target:03d}-{protocol}-dev'
                output = RUNS / (name + '.json')
                command = [str(PYTHON), str(EVAL), '--output', str(output),
                           '--checkpoint', pair_results[group]['checkpoint'],
                           '--launch-checkpoint', str(LAUNCH), '--stage', 'RALLY',
                           '--num-envs', str(count), '--seed', '2001',
                           '--experiment-config', str(CONFIG), '--protocol', protocol]
                if protocol == 'recovery':
                    command += ['--case-bank', str(ROOT / 'configs/recenter_case_banks/development-cases.json')]
                report = run_job(name, command, state)
                if not all(row['contact_audit_passed'] for row in report['outcomes']):
                    raise RuntimeError(f'Physical contact audit failed in {name}')
                pair_dev[(group, protocol)] = report
            natural = pair_dev[(group, 'natural')]
            recovery = pair_dev[(group, 'recovery')]
            metrics = {'natural': natural['summary']['safe10_count'],
                       'recovery': recovery['summary']['recovery_success_count'],
                       'prefix': natural['summary']['mean_centered_prefix_rallies']}
            earlier = best_metrics[group]
            pilot = group == 'C' and target == 50
            improved = not pilot and (metrics['natural'] > earlier['natural']
                                      or metrics['recovery'] > earlier['recovery']
                                      or metrics['prefix'] >= earlier['prefix'] + 0.5)
            improved_pair |= improved
            if not pilot:
                for key in earlier:
                    earlier[key] = max(earlier[key], metrics[key])
            eligible = metrics['natural'] >= 26 and recovery_gate(recovery)
            metrics['eligible'] = eligible
            pair_metrics[group] = metrics
            consecutive_dev_pass[group] = consecutive_dev_pass[group] + 1 if eligible else 0
            candidate = {'group': group, 'update': target, 'checkpoint': pair_results[group]['checkpoint'],
                         'training_state': pair_results[group]['training_state'], 'metrics': metrics,
                         'score': dev_score(natural, recovery, target)}
            if not pilot and (state['best'] is None or candidate['score'] > tuple(state['best']['score'])):
                state['best'] = candidate
        state['completed_pairs'].append(target)
        state['pair_metrics'][str(target)] = pair_metrics
        atomic_json(state_path, state)
        print(json.dumps({'pair_update': target, 'best': state['best'],
                          'total_gpu_seconds': state['total_gpu_seconds']}), flush=True)
        no_improvement = 0 if improved_pair else no_improvement + 1
        if max(consecutive_dev_pass.values()) >= 2:
            state['stop_reason'] = 'two_development_gates'
            break
        if target >= 300 and no_improvement >= 2:
            state['stop_reason'] = 'development_plateau'
            break
    else:
        state['stop_reason'] = 'update_cap'
    atomic_json(state_path, state)


if __name__ == '__main__':
    main()
