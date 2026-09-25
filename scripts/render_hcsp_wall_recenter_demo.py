"""Render a three-stage natural demo with corrected wall lighting and framing."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from finalize_hcsp_wall_recenter_experiment import post_report_close_crash
from run_hcsp_wall_recenter_experiment import ROOT, RUNS, PYTHON, EVAL, CONFIG, LAUNCH
from project_paths import CHECKPOINT_DIR, runtime_environment


EARLY_CHECKPOINT = CHECKPOINT_DIR / 'hcsp-wall-goal2-return-early-train.pt'
MIDDLE_CHECKPOINT = CHECKPOINT_DIR / 'hcsp-wall-goal3-round2-rally-train.pt'
ARTIFACT = ROOT / 'artifacts/hcsp-wall-recenter-demo-visual-fix-900'


def run_evaluation(name, checkpoint, seed, rgb=False, experiment_config=CONFIG):
    report_path = RUNS / (name + '.json')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report['status'] == 'passed'
        assert report['checkpoint_sha256'] == hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        assert all(row['contact_audit_passed'] for row in report['outcomes'])
        return report
    command = [str(PYTHON), str(EVAL), '--output', str(report_path),
               '--checkpoint', str(checkpoint), '--launch-checkpoint', str(LAUNCH),
               '--stage', 'RALLY', '--num-envs', '1', '--seed', str(seed),
               '--experiment-config', str(experiment_config), '--protocol', 'natural']
    if rgb:
        command += ['--record-rgb']
    environment = runtime_environment()
    environment['OMP_NUM_THREADS'] = environment['MKL_NUM_THREADS'] = '4'
    with (RUNS / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if report.get('status') != 'passed':
        raise RuntimeError(f'{name} failed: {report.get("error")}')
    assert report['outcomes'][0]['contact_audit_passed']
    if result.returncode:
        close_crash = post_report_close_crash((RUNS / (name + '.log')).read_text(errors='replace'))
        if not close_crash:
            raise RuntimeError(f'{name} exited {result.returncode} after reporting success')
        print(json.dumps({'name': name, 'post_report_close_crash': True,
                          'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest()}), flush=True)
    return report


def main():
    formal = json.loads((RUNS / 'formal-summary.json').read_text())
    assert formal['status'] == 'passed'
    selected_group = formal['selection']['selected_demo_group']
    checkpoint = formal['selection']['groups'][selected_group]['checkpoint']
    assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == formal['selection']['groups'][selected_group]['sha256']
    assert EARLY_CHECKPOINT.exists() and MIDDLE_CHECKPOINT.exists()
    started = time.monotonic()
    early_report = run_evaluation('early-visual-fix-900', EARLY_CHECKPOINT, 101, rgb=True)
    middle_report = run_evaluation('middle-visual-fix-900', MIDDLE_CHECKPOINT, 101, rgb=True)
    assert early_report['outcomes'][0]['rallies'] == 1
    assert middle_report['outcomes'][0]['rallies'] == 2
    # Seed 5001 was already selected from a presentation-only seed range. This
    # pass changes rendering only, so keep the same natural policy rollout.
    chosen = 5001
    final_name = f'{selected_group.lower()}-visual-fixed-900-final-{chosen}'
    final_report = run_evaluation(final_name, checkpoint, chosen, rgb=True)
    final_path = RUNS / (final_name + '.json')
    early_frames = int(early_report['rgb']['frames'])
    middle_frames = int(middle_report['rgb']['frames'])
    segments = {'segments': [
        {'role': 'early', 'report': str(RUNS / 'early-visual-fix-900.json'), 'start_frame': 0,
         'end_frame': min(125, early_frames), 'minimum_rallies': 1},
        {'role': 'middle', 'report': str(RUNS / 'middle-visual-fix-900.json'), 'start_frame': 0,
         'end_frame': middle_frames, 'minimum_rallies': 2},
        {'role': 'final', 'report': str(final_path), 'start_frame': 0,
         'end_frame': int(final_report['rgb']['frames']),
         'minimum_rallies': int(final_report['outcomes'][0]['rallies'])},
    ]}
    segment_path = RUNS / 'demo-segments.json'
    segment_path.write_text(json.dumps(segments, indent=2) + '\n')
    subprocess.run([str(PYTHON), str(ROOT / 'scripts/compose_clean_hcsp_wall_demo.py'),
                    '--segments', str(segment_path), '--output', str(ARTIFACT)],
                   cwd=ROOT, check=True)
    result = {
        'status': 'passed', 'selected_group': selected_group,
        'presentation_seed': chosen, 'presentation_candidates_checked': 1,
        'final_rallies': int(final_report['outcomes'][0]['rallies']),
        'final_safe10': bool(final_report['outcomes'][0]['safe10']),
        'video': str(ARTIFACT / 'HCSP-wall-learning-demo-clean.mp4'),
        'video_seconds': json.loads((ARTIFACT / 'manifest.json').read_text())['probe']['format']['duration'],
        'video_gpu_seconds': time.monotonic() - started,
        "visual_fixes": ["hid the referenced sphere-light emitter while preserving ambient lighting", "raised camera framing while preserving view direction"],
    }
    (ARTIFACT / 'delivery-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
