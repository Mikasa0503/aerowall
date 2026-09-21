"""One-shot evaluation chain for the currently authorized single-drone trial."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
output = Path('runs/self-return-smallstep-evaluation-chain.json')
if output.exists():
    raise RuntimeError('Existing chain report: inspect its process before rerunning')
report = {'status': 'waiting', 'pid': os.getpid(), 'completed': []}
def record(**kwargs):
    report.update(kwargs)
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2)+'\n')
    tmp.replace(output)

def run(args):
    subprocess.run(args, check=True)

record()
try:
    for update, frames in [(25, 34062336), (50, 34267136)]:
        checkpoint = Path('checkpoints/self-return-smallstep-dev-001') / ('frames-%012d.pt' % frames)
        deadline = time.monotonic()+1800
        while not checkpoint.exists():
            training = json.loads(Path('runs/self-return-smallstep-dev-001.json').read_text())
            if training['status'] in ('failed', 'passed'):
                raise RuntimeError('Training ended without expected checkpoint: '+str(checkpoint))
            # Revalidate the reported process; no automatic restart on failure.
            os.kill(training['pid'], 0)
            if time.monotonic() > deadline:
                raise TimeoutError('Checkpoint readiness wait expired; inspect existing training')
            time.sleep(10)
        # Trainer writes to .tmp then atomically renames the complete checkpoint.
        name = 'self-return-smallstep-eval-%d-01' % update
        evaluation = 'runs/'+name+'.json'
        record(status='evaluating', current_update=update, evaluation=evaluation)
        with open('runs/'+name+'.log', 'x') as log:
            subprocess.run(['./scripts/run_probe.sh', 'scripts/evaluate_wall_rally.py', evaluation,
                '--checkpoint', str(checkpoint), '--config', 'runs/self-return-smallstep-dev-001.yaml',
                '--scenarios', 'artifacts/singlejuggle-development-scenarios-100.json'], stdout=log, stderr=subprocess.STDOUT, check=True)
        run(['./scripts/python.sh', '--plain', 'scripts/audit_wall_evaluation.py', '--evaluation', evaluation,
             '--output', 'docs/'+name+'-event-audit.json'])
        report['completed'].append(update)
        record(status='waiting')
    run(['./scripts/python.sh', '--plain', 'scripts/compare_wall_evaluations.py',
         '--evaluation', 'runs/skill-distillation-eval-01.json',
         '--evaluation', 'runs/self-return-smallstep-eval-25-01.json',
         '--evaluation', 'runs/self-return-smallstep-eval-50-01.json',
         '--output', 'docs/self-return-smallstep-comparison.json'])
    record(status='passed')
except Exception as error:
    record(status='failed', error=repr(error))
    raise
