"""Continue the frozen ablation into evaluation after all training jobs finish."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from project_paths import PYTHON, ROOT, runtime_environment


SPEC = ROOT / "configs/wall_reward_ablation.json"
EVALUATE = ROOT / "scripts/evaluate_hcsp_wall_reward_ablation.py"


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def is_training_runner(pid):
    command_path = Path(f"/proc/{pid}/cmdline")
    if not command_path.exists():
        return False
    command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    return "run_hcsp_wall_reward_ablation.py" in command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-pid-file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/wall-reward-ablation")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    runner_pid = int(args.runner_pid_file.read_text())
    state_path = output_dir / "pipeline-state.json"
    if state_path.exists():
        raise RuntimeError(f"Pipeline monitor already registered: {state_path}")
    spec_sha = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    state = {"status": "waiting_for_training", "runner_pid": runner_pid,
             "spec_sha256": spec_sha, "monitor_pid": os.getpid()}
    atomic_json(state_path, state)
    started = time.monotonic()
    try:
        while True:
            manifest = output_dir / "manifest.json"
            if manifest.exists():
                training = json.loads(manifest.read_text())
                if training.get("spec_sha256") != spec_sha:
                    raise RuntimeError("Training manifest belongs to a different reward experiment")
                status = training.get("status")
                if status == "training_complete":
                    break
                if status in ("failed", "failed_initial_state_mismatch", "budget_stopped"):
                    state.update(status="training_incomplete", training_status=status)
                    atomic_json(state_path, state)
                    return
            if not is_training_runner(runner_pid):
                raise RuntimeError("Training runner ended without a complete or explicitly stopped manifest")
            if time.monotonic() - started > 9 * 3600:
                raise RuntimeError("Training runner exceeded the 9-hour monitor limit")
            time.sleep(30)
        state["status"] = "evaluating"
        atomic_json(state_path, state)
        environment = runtime_environment()
        environment["AEROWALL_ISAACSIM_PATH"] = str(Path(environment["AEROWALL_ISAACSIM_PATH"]).resolve())
        environment["OMP_NUM_THREADS"] = environment["MKL_NUM_THREADS"] = "4"
        log_path = output_dir / "evaluation-driver.log"
        with log_path.open("w") as log:
            process = subprocess.run([str(PYTHON), "--plain", str(EVALUATE),
                                      "--output-dir", str(output_dir)],
                                     cwd=ROOT, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT)
        evaluation_state = output_dir / "evaluation-state.json"
        if process.returncode:
            raise RuntimeError(f"Evaluation exited {process.returncode}; inspect {log_path}")
        if not evaluation_state.exists():
            raise RuntimeError("Evaluation did not record its state")
        evaluation = json.loads(evaluation_state.read_text())
        if evaluation.get("status") != "evaluation_complete":
            state.update(status="evaluation_incomplete", evaluation_status=evaluation.get("status"))
            atomic_json(state_path, state)
            return
        summary = output_dir / "summary.json"
        if not summary.exists() or json.loads(summary.read_text()).get("status") != "passed":
            raise RuntimeError("Evaluation finished without a complete summary")
        state.update(status="complete", summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest())
        atomic_json(state_path, state)
    except Exception as error:
        state.update(status="failed", error=repr(error))
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
