"""Run a frozen, paired wall-reward comparison on the Isaac Sim host."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from project_paths import PYTHON, ROOT, runtime_environment


SPEC = ROOT / "configs/wall_reward_ablation.json"
BASE = ROOT / "configs/recenter_recovery.json"
TRAIN = ROOT / "scripts/train_aerowall_wall_rl.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-checkpoint", required=True, type=Path)
    parser.add_argument("--launch-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/wall-reward-ablation")
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    source = json.loads(BASE.read_text())
    if digest(ROOT / "scripts/aerowall_wall_reward_logic.py") != spec["reward_logic_sha256"]:
        raise ValueError("Reward formula source differs from the frozen experiment")
    if digest(args.actor_checkpoint) != spec["actor_checkpoint_sha256"]:
        raise ValueError("Actor checkpoint hash differs from the frozen C350 selection")
    if digest(args.launch_checkpoint) != spec["launch_checkpoint_sha256"]:
        raise ValueError("Launch checkpoint hash differs from the frozen source")
    if source["reward_variant"] != "recenter":
        raise ValueError("Existing recenter reward must be enabled in both groups")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    spec_sha = digest(SPEC)
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state["spec_sha256"] != spec_sha:
            raise ValueError("A different experiment already uses this output directory")
    else:
        state = {"spec_sha256": spec_sha, "jobs": [], "total_gpu_seconds": 0.0,
                 "actor_checkpoint": str(args.actor_checkpoint.resolve()),
                 "launch_checkpoint": str(args.launch_checkpoint.resolve())}
        atomic_json(manifest, state)
    env = runtime_environment()
    # The launcher filters Isaac paths after realpath normalization. Resolve
    # a shared-runtime symlink before invoking it from this isolated checkout.
    env["AEROWALL_ISAACSIM_PATH"] = str(Path(env["AEROWALL_ISAACSIM_PATH"]).resolve())
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "4"
    for seed_index, seed in enumerate(spec["training_seeds"]):
        order = list(spec["variants"])
        if seed_index % 2:
            order.reverse()
        for variant in order:
            name = f"seed-{seed}-{variant}"
            report = output_dir / f"{name}.json"
            config = output_dir / f"{name}.config.json"
            runtime = json.loads(json.dumps(source))
            runtime["horizon_steps"] = spec["horizon_steps"]
            runtime["reward_variant"] = "recenter"
            runtime["wall_reward_variant"] = variant
            runtime["training"]["seed"] = seed
            runtime["training"]["max_updates_per_group"] = spec["training_updates"]
            runtime["training"]["max_gpu_seconds_per_group"] = spec["max_total_gpu_seconds"]
            runtime["training"]["max_total_gpu_seconds"] = spec["max_total_gpu_seconds"]
            config_text = json.dumps(runtime, indent=2, sort_keys=True) + "\n"
            if config.exists() and config.read_text() != config_text:
                raise ValueError(f"Frozen config changed: {config}")
            config.write_text(config_text)
            if report.exists() and json.loads(report.read_text()).get("status") == "passed":
                continue
            remaining = spec["max_total_gpu_seconds"] - state["total_gpu_seconds"]
            if remaining <= 0:
                state["status"] = "budget_stopped"
                atomic_json(manifest, state)
                return
            command = [str(PYTHON), str(TRAIN), "--output", str(report),
                       "--updates", str(spec["training_updates"]),
                       "--num-envs", str(source["training"]["num_envs"]),
                       "--seed", str(seed), "--stage", "RALLY",
                       "--launch-checkpoint", str(args.launch_checkpoint),
                       "--experiment-config", str(config), "--group", "B",
                       "--save-every", str(spec["save_every_updates"]),
                       "--max-gpu-seconds", str(remaining)]
            training_state = report.with_suffix(".state.pt")
            if training_state.exists():
                command.extend(["--training-state", str(training_state)])
                prior = json.loads(report.read_text()).get("cumulative_gpu_seconds", 0.0)
                command[command.index("--max-gpu-seconds") + 1] = str(remaining + prior)
            else:
                command.extend(["--actor-warmstart", str(args.actor_checkpoint)])
            started = time.monotonic()
            log = output_dir / f"{name}.log"
            with log.open("a") as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - started
            state["total_gpu_seconds"] += elapsed
            payload = json.loads(report.read_text()) if report.exists() else {}
            state["jobs"].append({"name": name, "returncode": result.returncode,
                                  "seconds": elapsed, "status": payload.get("status"),
                                  "updates_completed": payload.get("updates_completed"),
                                  "initial_policy_sha256": payload.get("initial_policy_sha256"),
                                  "checkpoint_sha256": payload.get("checkpoint_sha256")})
            atomic_json(manifest, state)
            if result.returncode or payload.get("status") not in ("passed", "budget_stopped"):
                state["status"] = "failed"
                atomic_json(manifest, state)
                raise RuntimeError(f"Training failed; inspect {log}")
            if payload["status"] == "budget_stopped":
                state["status"] = "budget_stopped"
                atomic_json(manifest, state)
                return
        pair = [json.loads((output_dir / f"seed-{seed}-{v}.json").read_text())
                for v in spec["variants"]]
        if pair[0]["initial_policy_sha256"] != pair[1]["initial_policy_sha256"]:
            state["status"] = "failed_initial_state_mismatch"
            atomic_json(manifest, state)
            raise RuntimeError(f"Paired initial states differed for seed {seed}")
    state["status"] = "training_complete"
    atomic_json(manifest, state)


if __name__ == "__main__":
    main()
