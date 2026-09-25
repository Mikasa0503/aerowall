"""Evaluate frozen reward-ablation checkpoints and publish paired evidence."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from aerowall_wall_reward_logic import center_band_wall_reward
from project_paths import PYTHON, ROOT, runtime_environment


SPEC = ROOT / "configs/wall_reward_ablation.json"
EVAL = ROOT / "scripts/evaluate_aerowall_wall_rl.py"
SAFE_FAILURES = {"drone_ground", "drone_wall", "illegal_contact"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def checked_report(path, checkpoint, variant):
    report = json.loads(path.read_text())
    if report.get("status") != "passed" or report.get("checkpoint_sha256") != digest(checkpoint):
        raise RuntimeError(f"Unusable evaluation report: {path}")
    if report.get("reward_logic_sha256") != digest(ROOT / "scripts/aerowall_wall_reward_logic.py"):
        raise RuntimeError(f"Evaluation used a different reward formula: {path}")
    audit = report["contact_audit"]
    if (audit["legal_events"] != audit["corroborated_events"]
            or audit["wall_events"] != audit["corroborated_wall_events"]
            or audit["callback_errors"]
            or not all(row["contact_audit_passed"] for row in report["outcomes"])):
        raise RuntimeError(f"Physical contact audit failed: {path}")
    for line in Path(report["first_episode_events"]).read_text().splitlines():
        event = json.loads(line)
        expected = (10.0 if variant == "constant"
                    else center_band_wall_reward(event["ball_position"][1]))
        scored = float(event.get("wall_reward", float("nan")))
        if not math.isfinite(scored):
            raise RuntimeError(f"Missing or invalid wall reward: {path}")
        if event.get("expected_wall"):
            if abs(scored - expected) > 1e-5:
                raise RuntimeError(f"Wall reward/event mismatch: {path}")
        elif abs(scored) > 1e-5:
            raise RuntimeError(f"Non-wall event received a wall reward: {path}")
    return report


def metrics(reports):
    process_prefixes = [statistics.mean(row["centered_prefix_rallies"] for row in report["outcomes"])
                        for report in reports]
    outcomes = [row for report in reports for row in report["outcomes"]]
    wall_sequences = [row["wall_y_sequence"] for row in outcomes]
    wall_ordinal = []
    for ordinal in range(max(map(len, wall_sequences), default=0)):
        values = [sequence[ordinal] for sequence in wall_sequences if len(sequence) > ordinal]
        wall_ordinal.append({"ordinal": ordinal + 1, "count": len(values),
                             "mean_y_m": statistics.mean(values),
                             "mean_abs_y_m": statistics.mean(map(abs, values))})
    return {
        "process_mean_centered_prefix": process_prefixes,
        "mean_centered_prefix": statistics.mean(process_prefixes),
        "mean_rallies": statistics.mean(row["rallies"] for row in outcomes),
        "safe10_rate": statistics.mean(row["safe10"] for row in outcomes),
        "safe15_rate": statistics.mean(row["safe15"] for row in outcomes),
        "safety_failure_rate": statistics.mean(row["failure_reason"] in SAFE_FAILURES for row in outcomes),
        "failure_reasons": {reason: sum(row["failure_reason"] == reason for row in outcomes)
                            for reason in sorted({row["failure_reason"] for row in outcomes})},
        "wall_y_by_ordinal": wall_ordinal,
        "episodes": len(outcomes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/wall-reward-ablation")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    spec = json.loads(SPEC.read_text())
    if digest(ROOT / "scripts/aerowall_wall_reward_logic.py") != spec["reward_logic_sha256"]:
        raise RuntimeError("Reward formula source differs from the frozen experiment")
    training = json.loads((output_dir / "manifest.json").read_text())
    if training.get("status") != "training_complete" or training["spec_sha256"] != digest(SPEC):
        raise RuntimeError("All six matched training runs must finish before evaluation")
    launch = Path(training["launch_checkpoint"])
    actor = Path(training["actor_checkpoint"])
    if digest(launch) != spec["launch_checkpoint_sha256"] or digest(actor) != spec["actor_checkpoint_sha256"]:
        raise RuntimeError("Frozen source checkpoint changed")
    eval_state_path = output_dir / "evaluation-state.json"
    state = json.loads(eval_state_path.read_text()) if eval_state_path.exists() else {
        "spec_sha256": digest(SPEC), "jobs": [], "evaluation_seconds": 0.0}
    if state["spec_sha256"] != digest(SPEC):
        raise RuntimeError("Evaluation state belongs to a different spec")
    env = runtime_environment()
    env["AEROWALL_ISAACSIM_PATH"] = str(Path(env["AEROWALL_ISAACSIM_PATH"]).resolve())
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "4"
    experiments = [(f"seed-{seed}-{variant}", variant,
                    output_dir / f"seed-{seed}-{variant}.pt",
                    output_dir / f"seed-{seed}-{variant}.config.json")
                   for seed in spec["training_seeds"] for variant in spec["variants"]]
    experiments.append(("frozen-c350", "constant", actor,
                        output_dir / f"seed-{spec['training_seeds'][0]}-constant.config.json"))
    for name, variant, checkpoint, config in experiments:
        if not checkpoint.exists():
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")
        for eval_seed in spec["evaluation_seeds"]:
            report_path = output_dir / f"eval-{name}-{eval_seed}.json"
            if report_path.exists() and json.loads(report_path.read_text()).get("status") == "passed":
                checked_report(report_path, checkpoint, variant)
                continue
            if training["total_gpu_seconds"] + state["evaluation_seconds"] >= spec["max_total_gpu_seconds"] - 120:
                state["status"] = "budget_stopped"
                atomic_json(eval_state_path, state)
                return
            command = [str(PYTHON), str(EVAL), "--output", str(report_path),
                       "--checkpoint", str(checkpoint), "--launch-checkpoint", str(launch),
                       "--stage", "RALLY", "--num-envs", str(spec["evaluation_num_envs"]),
                       "--seed", str(eval_seed), "--experiment-config", str(config),
                       "--protocol", "natural"]
            started = time.monotonic()
            log = output_dir / f"eval-{name}-{eval_seed}.log"
            with log.open("w") as stream:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - started
            state["evaluation_seconds"] += elapsed
            state["jobs"].append({"name": name, "evaluation_seed": eval_seed,
                                  "returncode": process.returncode, "seconds": elapsed})
            atomic_json(eval_state_path, state)
            checked_report(report_path, checkpoint, variant)
            if process.returncode:
                from finalize_hcsp_wall_recenter_experiment import post_report_close_crash
                if not post_report_close_crash(log.read_text(errors="replace")):
                    raise RuntimeError(f"Evaluation exited {process.returncode}: {log}")
    result = {}
    for name, variant, checkpoint, _ in experiments:
        reports = [checked_report(output_dir / f"eval-{name}-{seed}.json", checkpoint, variant)
                   for seed in spec["evaluation_seeds"]]
        result[name] = metrics(reports)
    differences = {}
    for seed in spec["training_seeds"]:
        baseline = result[f"seed-{seed}-constant"]
        candidate = result[f"seed-{seed}-center_band"]
        differences[str(seed)] = candidate["mean_centered_prefix"] - baseline["mean_centered_prefix"]
    baseline_rallies = statistics.mean(result[f"seed-{s}-constant"]["mean_rallies"]
                                       for s in spec["training_seeds"])
    candidate_rallies = statistics.mean(result[f"seed-{s}-center_band"]["mean_rallies"]
                                        for s in spec["training_seeds"])
    baseline_safety = statistics.mean(result[f"seed-{s}-constant"]["safety_failure_rate"]
                                      for s in spec["training_seeds"])
    candidate_safety = statistics.mean(result[f"seed-{s}-center_band"]["safety_failure_rate"]
                                       for s in spec["training_seeds"])
    candidate_prefix = statistics.mean(result[f"seed-{s}-center_band"]["mean_centered_prefix"]
                                       for s in spec["training_seeds"])
    decision = {
        "paired_prefix_improvement": statistics.mean(differences.values()) >= 2.0,
        "at_least_two_seeds_improved": sum(value > 0 for value in differences.values()) >= 2,
        "remaining_seed_drop_at_most_one": min(differences.values()) >= -1.0,
        "rally_retention": candidate_rallies >= 0.9 * baseline_rallies,
        "safety_noninferiority": candidate_safety <= baseline_safety,
        "frozen_c350_noninferiority": candidate_prefix >= result["frozen-c350"]["mean_centered_prefix"],
    }
    summary = {"status": "passed", "spec_sha256": digest(SPEC),
               "training_manifest_sha256": digest(output_dir / "manifest.json"),
               "paired_prefix_differences": differences, "metrics": result,
               "decision_checks": decision, "recommend_new_reward": all(decision.values()),
               "total_gpu_seconds": training["total_gpu_seconds"] + state["evaluation_seconds"]}
    atomic_json(output_dir / "summary.json", summary)
    lines = ["# AeroWall 真实碰墙奖励消融", "",
             "两组均从同一 C350 回接 actor 开始，保留预测回中奖励，只改变合法碰墙事件的计分。",
             "每个训练种子对照使用相同初始化与 150 次 PPO 更新。评测是固定投球；并行环境不代表不同来球。",
             "", "| 训练种子 | 固定碰墙奖励：居中前缀 | 中心带奖励：居中前缀 | 差值 |",
             "|---|---:|---:|---:|"]
    for seed in spec["training_seeds"]:
        base = result[f"seed-{seed}-constant"]["mean_centered_prefix"]
        new = result[f"seed-{seed}-center_band"]["mean_centered_prefix"]
        lines.append(f"| {seed} | {base:.2f} | {new:.2f} | {new-base:+.2f} |")
    lines += ["", f"冻结 C350 的居中前缀：{result['frozen-c350']['mean_centered_prefix']:.2f}。",
              f"配对差值均值：{statistics.mean(differences.values()):+.2f} 轮。",
              "", "| 预设保留条件 | 是否达到 |", "|---|---|"]
    lines += [f"| {name} | {'是' if passed else '否'} |" for name, passed in decision.items()]
    lines += ["", f"**结论：{'推荐新奖励' if summary['recommend_new_reward'] else '未证明新奖励更优，沿用旧奖励'}。**",
              "", "真实墙面横向落点按回合及存活数量列于 `summary.json`。图中后期落点只来自实际打到该回合的轨迹，需与样本数一同解读。",
              "本实验只有三个微调训练种子，且都源于同一个 C350 actor；不能据此宣称对新投球分布的泛化能力。", ""]
    (output_dir / "report.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for variant, label, color in (("constant", "Fixed +10", "#365c88"),
                                  ("center_band", "Measured center-band", "#c4512d")):
        sequences = [result[f"seed-{seed}-{variant}"]["wall_y_by_ordinal"]
                     for seed in spec["training_seeds"]]
        ordinates = sorted({row["ordinal"] for sequence in sequences for row in sequence})
        x, y = [], []
        for ordinal in ordinates:
            samples = [(row["mean_abs_y_m"], row["count"])
                       for sequence in sequences for row in sequence if row["ordinal"] == ordinal]
            if samples:
                x.append(ordinal)
                y.append(sum(value * count for value, count in samples) / sum(count for _, count in samples))
        ax.plot(x, y, marker="o", markersize=3, label=label, color=color)
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1, label="Center-band boundary")
    ax.set(xlabel="Wall contact ordinal", ylabel="Mean absolute wall y (m)",
           title="Wall drift among episodes reaching each contact")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wall-drift.png", dpi=180)
    plt.close(fig)
    state["status"] = "evaluation_complete"
    atomic_json(eval_state_path, state)


if __name__ == "__main__":
    main()
