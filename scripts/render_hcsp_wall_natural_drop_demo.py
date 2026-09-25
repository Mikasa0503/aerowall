"""Render the learned-policy demo through its first natural terminal event."""
import argparse
import json
from pathlib import Path

from render_hcsp_wall_recenter_demo import (
    EARLY_CHECKPOINT,
    MIDDLE_CHECKPOINT,
    ROOT,
    RUNS,
    PYTHON,
    CONFIG,
    run_evaluation,
)

SEED = 5001


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-horizon-steps",
        type=int,
        default=4800,
        help="Safety cap; the recorded final segment must end before this cap on an actual task failure.",
    )
    args = parser.parse_args()
    if args.final_horizon_steps <= 1200:
        parser.error("--final-horizon-steps must exceed the previous 1200-step cutoff")

    formal = json.loads((RUNS / "formal-summary.json").read_text())
    assert formal["status"] == "passed"
    selected_group = formal["selection"]["selected_demo_group"]
    checkpoint = Path(formal["selection"]["groups"][selected_group]["checkpoint"])
    assert EARLY_CHECKPOINT.exists() and MIDDLE_CHECKPOINT.exists()

    early_report = run_evaluation("early-visual-fix-900", EARLY_CHECKPOINT, 101, rgb=True)
    middle_report = run_evaluation("middle-visual-fix-900", MIDDLE_CHECKPOINT, 101, rgb=True)
    assert early_report["outcomes"][0]["rallies"] == 1
    assert middle_report["outcomes"][0]["rallies"] == 2

    final_config = json.loads(CONFIG.read_text())
    final_config["horizon_steps"] = args.final_horizon_steps
    final_config["end_on_ball_drop"] = True
    final_config_path = RUNS / f"natural-drop-h{args.final_horizon_steps}.json"
    final_config_path.write_text(json.dumps(final_config, indent=2) + "\n")

    final_name = f"{selected_group.lower()}-natural-ball-drop-final-{SEED}-h{args.final_horizon_steps}"
    final_report = run_evaluation(
        final_name, checkpoint, SEED, rgb=True, experiment_config=final_config_path
    )
    outcome = final_report["outcomes"][0]
    if outcome["timed_out"] or outcome["failure_reason"] != "ball_ground":
        raise RuntimeError(
            f"Ball-ground termination was not reached before the {args.final_horizon_steps}-step safety cap; "
            f"rallies={outcome['rallies']}, timed_out={outcome['timed_out']}"
        )
    assert outcome["failure_reason"] != "timeout_or_unfinished"
    assert outcome["contact_audit_passed"]

    output_dir = ROOT / f"artifacts/hcsp-wall-recenter-demo-ball-drop-h{args.final_horizon_steps}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing artifact directory: {output_dir}")
    segments = {
        "segments": [
            {
                "role": "early",
                "report": str(RUNS / "early-visual-fix-900.json"),
                "start_frame": 0,
                "end_frame": min(125, int(early_report["rgb"]["frames"])),
                "minimum_rallies": 1,
            },
            {
                "role": "middle",
                "report": str(RUNS / "middle-visual-fix-900.json"),
                "start_frame": 0,
                "end_frame": int(middle_report["rgb"]["frames"]),
                "minimum_rallies": 2,
            },
            {
                "role": "final",
                "report": str(RUNS / f"{final_name}.json"),
                "start_frame": 0,
                "end_frame": int(final_report["rgb"]["frames"]),
                "minimum_rallies": int(outcome["rallies"]),
            },
        ]
    }
    segment_path = RUNS / f"natural-drop-demo-segments-h{args.final_horizon_steps}.json"
    segment_path.write_text(json.dumps(segments, indent=2) + "\n")
    import subprocess

    subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts/compose_clean_hcsp_wall_demo.py"),
            "--segments",
            str(segment_path),
            "--output",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    result = {
        "status": "passed",
        "selected_group": selected_group,
        "presentation_seed": SEED,
        "final_rallies": int(outcome["rallies"]),
        "final_caps": int(outcome["caps"]),
        "final_walls": int(outcome["walls"]),
        "final_policy_steps": int(outcome["policy_steps"]),
        "final_failure": outcome["failure_reason"],
        "final_timed_out": bool(outcome["timed_out"]),
        "final_safe10": bool(outcome["safe10"]),
        "video": str(output_dir / "HCSP-wall-learning-demo-clean.mp4"),
        "video_seconds": manifest["probe"]["format"]["duration"],
        "video_sha256": manifest["demo_sha256"],
        "safety_horizon_steps": args.final_horizon_steps,
        "visual_fixes": [
            "hid the referenced sphere-light emitter while preserving ambient lighting",
            "raised camera framing while preserving view direction",
        ],
        "natural_ending": "recorded through ball-ground contact before the safety horizon",
    }
    (output_dir / "delivery-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
