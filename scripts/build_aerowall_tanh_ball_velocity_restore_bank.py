#!/usr/bin/env python3
"""Build exact paired post-step ball-velocity targets from an audited control run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CASES = (
    "heldout-0007", "heldout-0013", "heldout-0015", "heldout-0033",
    "heldout-0035", "heldout-0037", "heldout-0054", "heldout-0061",
    "heldout-0075", "heldout-0090", "heldout-0098", "heldout-0101",
    "heldout-0103",
)
BANK_SHA256 = "b6ebf13209ba42b1ad77eb452248b6f743c0f0fbcc32805d250fbc162b3804da"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--case-bank", type=Path, required=True)
    parser.add_argument("--phase-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/plans/2026-09-25-aerowall-launch-postcontact-ball-velocity-restore-v1.md")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.control_report = args.control_report.resolve()
    args.case_bank = args.case_bank.resolve()
    args.phase_report = args.phase_report.resolve()
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()

    report = load(args.control_report)
    case_bank = load(args.case_bank)
    phase = load(args.phase_report)
    if report.get("status") != "passed" or report.get("seed") != 260925:
        raise SystemExit("control report must be the passed seed-260925 replication")
    if report.get("stage") != "RALLY" or report.get("protocol") != "natural" or report.get("num_envs") != 128:
        raise SystemExit("control report must be the natural 128-environment RALLY run")
    if report.get("launch_action_distribution") != "default":
        raise SystemExit("control report must use the default Launch action distribution")
    if report.get("initial_case_bank_sha256") != BANK_SHA256 or sha256(args.case_bank) != BANK_SHA256:
        raise SystemExit("case bank SHA256 does not match the frozen development bank")
    if sha256(args.phase_report) != "9907c17713cc7221deecb170dd839a3780d7ed31169548d046c7209d4696bd76":
        raise SystemExit("phase-report SHA256 differs from the frozen selection source")

    selected = phase["safety_failure_case_changes"]["tanh_only_cases_with_first_contact_unchanged"]
    if tuple(selected) != EXPECTED_CASES:
        raise SystemExit("selected case IDs or order differ from the frozen protocol")
    if not report.get("summary", {}).get("all_outcomes_contact_audited"):
        raise SystemExit("control outcomes are not fully contact audited")
    audit = report.get("contact_audit", {})
    if audit.get("callback_errors") or audit.get("legal_events") != audit.get("corroborated_events"):
        raise SystemExit("control contact audit is incomplete")

    trajectory_path = Path(report["trajectory"]).resolve()
    events_path = Path(report["first_episode_events"]).resolve()
    contacts_path = args.control_report.with_suffix(".contacts.json")
    trajectory = np.load(trajectory_path)
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    contacts = load(contacts_path)
    outcomes = {item["case_id"]: item for item in report["outcomes"]}
    cases_in_bank = case_bank["cases"]
    env_by_case = {item["case_id"]: index for index, item in enumerate(cases_in_bank)}
    events_by_env = {}
    for event in events:
        events_by_env.setdefault(int(event["env"]), []).append(event)

    target_cases = []
    for case_id in EXPECTED_CASES:
        env = env_by_case.get(case_id)
        if env is None or outcomes.get(case_id, {}).get("env") != env:
            raise SystemExit(f"case-to-environment mapping mismatch for {case_id}")
        cap_events = [
            event for event in events_by_env.get(env, [])
            if bool(event.get("body")) and bool(event.get("legal_cap"))
        ]
        if not cap_events:
            raise SystemExit(f"missing legal first cap for {case_id}")
        contact = min(cap_events, key=lambda e: (int(e["policy_step"]), int(e.get("substep", 0))))
        step = int(contact["policy_step"])
        if step >= trajectory["ball_velocity"].shape[0] or not bool(trajectory["active"][step, env]):
            raise SystemExit(f"control trajectory does not contain the post-cap step for {case_id}")
        physx_confirmed = any(
            row.get("step") == step
            and row.get("substep") == int(contact["substep"])
            and row.get("contact_count", 0) > 0
            and any(path.endswith(f"/env_{env}/ball") for path in row.get("actors", []))
            and any("/IrisTest_0/" in path for path in row.get("actors", []))
            for row in contacts
        )
        if not physx_confirmed:
            raise SystemExit(f"control first cap lacks an exact PhysX contact match for {case_id}")
        target_velocity = trajectory["ball_velocity"][step, env, :3].astype(float).tolist()
        if len(target_velocity) != 3 or not all(np.isfinite(target_velocity)):
            raise SystemExit(f"invalid target ball velocity for {case_id}")
        target_cases.append({
            "case_id": case_id,
            "env": env,
            "control_first_cap_policy_step": step,
            "control_first_cap_substep": int(contact["substep"]),
            "control_contact_ball_velocity_after_mps": contact["ball_velocity_after"],
            "control_post_step_ball_linear_velocity_mps": target_velocity,
            "control_contact_physx_confirmed": True,
        })

    checkpoint_paths = {
        "launch": Path(report["launch_checkpoint"]).resolve(),
        "hit": Path(report["hit_checkpoint"]).resolve(),
        "recovery": Path(report["recovery_checkpoint"]).resolve(),
    }
    checkpoint_hashes = {role: sha256(path) for role, path in checkpoint_paths.items()}
    payload = {
        "schema_version": 1,
        "experiment_name": "AeroWallLaunchTanhPostContactBallVelocityRestoreV1",
        "bank_kind": "aerowall_launch_tanh_postcontact_ball_velocity_restore_v1",
        "scope": "post-hoc selected development cases; state-intervention targets only; no P2 or promotion use",
        "source_initial_case_bank_path": str(args.case_bank),
        "source_initial_case_bank_sha256": sha256(args.case_bank),
        "source_phase_report_path": str(args.phase_report),
        "source_phase_report_sha256": sha256(args.phase_report),
        "source_protocol_path": str(args.protocol),
        "source_protocol_sha256": sha256(args.protocol),
        "source_control_report_path": str(args.control_report),
        "source_control_report_sha256": sha256(args.control_report),
        "source_control_trajectory_path": str(trajectory_path),
        "source_control_trajectory_sha256": sha256(trajectory_path),
        "source_control_events_path": str(events_path),
        "source_control_events_sha256": sha256(events_path),
        "source_control_physx_contacts_path": str(contacts_path),
        "source_control_physx_contacts_sha256": sha256(contacts_path),
        "source_checkpoint_sha256": checkpoint_hashes,
        "seed": 260925,
        "num_envs": 128,
        "selected_cases": list(EXPECTED_CASES),
        "cases": target_cases,
        "target_semantics": "replace only ball linear velocity after the first legal cap policy step; sample the same-case control trajectory row at its first legal cap policy step",
        "decision_limits": {
            "minimum_cases_without_later_illegal_contact": 7,
            "require_no_new_ball_ground_out_of_bounds_drone_ground_or_drone_wall_failures": True,
            "require_no_loss_of_legal_second_hit": True,
            "p2_eligible": False,
            "training_authorized": False,
            "promotion_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "case_count": len(target_cases),
        "control_trajectory_sha256": payload["source_control_trajectory_sha256"],
        "checkpoint_sha256": checkpoint_hashes,
    }, indent=2))


if __name__ == "__main__":
    main()
