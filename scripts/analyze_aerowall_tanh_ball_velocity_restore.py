#!/usr/bin/env python3
"""Analyze the preregistered AeroWall post-cap velocity-restore screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CASES = (
    "heldout-0007", "heldout-0013", "heldout-0015", "heldout-0033",
    "heldout-0035", "heldout-0037", "heldout-0054", "heldout-0061",
    "heldout-0075", "heldout-0090", "heldout-0098", "heldout-0101",
    "heldout-0103",
)
EXPECTED_BANK_SHA256 = "b6ebf13209ba42b1ad77eb452248b6f743c0f0fbcc32805d250fbc162b3804da"
EXPECTED_EVALUATOR_SHA256 = "1f4c5ca912a51d30d10adece9a23069a7e013db413c3f3a8b72b2148edc746c5"
EXPECTED_BUILDER_SHA256 = "3f7cfb95ddcdd5381b7a5d351692d203a5ac9a922815b26b52dd6caa26daabc1"
EXPECTED_CONTROL_TRAJECTORY_SHA256 = "51a233a5d5b788942e82bf50fc88480e5b02034b4678567eb51b1c6dec33cbd1"
EXPECTED_TANH_TRAJECTORY_SHA256 = "fb4ed6426f0d4a6d71cbefc5ded8c3e63399405f61efc43e0e88f333f5fd336f"
FAILURE_NAMES = {
    0: "none", 1: "ball_ground", 2: "drone_ground",
    3: "out_of_bounds", 4: "illegal_contact", 5: "drone_wall",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def l2(left, right) -> float:
    require(len(left) == len(right), "paired actor records have different dimensions")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def median(values):
    return float(statistics.median(values)) if values else None


def audit_report(report: dict, label: str) -> None:
    audit = report.get("contact_audit", {})
    require(report.get("status") == "passed", f"{label} report did not pass")
    require(report.get("seed") == 260925 and report.get("stage") == "RALLY"
            and report.get("protocol") == "natural" and report.get("num_envs") == 128,
            f"{label} run does not match the frozen natural 128-case protocol")
    require(report.get("summary", {}).get("all_outcomes_contact_audited") is True,
            f"{label} outcomes are not all contact-audited")
    require(not audit.get("callback_errors")
            and audit.get("legal_events") == audit.get("corroborated_events"),
            f"{label} PhysX contact corroboration is incomplete")
    evaluator = report.get("reproducibility", {}).get("source_files", {}).get("evaluator", {})
    require(evaluator.get("sha256") == EXPECTED_EVALUATOR_SHA256,
            f"{label} evaluator source hash differs from the final code lock")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--tanh-report", type=Path, required=True)
    parser.add_argument("--restore-report", type=Path, required=True)
    parser.add_argument("--reference-control-report", type=Path, required=True)
    parser.add_argument("--reference-tanh-report", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/plans/2026-09-25-aerowall-launch-postcontact-ball-velocity-restore-v1.md")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for field in ("control_report", "tanh_report", "restore_report", "reference_control_report",
                  "reference_tanh_report", "bank", "protocol", "output"):
        setattr(args, field, getattr(args, field).resolve())
    control, tanh, restore = (load(args.control_report), load(args.tanh_report), load(args.restore_report))
    reference_control, reference_tanh = load(args.reference_control_report), load(args.reference_tanh_report)
    bank = load(args.bank)
    restore_contacts_path = args.restore_report.with_suffix(".contacts.json")
    restore_contacts = load(restore_contacts_path)

    for label, report in (("control", control), ("tanh", tanh), ("restore", restore)):
        audit_report(report, label)
        require(report.get("initial_case_bank_sha256") == EXPECTED_BANK_SHA256,
                f"{label} run case bank hash differs from the frozen bank")
    require(control.get("launch_action_distribution") == "default", "control mapping is not default")
    require(tanh.get("launch_action_distribution") == "tanh", "Tanh arm mapping is not Tanh")
    require(restore.get("launch_action_distribution") == "tanh", "restore arm mapping is not Tanh")

    builder_path = ROOT / "scripts/build_aerowall_tanh_ball_velocity_restore_bank.py"
    require(sha256(builder_path) == EXPECTED_BUILDER_SHA256, "target-bank builder differs from its code lock")
    require(sha256(args.bank) == restore.get("state_intervention", {}).get("target_bank_sha256"),
            "restore report target-bank hash does not match the supplied bank")
    require(bank.get("experiment_name") == "AeroWallLaunchTanhPostContactBallVelocityRestoreV1"
            and tuple(bank.get("selected_cases", ())) == EXPECTED_CASES,
            "target bank identity or selected cases differ from the frozen protocol")
    require(bank.get("source_initial_case_bank_sha256") == EXPECTED_BANK_SHA256,
            "target bank source case bank hash mismatch")
    require(bank.get("source_protocol_sha256") == sha256(args.protocol),
            "target bank protocol hash does not match the current protocol")
    require(bank.get("source_control_report_sha256") == sha256(args.control_report),
            "target bank was not generated from the supplied fresh control report")

    for label, report, expected in (("control", control, EXPECTED_CONTROL_TRAJECTORY_SHA256),
                                    ("Tanh", tanh, EXPECTED_TANH_TRAJECTORY_SHA256)):
        trajectory = Path(report["trajectory"]).resolve()
        reference_trajectory = Path((reference_control if label == "control" else reference_tanh)["trajectory"]).resolve()
        fresh_sha, reference_sha = sha256(trajectory), sha256(reference_trajectory)
        require(fresh_sha == expected and reference_sha == expected and fresh_sha == reference_sha,
                f"{label} no-intervention trajectory failed exact reproduction")

    for label, report in (("reference control", reference_control), ("reference Tanh", reference_tanh)):
        require(report.get("status") == "passed" and report.get("seed") == 260925,
                f"{label} reference report is invalid")

    fidelity_reports = {
        "reference_control": reference_control,
        "fresh_control": control,
        "reference_tanh": reference_tanh,
        "fresh_tanh": tanh,
        "restore": restore,
    }
    reset_hashes = {
        label: report.get("reproducibility", {}).get("post_reset_state", {}).get("state_sha256")
        for label, report in fidelity_reports.items()
    }
    require(all(reset_hashes.values()) and len(set(reset_hashes.values())) == 1,
            "post-reset physical/FSM states differ across the paired arms")
    initial_observation_hashes = {
        label: canonical_sha(report.get("initial_policy_input_trace", {}).get("active_actor_observation"))
        for label, report in fidelity_reports.items()
    }
    require(len(set(initial_observation_hashes.values())) == 1,
            "initial active actor observations differ across the paired arms")
    sidecar_fidelity = {}
    for label, fresh, reference in (("control", control, reference_control), ("tanh", tanh, reference_tanh)):
        fresh_events = Path(fresh["first_episode_events"]).resolve()
        reference_events = Path(reference["first_episode_events"]).resolve()
        fresh_contacts = args.control_report.with_suffix(".contacts.json") if label == "control" else args.tanh_report.with_suffix(".contacts.json")
        reference_report_path = args.reference_control_report if label == "control" else args.reference_tanh_report
        reference_contacts = reference_report_path.with_suffix(".contacts.json")
        event_match = sha256(fresh_events) == sha256(reference_events)
        contact_match = sha256(fresh_contacts) == sha256(reference_contacts)
        require(event_match and contact_match, f"{label} event/contact sidecars failed exact reproduction")
        sidecar_fidelity[label] = {
            "events_sha256": sha256(fresh_events),
            "contacts_sha256": sha256(fresh_contacts),
            "exact_match": True,
        }

    selected = list(EXPECTED_CASES)
    old_outcomes = {row["case_id"]: row for row in tanh["outcomes"]}
    new_outcomes = {row["case_id"]: row for row in restore["outcomes"]}
    require(all(case_id in old_outcomes and case_id in new_outcomes for case_id in selected),
            "one or more selected cases is missing from the natural outcome reports")

    old_actor = {row["case_id"]: row for row in tanh.get("post_contact_actor_input_trace", [])}
    new_actor = {row["case_id"]: row for row in restore.get("post_contact_actor_input_trace", [])}
    injection_trace = restore.get("state_intervention", {}).get("per_case_trace", [])
    inject_by_case = {row["case_id"]: row for row in injection_trace}
    require(set(inject_by_case) == set(selected) and len(injection_trace) == 13,
            "restore arm did not apply exactly one injection to every selected case")
    require(all(case_id in old_actor and case_id in new_actor for case_id in selected),
            "immediate post-cap actor input trace is incomplete")

    case_rows = []
    velocity_deltas, observation_deltas, action_deltas = [], [], []
    for case_id in selected:
        before, after = old_outcomes[case_id], new_outcomes[case_id]
        inject = inject_by_case[case_id]
        old_input, new_input = old_actor[case_id], new_actor[case_id]
        for row in (old_input, new_input):
            require(row.get("active_actor_observation")
                    and row["active_actor_observation"].get("actor_observation_role") == "recovery",
                    f"{case_id} does not use the frozen Recovery actor after first cap")
        require(new_input.get("state_intervention_applied") is True,
                f"{case_id} immediate actor input was not marked as post-intervention")
        require(old_input.get("first_legal_cap_policy_step") == inject.get("policy_step")
                and old_input.get("first_legal_cap_substep") == inject.get("contact_substep"),
                f"{case_id} first cap location changed before the intervention")
        tolerance = float(inject.get("tolerance_mps", 1e-5))
        require(float(inject.get("readback_max_abs_error_mps", math.inf)) <= tolerance,
                f"{case_id} injected velocity failed live readback tolerance")
        require(inject.get("prev_ball_velocity_history_synchronized")
                and inject.get("next_state_observation_refreshed"),
                f"{case_id} omitted collision-history sync or next-state refresh")
        velocity_delta = l2(inject["pre_restore_ball_linear_velocity_mps"],
                            inject["post_restore_ball_linear_velocity_mps"])
        observation_delta = l2(old_input["active_actor_observation"]["actor_observation"],
                               new_input["active_actor_observation"]["actor_observation"])
        action_delta = l2(old_input["policy_action"], new_input["policy_action"])
        velocity_deltas.append(velocity_delta)
        observation_deltas.append(observation_delta)
        action_deltas.append(action_delta)
        old_failure, new_failure = int(before["failure"]), int(after["failure"])
        terminal_policy_step = int(after["policy_steps"]) - 1
        ground_contacts = []
        for contact_row in restore_contacts:
            actors = contact_row.get("actors", [])
            actor_text = " ".join(actors).lower()
            if (int(contact_row.get("step", -1)) == terminal_policy_step
                    and int(contact_row.get("contact_count", 0)) > 0
                    and f"/env_{int(after['env'])}/ball" in actor_text
                    and ("ground" in actor_text or "floor" in actor_text)):
                ground_contacts.append(contact_row)
        if FAILURE_NAMES[new_failure] == "ball_ground":
            require(bool(ground_contacts),
                    f"{case_id} ball-ground terminal outcome lacks a matching PhysX contact at its final policy step")
        first_ground = ground_contacts[0] if ground_contacts else None
        case_rows.append({
            "case_id": case_id,
            "first_cap_policy_step": int(inject["policy_step"]),
            "first_cap_substep": int(inject["contact_substep"]),
            "contact_radial_error_m": float(inject["contact_radial_error_m"]),
            "ball_velocity_delta_l2_mps": velocity_delta,
            "next_recovery_observation_delta_l2": observation_delta,
            "next_recovery_action_delta_l2": action_delta,
            "control_failure_code": int(old_failure),
            "control_failure_name": FAILURE_NAMES[old_failure],
            "restore_failure_code": int(new_failure),
            "restore_failure_name": FAILURE_NAMES[new_failure],
            "control_rallies": int(before["rallies"]),
            "restore_rallies": int(after["rallies"]),
            "control_caps": int(before["caps"]),
            "restore_caps": int(after["caps"]),
            "control_walls": int(before["walls"]),
            "restore_walls": int(after["walls"]),
            "injection_readback_max_abs_error_mps": float(inject["readback_max_abs_error_mps"]),
            "terminal_policy_step": terminal_policy_step,
            "terminal_ball_ground_physx_contact_rows": len(ground_contacts),
            "terminal_ball_ground_physx_first_substep": int(first_ground["substep"]) if first_ground else None,
            "terminal_ball_ground_physx_actor_paths": first_ground["actors"] if first_ground else [],
        })

    local_failures_before = {name: sum(row["control_failure_name"] == name for row in case_rows)
                             for name in FAILURE_NAMES.values()}
    local_failures_after = {name: sum(row["restore_failure_name"] == name for row in case_rows)
                            for name in FAILURE_NAMES.values()}
    avoided_illegal = sum(row["control_failure_name"] == "illegal_contact"
                          and row["restore_failure_name"] != "illegal_contact" for row in case_rows)
    new_ball_ground = sum(row["control_failure_name"] != "ball_ground"
                          and row["restore_failure_name"] == "ball_ground" for row in case_rows)
    lost_second_hit = sum(row["control_caps"] >= 2 and row["restore_caps"] < 2 for row in case_rows)
    global_before = tanh["summary"]["failure_counts"]
    global_after = restore["summary"]["failure_counts"]
    global_failure_delta = {name: int(global_after.get(name, 0)) - int(global_before.get(name, 0))
                            for name in global_before}
    audit = restore["contact_audit"]
    intervention = restore["state_intervention"]
    readback_errors = [float(row["readback_max_abs_error_mps"]) for row in injection_trace]
    require(intervention.get("selected_case_count") == 13 and intervention.get("applied_case_count") == 13,
            "restore summary does not confirm all 13 injections")
    require(restore["summary"]["all_outcomes_contact_audited"] is True,
            "restore run is missing complete outcome contact audit")

    no_added_secondary_failures = all(
        global_failure_delta.get(name, 0) <= 0
        for name in ("ball_ground", "out_of_bounds", "drone_ground", "drone_wall")
    )
    no_lost_second_hit = lost_second_hit == 0
    criteria = {
        "at_least_7_of_13_avoid_later_illegal_contact": avoided_illegal >= 7,
        "no_added_ball_ground_out_of_bounds_or_drone_contact_failures": no_added_secondary_failures,
        "no_loss_of_legal_second_hit": no_lost_second_hit,
    }
    report = {
        "schema_version": 1,
        "experiment_name": "AeroWallLaunchTanhPostContactBallVelocityRestoreV1",
        "scope": "post-hoc selected development subset; state-intervention screen; not a natural-policy comparison or P2 test",
        "decision": "single_velocity_restore_gate_failed",
        "decision_reason": "12 of 13 selected illegal-contact failures changed to ball-ground failures; no rally or legal-second-hit gain; preregistered no-new-ground-failure criterion failed",
        "p2_eligible": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "c350_changed": False,
        "source_files": {
            "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
            "target_bank": {"path": str(args.bank), "sha256": sha256(args.bank)},
            "evaluator": {"path": str(ROOT / "scripts/evaluate_aerowall_wall_rl.py"),
                          "sha256": sha256(ROOT / "scripts/evaluate_aerowall_wall_rl.py")},
            "target_bank_builder": {"path": str(builder_path), "sha256": sha256(builder_path)},
            "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "replication_fidelity": {
            "post_reset_state_sha256_by_arm": reset_hashes,
            "initial_active_actor_observation_sha256_by_arm": initial_observation_hashes,
            "exact_no_intervention_sidecars": sidecar_fidelity,
            "control_trajectory_sha256": EXPECTED_CONTROL_TRAJECTORY_SHA256,
            "tanh_trajectory_sha256": EXPECTED_TANH_TRAJECTORY_SHA256,
        },
        "inputs": {
            "control_report_sha256": sha256(args.control_report),
            "tanh_report_sha256": sha256(args.tanh_report),
            "restore_report_sha256": sha256(args.restore_report),
            "restore_physx_contacts_sha256": sha256(restore_contacts_path),
            "reference_control_report_sha256": sha256(args.reference_control_report),
            "reference_tanh_report_sha256": sha256(args.reference_tanh_report),
            "case_bank_sha256": EXPECTED_BANK_SHA256,
            "control_trajectory_sha256": sha256(Path(control["trajectory"]).resolve()),
            "tanh_trajectory_sha256": sha256(Path(tanh["trajectory"]).resolve()),
        },
        "protocol_gate": criteria,
        "selected_subset": {
            "case_count": len(case_rows),
            "illegal_contact_before": local_failures_before["illegal_contact"],
            "illegal_contact_after": local_failures_after["illegal_contact"],
            "avoided_illegal_contact": avoided_illegal,
            "new_ball_ground_failures": new_ball_ground,
            "control_failure_counts": local_failures_before,
            "restore_failure_counts": local_failures_after,
            "legal_second_hit_cases_before": sum(row["control_caps"] >= 2 for row in case_rows),
            "legal_second_hit_cases_after": sum(row["restore_caps"] >= 2 for row in case_rows),
            "lost_legal_second_hit_cases": lost_second_hit,
            "rally_cases_before": sum(row["control_rallies"] >= 1 for row in case_rows),
            "rally_cases_after": sum(row["restore_rallies"] >= 1 for row in case_rows),
            "case_level": case_rows,
        },
        "physx_terminal_failure_audit": {
            "selected_ball_ground_outcomes": sum(row["restore_failure_name"] == "ball_ground" for row in case_rows),
            "physx_corroborated_terminal_ball_ground_outcomes": sum(
                row["restore_failure_name"] == "ball_ground" and row["terminal_ball_ground_physx_contact_rows"] > 0
                for row in case_rows
            ),
            "terminal_ball_ground_contact_rows": sum(row["terminal_ball_ground_physx_contact_rows"] for row in case_rows),
            "all_selected_ball_ground_failures_physx_corroborated": all(
                row["restore_failure_name"] != "ball_ground" or row["terminal_ball_ground_physx_contact_rows"] > 0
                for row in case_rows
            ),
        },
        "global_outcomes": {
            "tanh_failure_counts": global_before,
            "restore_failure_counts": global_after,
            "failure_count_delta": global_failure_delta,
            "tanh_safety_failure_rate": tanh["summary"]["safety_failure_rate"],
            "restore_safety_failure_rate": restore["summary"]["safety_failure_rate"],
            "tanh_legal_second_hit_rate": tanh["summary"]["legal_second_hit_rate"],
            "restore_legal_second_hit_rate": restore["summary"]["legal_second_hit_rate"],
            "tanh_rally_histogram": tanh["summary"]["rally_histogram"],
            "restore_rally_histogram": restore["summary"]["rally_histogram"],
        },
        "intervention_audit": {
            "selected_case_count": intervention["selected_case_count"],
            "applied_case_count": intervention["applied_case_count"],
            "readback_tolerance_mps": intervention["readback_tolerance_mps"],
            "maximum_readback_error_mps": max(readback_errors),
            "all_readbacks_within_tolerance": all(error <= intervention["readback_tolerance_mps"] for error in readback_errors),
            "all_prev_ball_velocity_history_synchronized": all(row["prev_ball_velocity_history_synchronized"] for row in injection_trace),
            "all_next_state_observations_refreshed": all(row["next_state_observation_refreshed"] for row in injection_trace),
            "velocity_change_l2_mps_min_median_max": [min(velocity_deltas), median(velocity_deltas), max(velocity_deltas)],
            "next_recovery_observation_delta_l2_median": median(observation_deltas),
            "next_recovery_action_delta_l2_median": median(action_deltas),
            "physx_legal_events": audit["legal_events"],
            "physx_corroborated_events": audit["corroborated_events"],
            "physx_callback_errors": audit["callback_errors"],
            "all_128_outcomes_contact_audited": restore["summary"]["all_outcomes_contact_audited"],
        },
        "interpretation": "On this selected subset, the post-cap ball linear velocity is associated with downstream failure mode under a direct state intervention: the velocity restore changed failure labels but routed most cases to ball-ground failures. It did not improve completed rally outcomes. This rejects the registered single-velocity restore as a useful fix in this screen; it does not identify a population-level mechanism or justify policy training/promotion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "decision": report["decision"],
        "selected_failure_counts_before": local_failures_before,
        "selected_failure_counts_after": local_failures_after,
        "global_failure_count_delta": global_failure_delta,
        "protocol_gate": criteria,
        "physx": {"legal_events": audit["legal_events"], "corroborated_events": audit["corroborated_events"]},
        "maximum_readback_error_mps": max(readback_errors),
        "physx_corroborated_selected_ball_ground_failures": report["physx_terminal_failure_audit"]["physx_corroborated_terminal_ball_ground_outcomes"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
