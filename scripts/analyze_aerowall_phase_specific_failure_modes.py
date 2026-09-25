#!/usr/bin/env python3
"""Synthesize existing AeroWall paired failure evidence without rerunning simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


INDEX = Path("artifacts/wall-skill-upgrade-v3/launch-diagnostics/analysis.json")
REPRO = Path("artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility")
DEFAULT_OUTPUT = REPRO / "aerowall-phase-specific-failure-mode-synthesis-v1-s9524.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verified_source(root: Path, label: str, relative_path: str, expected: str):
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    require(actual == expected, f"{label} hash mismatch: {actual} != {expected}")
    return {"path": relative_path, "sha256": actual}


def load_json(path: Path):
    return json.loads(path.read_text())


def compare_first_divergence_states(
    root: Path,
    paired_report: dict,
    ablation_report: dict,
    control_trajectory_path: Path,
    ablation_trajectory_path: Path,
):
    divergences = paired_report["paired_trajectory_comparison"][
        "first_action_divergence_by_case"
    ]
    control_envs = {row["case_id"]: int(row["env"]) for row in
                    load_json(root / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/legacy-control-launch-relv3-corrected-s9524.json")["outcomes"]}
    ablation_envs = {row["case_id"]: int(row["env"]) for row in ablation_report["outcomes"]}
    require(control_envs == ablation_envs, "Control and ablation env-to-case mappings differ")

    compared_fields = (
        "drone_state",
        "ball_position",
        "ball_velocity",
        "actuator_command",
        "motor_throttle",
        "rotor_thrust_local_z_n",
    )
    field_max = {field: 0.0 for field in compared_fields}
    action_deltas = []
    indices = []
    exact_case_ids = []
    with np.load(control_trajectory_path, allow_pickle=False) as control, np.load(
        ablation_trajectory_path, allow_pickle=False
    ) as ablation:
        require(control["active"].shape == ablation["active"].shape,
                "Control and ablation trajectory active masks differ in shape")
        for row in divergences:
            case_id = row["case_id"]
            index = int(row["trajectory_index"])
            env = control_envs[case_id]
            pre_action_index = index - 1
            require(pre_action_index >= 0, f"Invalid divergence index for {case_id}: {index}")
            require(bool(control["active"][pre_action_index, env]) and
                    bool(ablation["active"][pre_action_index, env]),
                    f"Pre-action row is not active for {case_id}")
            exact = True
            for field in compared_fields:
                left = np.asarray(control[field][pre_action_index, env])
                right = np.asarray(ablation[field][pre_action_index, env])
                delta = float(np.max(np.abs(left - right), initial=0.0))
                require(np.isfinite(delta), f"Non-finite state difference for {case_id}/{field}")
                field_max[field] = max(field_max[field], delta)
                exact &= delta == 0.0
            actual_action_delta = float(np.max(np.abs(
                control["action"][index, env] - ablation["action"][index, env]
            ), initial=0.0))
            expected_action_delta = float(row["max_abs_action_difference_at_first_divergence"])
            require(abs(actual_action_delta - expected_action_delta) <= 1e-6,
                    f"Action delta mismatch for {case_id}: {actual_action_delta} != {expected_action_delta}")
            require(int(control["skill_id"][index, env]) == 0 and
                    int(ablation["skill_id"][index, env]) == 0,
                    f"First divergence for {case_id} is not in Launch")
            action_deltas.append(actual_action_delta)
            indices.append(index)
            if exact:
                exact_case_ids.append(case_id)

    require(len(divergences) == 14, f"Expected 14 action-divergence cases, found {len(divergences)}")
    return {
        "cases_with_first_launch_action_divergence": len(divergences),
        "first_divergence_index_min_max": [min(indices), max(indices)],
        "pre_action_trajectory_index_rule": "first action divergence index minus one",
        "pre_action_state_fields": list(compared_fields),
        "pre_action_fields_bitwise_equal_case_count": len(exact_case_ids),
        "pre_action_fields_bitwise_equal_case_ids": sorted(exact_case_ids),
        "pre_action_max_abs_difference_by_field": field_max,
        "max_abs_action_difference_at_first_divergence": {
            "min": min(action_deltas),
            "median": statistics.median(action_deltas),
            "max": max(action_deltas),
        },
        "interpretation": (
            "For every changed case, the recorded physical/motor trajectory fields immediately "
            "before the first differing Launch action are bitwise identical. This isolates an "
            "immediate action response on a common recorded state, but does not establish a "
            "beneficial downstream effect or identify which feature component caused it."
        ),
    }


def build_report(root: Path):
    index_path = root / INDEX
    index = load_json(index_path)
    recover = index["aerowall_recover_tanh_mapping_ablation_v1_corrected"]
    phase0_audit = index["phase0_preaction_contact_alignment_v1"]
    phase2 = index["phase2_shared_case_actor_contact_comparison_v1"]
    fullbank = index["fullbank_contact_approach_alignment_v1"]
    retention = index["aerowall_launch_intercept_target_retention_ablation_v1"]

    src = {}
    recover_report = recover["artifacts"]["report"]
    src["recover_tanh_paired_run"] = verified_source(
        root, "Recover-Tanh run report", recover_report, recover["artifacts"]["report_sha256"]
    )
    src["phase0_preaction_alignment"] = verified_source(
        root, "Phase-0 pre-action report", phase0_audit["report"], phase0_audit["report_sha256"]
    )
    src["fullbank_approach_alignment"] = verified_source(
        root, "Full-bank alignment report", fullbank["report"], fullbank["report_sha256"]
    )
    src["phase2_control_report"] = verified_source(
        root, "Phase-2 control report", phase2["control"]["report"], phase2["control"]["report_sha256"]
    )
    src["phase2_tanh_report"] = verified_source(
        root, "Phase-2 Recover-Tanh report", phase2["tanh"]["report"], phase2["tanh"]["report_sha256"]
    )
    paired_path = retention["artifacts"]["analysis_report"]["path"]
    src["launch_target_retention_pair_report"] = verified_source(
        root, "Launch target-retention paired report", paired_path,
        retention["artifacts"]["analysis_report"]["sha256"],
    )
    raw_path = retention["artifacts"]["raw_ablation_report"]["path"]
    src["launch_target_retention_run_report"] = verified_source(
        root, "Launch target-retention run report", raw_path,
        retention["artifacts"]["raw_ablation_report"]["sha256"],
    )

    control_run_path = REPRO / "legacy-control-launch-relv3-corrected-s9524.json"
    control_trajectory_path = REPRO / "legacy-control-launch-relv3-corrected-s9524.trajectory.npz"
    ablation_trajectory_path = REPRO / "aerowall-launch-intercept-target-retention-v1-s9524.trajectory.npz"
    src["launch_control_run_report"] = verified_source(
        root, "Launch control run report", str(control_run_path),
        retention["artifacts"]["control_report_sha256"],
    )
    src["launch_control_trajectory"] = verified_source(
        root, "Launch control trajectory", str(control_trajectory_path),
        retention["artifacts"]["control_trajectory_sha256"],
    )
    src["launch_target_retention_trajectory"] = verified_source(
        root, "Launch target-retention trajectory", str(ablation_trajectory_path),
        retention["artifacts"]["trajectory"]["sha256"],
    )

    paired_report = load_json(root / paired_path)
    ablation_report = load_json(root / raw_path)
    state_comparison = compare_first_divergence_states(
        root, paired_report, ablation_report,
        root / control_trajectory_path, root / ablation_trajectory_path,
    )
    require(state_comparison["pre_action_fields_bitwise_equal_case_count"] == 14,
            "Not all first-divergence states are bitwise equal")

    recover_metrics = recover["metrics"]
    p0 = fullbank["phase0"]
    p2 = fullbank["phase2"]
    target_metrics = retention["metrics"]
    require(p0["paired_case_labels_identical"], "Phase-0 labels unexpectedly differ")
    require(target_metrics["control"]["legal_first_phase0_contacts"] ==
            target_metrics["ablation"]["legal_first_phase0_contacts"] == 116,
            "Unexpected Launch target-retention first-contact counts")

    return {
        "experiment_name": "AeroWallPhaseSpecificFailureModeSynthesisV1",
        "status": "completed_read_only_evidence_synthesis",
        "source_scope": "Same-seed, same 128-case v4 development bank plus a failure-enriched 33-case contact replay; not final acceptance evidence.",
        "training_performed": False,
        "promotion": False,
        "sources": src,
        "phase0_launch_first_contact": {
            "action_semantics": fullbank["action_interface"],
            "matched_illegal_minus_legal_limited_rotor_command_4ch_by_lead": p0["matched_approach"]["median_illegal_minus_legal_limited_rotor_command_4ch_by_lead"],
            "matched_illegal_minus_legal_target_rotor_throttle_4ch_by_lead": p0["matched_approach"]["median_illegal_minus_legal_target_rotor_throttle_4ch_by_lead"],
            "control_and_recover_tanh_first_contact_labels_identical": p0["paired_case_labels_identical"],
            "legal_first_contacts": 116,
            "illegal_first_contacts": 12,
            "illegal_cases_over_radial_limit_at_contact": 12,
            "illegal_cases_already_over_radial_limit_before_final_action": 9,
            "illegal_cases_crossing_limit_during_final_action_interval": 3,
            "actual_actor_for_all_pre_first_cap_illegal_contacts": "Launch",
            "matched_approach_pairs": p0["matched_approach"]["pair_count"],
            "matched_pairs_with_standardized_state_distance_le_1": p0["matched_approach"]["distance_threshold_counts"]["le_1"],
            "matched_pairs_with_standardized_state_distance_le_1p5": p0["matched_approach"]["distance_threshold_counts"]["le_1p5"],
            "matched_illegal_minus_legal_local_y_at_lead_20_m": p0["matched_approach"]["median_illegal_minus_legal_local_y_by_lead_m"]["20"],
            "matched_illegal_minus_legal_drone_vy_at_lead_20_m_s": p0["matched_approach"]["median_illegal_minus_legal_drone_vy_by_lead_m_s"]["20"],
            "scope_note": "The matched approach difference is observational, based on one development bank, and does not establish whether initial conditions or earlier Launch responses caused the miss.",
        },
        "phase1_wall_transition": {
            "tanh_recover_illegal_case_in_failure_enriched_replay": "heldout-0083",
            "actual_actor": "Recover",
            "scope_note": "One event in the selected 33-case failure-enriched replay; not a population rate.",
        },
        "phase2_return_contact": {
            "failure_enriched_replay_case_count": 33,
            "control_illegal_cases": p2["control"]["illegal_cases"],
            "recover_tanh_illegal_cases": p2["recover_tanh"]["illegal_cases"],
            "shared_illegal_cases": p2["case_comparison"]["shared_case_count"],
            "recover_tanh_only_cases": p2["case_comparison"]["tanh_only_case_ids"],
            "active_actor_counts_control": p2["control"]["active_actor_counts"],
            "active_actor_counts_recover_tanh": p2["recover_tanh"]["active_actor_counts"],
            "shared_case_actor_role_changes": phase2["paired_case_comparison"]["active_actor_role_changed_count"],
            "shared_case_radial_error_changes": phase2["paired_case_comparison"]["tanh_minus_control_radial_error_m"],
            "median_pre_action_radial_error_m_control": p2["control"]["median_preaction_radial_m"],
            "median_pre_action_radial_error_m_recover_tanh": p2["recover_tanh"]["median_preaction_radial_m"],
            "scope_note": "The selected replay is failure-enriched; all of its illegal contacts were outside the 0.20 m radial limit before the final action. Case effects are mixed and do not isolate one action mechanism.",
        },
        "recover_tanh_mapping_outcomes": {
            "action_limit_fraction_control_vs_tanh_recover": recover_metrics["actuator_limit_fraction_total"],
            "recover_action_limit_fraction_control": recover_metrics["per_skill_action_limit_fraction_control"]["recovery"]["fraction_action_values_limited"],
            "recover_action_limit_fraction_tanh": recover_metrics["per_skill_action_limit_fraction_tanh_recover"]["recovery"]["fraction_action_values_limited"],
            "legal_first_contacts_control_vs_tanh": recover_metrics["valid_first_contacts"],
            "legal_second_hits_control_vs_tanh": recover_metrics["legal_second_hits"],
            "three_rally_cases_control_vs_tanh": recover_metrics["three_rally_cases"],
            "five_rally_cases_control_vs_tanh": recover_metrics["five_rally_cases"],
            "mean_rallies_control_vs_tanh": recover_metrics["mean_rallies"],
            "safety_failures_control_vs_tanh": recover_metrics["safety_failures"],
            "failure_class_counts_control": recover_metrics["failure_counts_control"],
            "failure_class_counts_tanh": recover_metrics["failure_counts_tanh_recover"],
        },
        "launch_target_retention_outcomes": {
            "cases_with_changed_launch_actions": target_metrics["paired_action_trajectory"]["episodes_with_any_difference"],
            "shared_active_action_rows_changed": target_metrics["paired_action_trajectory"]["active_rows_with_any_difference"],
            "first_contact_labels_changed_cases": target_metrics["first_phase0_labels_changed_case_ids"],
            "legal_first_contacts_control_vs_ablation": [
                target_metrics["control"]["legal_first_phase0_contacts"],
                target_metrics["ablation"]["legal_first_phase0_contacts"],
            ],
            "legal_second_hits_control_vs_ablation": [
                target_metrics["control"]["legal_second_hit_rate"],
                target_metrics["ablation"]["legal_second_hit_rate"],
            ],
            "three_rally_rate_control_vs_ablation": [
                target_metrics["control"]["three_rally_rate"],
                target_metrics["ablation"]["three_rally_rate"],
            ],
            "five_rally_rate_control_vs_ablation": [
                target_metrics["control"]["five_rally_rate"],
                target_metrics["ablation"]["five_rally_rate"],
            ],
            "safety_failures_control_vs_ablation": [
                target_metrics["control"]["safety_failure_episodes"],
                target_metrics["ablation"]["safety_failure_episodes"],
            ],
            "heldout_0060_failure_path": retention["heldout_0060_failure_path"],
            "first_divergence_common_state_check": state_comparison,
        },
        "cross_phase_conclusion": (
            "The strongest repeated failure symptom is a phase-0 Launch radial miss; it is unchanged by Recover-only Tanh. "
            "The phase-2 return misses are actor-role heterogeneous and not fixed by removing Recover clipping. "
            "Retaining the predicted intercept changed Launch actions on common pre-divergence states but did not improve "
            "first-contact labels or rally outcomes and added one safety failure. The evidence does not justify a fixed "
            "fixed per-rotor action offset, a Recover-Tanh candidate, or promotion."
        ),
        "next_step": (
            "Keep P2 false and formal C350 unchanged. The separate Launch world-v_y state intervention is not evidence that the actor can realize that state change. "
            "Before any policy change, pre-register a no-training local controllability screen that perturbs direct rotor commands from the same saved Launch states, "
            "reports all four command-channel responses, and records ball world-y boundary terminations. Do not train, promote, or start P3-P6 from the selected-case evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path.cwd()
    report = build_report(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "sources_verified": len(report["sources"]),
        "phase0_illegal_contacts": report["phase0_launch_first_contact"]["illegal_first_contacts"],
        "phase2_control_vs_tanh": [report["phase2_return_contact"]["control_illegal_cases"], report["phase2_return_contact"]["recover_tanh_illegal_cases"]],
        "launch_view_changed_action_cases": report["launch_target_retention_outcomes"]["cases_with_changed_launch_actions"],
        "common_pre_action_states_equal": report["launch_target_retention_outcomes"]["first_divergence_common_state_check"]["pre_action_fields_bitwise_equal_case_count"],
        "training_performed": report["training_performed"],
        "promotion": report["promotion"],
    }, indent=2))


if __name__ == "__main__":
    main()
