#!/usr/bin/env python3
"""Compare the named AeroWall intercept-target retention observation ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPRO_DIR = Path("artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility")
CONTROL_STEM = "legacy-control-launch-relv3-corrected-s9524"
ABLATION_STEM = "aerowall-launch-intercept-target-retention-v1-s9524"
OUTPUT = REPRO_DIR / "aerowall-launch-intercept-target-retention-ablation-v1-s9524.json"
PROTECTED_FIELDS = (
    "seed", "stage", "num_envs", "initial_case_bank_sha256", "hcsp_commit",
    "candidate_name", "candidate_training_report_sha256", "candidate_actor_checkpoint_sha256",
    "hit_candidate_name", "hit_training_report_sha256", "hit_checkpoint_sha256",
    "recovery_checkpoint_sha256", "physics_dt", "policy_dt", "reward_design",
    "hit_window", "hit_exit_window", "min_dwell_steps", "observation_version",
    "skill_observation_version",
    "launch_action_distribution", "hit_action_distribution", "recovery_action_distribution",
)
OUTCOME_FIELDS = ("caps", "walls", "rallies", "policy_steps")
SAFETY_FAILURE_CODES = {2, 4, 5}  # evaluator: drone ground, illegal contact, drone wall
FAILURE_CODE_NAMES = {
    0: "timeout_or_unfinished",
    1: "ball_ground",
    2: "drone_ground",
    3: "out_of_bounds",
    4: "illegal_contact",
    5: "drone_wall_threshold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_versions_by_role(report):
    versions = report.get("evaluation_observation_versions_by_role")
    if versions is None:
        versions = report.get("skill_observation_versions_by_role")
    if versions is None:
        versions = report.get("initial_policy_input_trace", {}).get(
            "observation_versions_by_role", {}
        )
    return versions or {}


def read_run(artifact_dir: Path, stem: str):
    paths = {
        "report": artifact_dir / f"{stem}.json",
        "events": artifact_dir / f"{stem}.events.jsonl",
        "trajectory": artifact_dir / f"{stem}.trajectory.npz",
        "contacts": artifact_dir / f"{stem}.contacts.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation artifacts for {stem}: {missing}")
    report = json.loads(paths["report"].read_text())
    events = [json.loads(line) for line in paths["events"].read_text().splitlines() if line.strip()]
    outcomes = {row["case_id"]: row for row in report["outcomes"]}
    env_to_case = {int(row["env"]): row["case_id"] for row in report["outcomes"]}
    first_phase0 = {}
    for event in events:
        if not event.get("body") or event.get("phase_before") != 0:
            continue
        env = int(event["env"])
        key = (int(event["policy_step"]), int(event["substep"]))
        if env not in first_phase0 or key < first_phase0[env][0]:
            first_phase0[env] = (key, event)
    phase0_labels = {
        env_to_case[env]: bool(event["legal_cap"])
        for env, (_, event) in first_phase0.items()
    }
    return paths, report, outcomes, phase0_labels


def summarize_condition(report, outcomes, labels):
    failure_cases = sorted(
        case for case, row in outcomes.items()
        if int(row["failure"]) in SAFETY_FAILURE_CODES
    )
    failure_counts = {
        str(code): sum(int(row["failure"]) == code for row in outcomes.values())
        for code in range(6)
    }
    legal = sum(labels.values())
    illegal = sum(not label for label in labels.values())
    summary = report["summary"]
    return {
        "episode_count": len(outcomes),
        "first_phase0_body_contacts": len(labels),
        "legal_first_phase0_contacts": legal,
        "illegal_first_phase0_contacts": illegal,
        "safety_failure_episodes": len(failure_cases),
        "safety_failure_case_ids": failure_cases,
        "failure_counts_by_evaluator_code": failure_counts,
        "mean_caps": summary.get("mean_caps"),
        "mean_walls": summary.get("mean_walls"),
        "mean_rallies": summary.get("mean_rallies"),
        "legal_second_hit_rate": summary.get("legal_second_hit_rate"),
        "one_rally_rate": summary.get("one_rally_rate"),
        "three_rally_rate": summary.get("three_rally_rate"),
        "five_rally_rate": summary.get("five_rally_rate"),
        "all_outcomes_contact_audited": summary.get("all_outcomes_contact_audited"),
        "actuator_trace_summary": report.get("actuator_trace_summary"),
    }


def compare_outcomes(control, ablation):
    case_ids = sorted(set(control) | set(ablation))
    rows = []
    for case_id in case_ids:
        left, right = control.get(case_id), ablation.get(case_id)
        row = {"case_id": case_id, "present_in_both": left is not None and right is not None}
        if left is not None and right is not None:
            row["delta"] = {field: right[field] - left[field] for field in OUTCOME_FIELDS}
        rows.append(row)
    delta_summary = {}
    for field in OUTCOME_FIELDS:
        deltas = [row["delta"][field] for row in rows if "delta" in row]
        delta_summary[field] = {
            "positive_delta_count": sum(value > 0 for value in deltas),
            "negative_delta_count": sum(value < 0 for value in deltas),
            "unchanged_count": sum(value == 0 for value in deltas),
            "minimum_delta": min(deltas) if deltas else None,
            "maximum_delta": max(deltas) if deltas else None,
        }
    return {"case_count_union": len(case_ids), "by_field": delta_summary, "cases": rows}


def compare_initial_launch_views(control_report, ablation_report):
    control_trace = control_report["initial_policy_input_trace"]
    ablation_trace = ablation_report["initial_policy_input_trace"]
    control_view = np.asarray(
        control_trace["actor_observation_views_by_role"]["launch"], dtype=np.float64,
    )
    ablation_view = np.asarray(
        ablation_trace["actor_observation_views_by_role"]["launch"], dtype=np.float64,
    )
    if control_view.shape != ablation_view.shape or control_view.ndim != 2 or control_view.shape[1] != 46:
        raise ValueError(f"Unexpected initial Launch actor-view shapes: {control_view.shape}, {ablation_view.shape}")
    delta = np.abs(ablation_view - control_view)
    prefix_changed = np.max(delta[:, :3], axis=1) > 1e-7
    other_changed = np.max(delta[:, 3:], axis=1) > 1e-7
    if bool(other_changed.any()):
        raise ValueError("The initial actor views differ outside the intended first three features")
    return {
        "episode_rows": int(control_view.shape[0]),
        "episodes_with_changed_goto_target_delta": int(prefix_changed.sum()),
        "changed_case_ids": [
            f"heldout-{index:04d}" for index in np.flatnonzero(prefix_changed)
        ],
        "max_abs_delta_in_features_0_to_2": float(delta[:, :3].max(initial=0.0)),
        "max_abs_delta_in_features_3_to_45": float(delta[:, 3:].max(initial=0.0)),
    }


def compare_trajectories(control_path: Path, ablation_path: Path, case_ids):
    fields = ("action", "actuator_command", "motor_throttle", "drone_state",
              "ball_position", "ball_velocity")
    summary = {"active_rows_compared": 0, "fields": {}}
    first_action_divergence = []
    with np.load(control_path) as control, np.load(ablation_path) as ablation:
        if control["active"].shape != ablation["active"].shape:
            raise ValueError("Control and ablation trajectory shapes differ")
        common_active = control["active"].astype(bool) & ablation["active"].astype(bool)
        summary["active_rows_compared"] = int(common_active.sum())
        for field in fields:
            left = control[field]
            right = ablation[field]
            if left.shape != right.shape:
                raise ValueError(f"Control and ablation {field} shapes differ")
            per_row = np.max(np.abs(left - right), axis=-1)
            comparable_delta = np.where(common_active, per_row, 0.0)
            per_episode = np.max(comparable_delta, axis=0)
            summary["fields"][field] = {
                "episodes_with_any_difference": int((per_episode > 1e-7).sum()),
                "active_rows_with_any_difference": int((comparable_delta > 1e-7).sum()),
                "max_abs_difference": float(per_episode.max(initial=0.0)),
            }
            if field == "action":
                for env in np.flatnonzero(per_episode > 1e-7):
                    steps = np.flatnonzero(comparable_delta[:, env] > 1e-7)
                    step = int(steps[0])
                    first_action_divergence.append({
                        "case_id": case_ids[int(env)],
                        "trajectory_index": step,
                        "control_skill_id": int(control["skill_id"][step, env]),
                        "ablation_skill_id": int(ablation["skill_id"][step, env]),
                        "max_abs_action_difference_at_first_divergence": float(
                            per_row[step, env]
                        ),
                    })
    summary["first_action_divergence_by_case"] = sorted(
        first_action_divergence, key=lambda row: row["case_id"],
    )
    return summary


def heldout_0060_failure_path(control_run, ablation_run, trajectory_comparison):
    case_id = "heldout-0060"
    transition = next(
        (row for row in trajectory_comparison["first_action_divergence_by_case"]
         if row["case_id"] == case_id),
        None,
    )
    if transition is None:
        return {"case_id": case_id, "action_divergence_observed": False}

    details = {}
    for label, run in (("control", control_run), ("ablation", ablation_run)):
        _, report, outcomes, _ = run
        outcome = outcomes[case_id]
        env = int(outcome["env"])
        events = [
            json.loads(line) for line in run[0]["events"].read_text().splitlines()
            if line.strip() and int(json.loads(line)["env"]) == env
        ]
        first_cap = next(
            row for row in events
            if row.get("body") and row.get("phase_before") == 0
        )
        first_wall = next(row for row in events if row.get("wall"))
        with np.load(run[0]["trajectory"]) as trajectory:
            active = trajectory["active"][:, env].astype(bool)
            drone_x = trajectory["drone_state"][:, env, 0]
            active_indices = np.flatnonzero(active)
            min_x_index = int(active_indices[np.argmin(drone_x[active_indices])])
            crossing_indices = np.flatnonzero(active & (drone_x < 0.5))
        contacts = json.loads(run[0]["contacts"].read_text())
        physx_drone_wall_contacts = sum(
            row.get("contact_count", 0) > 0
            and any(f"/World/envs/env_{env}/IrisTest_0/" in actor
                    for actor in row.get("actors", []))
            and any(actor.endswith("/single_wall") for actor in row.get("actors", []))
            for row in contacts
        )
        details[label] = {
            "failure_code": int(outcome["failure"]),
            "policy_steps": int(outcome["policy_steps"]),
            "phase0_contact": {
                "policy_step": int(first_cap["policy_step"]),
                "substep": int(first_cap["substep"]),
                "legal": bool(first_cap["legal_cap"]),
                "radial_error_m": float(first_cap["radial_error"]),
            },
            "first_wall_event": {
                "policy_step": int(first_wall["policy_step"]),
                "substep": int(first_wall["substep"]),
                "ball_position_world_xyz_m": first_wall["ball_position"],
            },
            "minimum_active_drone_world_x_m": float(drone_x[min_x_index]),
            "minimum_x_trajectory_index": min_x_index,
            "first_world_x_below_0p5_trajectory_index": (
                int(crossing_indices[0]) if len(crossing_indices) else None
            ),
            "physx_drone_single_wall_contact_count": int(physx_drone_wall_contacts),
            "reported_run_status": report.get("status"),
        }
    return {
        "case_id": case_id,
        "first_action_divergence": transition,
        "drone_wall_threshold_semantics": "The environment terminates when drone world-x < 0.5 m; this is a task threshold, not evidence of physical contact.",
        "runs": details,
        "interpretation": "Descriptive paired trajectory: both runs made legal phase-0 contact; the ablation later crossed the environment x threshold while control terminated by ball ground. Neither PhysX contact sidecar records a drone/single_wall contact.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=REPRO_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    control_paths, control_report, control_outcomes, control_labels = read_run(args.artifact_dir, CONTROL_STEM)
    ablation_paths, ablation_report, ablation_outcomes, ablation_labels = read_run(args.artifact_dir, ABLATION_STEM)

    mismatches = {
        field: {"control": control_report.get(field), "ablation": ablation_report.get(field)}
        for field in PROTECTED_FIELDS
        if control_report.get(field) != ablation_report.get(field)
    }
    if mismatches:
        raise ValueError(f"The ablation changed a protected evaluation input: {mismatches}")
    control_versions = observation_versions_by_role(control_report)
    ablation_versions = observation_versions_by_role(ablation_report)
    role_version_mismatches = {
        role: {"control": control_versions.get(role), "ablation": ablation_versions.get(role)}
        for role in ("hit", "recovery")
        if control_versions.get(role) != ablation_versions.get(role)
    }
    if role_version_mismatches:
        raise ValueError(f"The ablation changed another role's observation view: {role_version_mismatches}")
    if control_versions.get("launch") != "relative_v3":
        raise ValueError("The control Launch observation is not relative_v3")
    if ablation_versions.get("launch") != "aerowall_intercept_target_retention_v1":
        raise ValueError("The ablation report does not record the named AeroWall Launch observation")
    if control_report.get("train_skill") is not None or ablation_report.get("train_skill") is not None:
        raise ValueError("Expected inference-only evaluations with no trainable skill")
    if len(control_outcomes) != 128 or len(ablation_outcomes) != 128:
        raise ValueError("Expected paired 128-case development-bank evaluations")
    if set(control_outcomes) != set(ablation_outcomes):
        raise ValueError("Control and ablation outcome case sets differ")

    changed_labels = sorted(
        case_id for case_id in set(control_labels) & set(ablation_labels)
        if control_labels[case_id] != ablation_labels[case_id]
    )
    labels_missing_control = sorted(set(ablation_labels) - set(control_labels))
    labels_missing_ablation = sorted(set(control_labels) - set(ablation_labels))
    control_failure_cases = {
        case_id for case_id, row in control_outcomes.items()
        if int(row["failure"]) in SAFETY_FAILURE_CODES
    }
    ablation_failure_cases = {
        case_id for case_id, row in ablation_outcomes.items()
        if int(row["failure"]) in SAFETY_FAILURE_CODES
    }
    trajectory_comparison = compare_trajectories(
        control_paths["trajectory"], ablation_paths["trajectory"],
        [case for case, _ in sorted(
            ((case, row["env"]) for case, row in control_outcomes.items()),
            key=lambda pair: pair[1],
        )],
    )
    result = {
        "experiment_name": "AeroWallLaunchInterceptTargetRetentionV1",
        "status": "analysis_passed",
        "training_performed": False,
        "promotion": False,
        "scope": {
            "bank": "paired 128-case v4 development bank; not the frozen final acceptance bank",
            "intervention": "Launch actor receives the same 46-D relative_v3 view except channels 0:3 retain the bounded predicted racket-height intercept target in phase 0 when crossing time and intercept bounds are valid, regardless of conservative drone reachability",
            "unchanged": "checkpoint weights, action distributions, reward, FSM, public critic observation, Hit/Recover actor views, and all other observation features",
        },
        "input_match": {
            "protected_fields": list(PROTECTED_FIELDS),
            "mismatches": mismatches,
            "non_launch_observation_version_mismatches": role_version_mismatches,
            "control_initial_case_bank_sha256": control_report.get("initial_case_bank_sha256"),
            "ablation_initial_case_bank_sha256": ablation_report.get("initial_case_bank_sha256"),
            "control_evaluation_observation_versions_by_role": control_versions,
            "ablation_evaluation_observation_versions_by_role": ablation_versions,
            "ablation_allow_observation_version_ablation": ablation_report.get("allow_observation_version_ablation"),
        },
        "control": summarize_condition(control_report, control_outcomes, control_labels),
        "ablation": summarize_condition(ablation_report, ablation_outcomes, ablation_labels),
        "first_phase0_contact_label_comparison": {
            "control_case_count": len(control_labels),
            "ablation_case_count": len(ablation_labels),
            "changed_case_ids": changed_labels,
            "missing_from_control_case_ids": labels_missing_control,
            "missing_from_ablation_case_ids": labels_missing_ablation,
        },
        "safety_failure_case_comparison": {
            "failure_codes_counted_as_safety": sorted(SAFETY_FAILURE_CODES),
            "failure_code_names": {str(code): name for code, name in FAILURE_CODE_NAMES.items()},
            "drone_wall_code_semantics": "The environment terminates when drone world-x < 0.5 m; this is a task threshold and does not assert a PhysX drone-wall contact.",
            "control_count": len(control_failure_cases),
            "ablation_count": len(ablation_failure_cases),
            "new_failure_case_ids": sorted(ablation_failure_cases - control_failure_cases),
            "recovered_failure_case_ids": sorted(control_failure_cases - ablation_failure_cases),
            "shared_failure_case_ids": sorted(control_failure_cases & ablation_failure_cases),
            "new_failure_code_transitions": [
                {
                    "case_id": case_id,
                    "control_failure_code": int(control_outcomes[case_id]["failure"]),
                    "control_failure_name": FAILURE_CODE_NAMES.get(int(control_outcomes[case_id]["failure"]), "unknown"),
                    "ablation_failure_code": int(ablation_outcomes[case_id]["failure"]),
                    "ablation_failure_name": FAILURE_CODE_NAMES.get(int(ablation_outcomes[case_id]["failure"]), "unknown"),
                }
                for case_id in sorted(ablation_failure_cases - control_failure_cases)
            ],
        },
        "initial_launch_actor_view_comparison": compare_initial_launch_views(
            control_report, ablation_report,
        ),
        "paired_trajectory_comparison": trajectory_comparison,
        "heldout_0060_failure_path": heldout_0060_failure_path(
            (control_paths, control_report, control_outcomes, control_labels),
            (ablation_paths, ablation_report, ablation_outcomes, ablation_labels),
            trajectory_comparison,
        ),
        "outcome_comparison": compare_outcomes(control_outcomes, ablation_outcomes),
        "source_artifact_sha256": {
            "control": {key: sha256(path) for key, path in control_paths.items()},
            "ablation": {key: sha256(path) for key, path in ablation_paths.items()},
            "analysis_script": sha256(Path(__file__).resolve()),
        },
        "next_step": "Retain this as an inference-only diagnostic. Do not train or promote from a single 128-case development-bank comparison; any future continuation requires review of safety and paired rally outcomes while keeping the formal C350 route unchanged.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "control": {key: value for key, value in result["control"].items() if key != "actuator_trace_summary"},
        "ablation": {key: value for key, value in result["ablation"].items() if key != "actuator_trace_summary"},
        "changed_first_contact_labels": changed_labels,
        "safety_failure_case_comparison": result["safety_failure_case_comparison"],
        "initial_launch_actor_view_comparison": result["initial_launch_actor_view_comparison"],
        "paired_trajectory_comparison": {
            "active_rows_compared": result["paired_trajectory_comparison"]["active_rows_compared"],
            "fields": result["paired_trajectory_comparison"]["fields"],
            "first_action_divergence_by_case": result["paired_trajectory_comparison"]["first_action_divergence_by_case"],
        },
        "heldout_0060_failure_path": result["heldout_0060_failure_path"],
        "missing_contact_labels": {
            "control": labels_missing_control,
            "ablation": labels_missing_ablation,
        },
        "outcome_deltas": result["outcome_comparison"]["by_field"],
    }, indent=2))


if __name__ == "__main__":
    main()
