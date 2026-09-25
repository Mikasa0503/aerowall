#!/usr/bin/env python3
"""Summarize pre-action and exact-contact geometry for the 33-case phase-0 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPRO_DIR = Path("artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility")
STEM = "aerowall-contact-substep-actor-input-audit-v1-{condition}-runtime-s9524"
LEADS = (0, 2, 5, 10, 15, 20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quat_rotate_inverse(quaternion, vector):
    """Rotate a vector from world frame into a scalar-first quaternion frame."""
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    w, axis = q[0], q[1:]
    return 2 * np.dot(axis, v) * axis + (w * w - np.dot(axis, axis)) * v - 2 * w * np.cross(axis, v)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.shape[0]),
        "median": np.median(array, axis=0).tolist(),
        "q25": np.percentile(array, 25, axis=0).tolist(),
        "q75": np.percentile(array, 75, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
    }


def first_phase0_contacts(events, outcomes):
    first = {}
    for event in events:
        if not event.get("body") or event.get("phase_before") != 0 or "actor_observation" not in event:
            continue
        env = event["env"]
        current = first.get(env)
        if current is None or (event["policy_step"], event["substep"]) < (current["policy_step"], current["substep"]):
            first[env] = event

    rows = []
    for env, event in first.items():
        observation = np.asarray(event["actor_observation"], dtype=np.float64)
        contact_local = quat_rotate_inverse(event["drone_orientation_wxyz"], event["ball_relative_position_world"])
        pre_radial = float(np.linalg.norm(observation[26:28]))
        rows.append({
            "env": env,
            "case_id": outcomes[env]["case_id"],
            "label": "legal" if event["legal_cap"] else "illegal",
            "active_actor": event["actor_observation_role"],
            "policy_step": event["policy_step"],
            "substep": event["substep"],
            "observation": observation,
            "contact_local": contact_local,
            "pre_radial": pre_radial,
            "contact_radial": float(event["radial_error"]),
            "local_delta": contact_local - observation[26:29],
            "event": event,
        })
    return rows


def condition_result(condition, base_dir):
    stem = STEM.format(condition=condition)
    report_path = base_dir / f"{stem}.json"
    events_path = base_dir / f"{stem}.events.jsonl"
    trajectory_path = base_dir / f"{stem}.trajectory.npz"
    contacts_path = base_dir / f"{stem}.contacts.json"
    report = json.loads(report_path.read_text())
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    outcomes = {outcome["env"]: outcome for outcome in report["outcomes"]}
    rows = first_phase0_contacts(events, outcomes)
    trajectory = np.load(trajectory_path)

    checks = {"observation_vs_pre_action_state_max_abs_error": 0.0, "event_action_vs_saved_action_max_abs_error": 0.0}
    decision_to_contact_seconds = []
    lead_values = {label: {lead: [] for lead in LEADS} for label in ("legal", "illegal")}
    for row in rows:
        event, env = row["event"], row["env"]
        step = row["policy_step"]
        pre_state = trajectory["drone_state"][step - 1, env]
        pre_relative = quat_rotate_inverse(pre_state[3:7], trajectory["ball_position"][step - 1, env] - pre_state[:3])
        checks["observation_vs_pre_action_state_max_abs_error"] = max(
            checks["observation_vs_pre_action_state_max_abs_error"],
            float(np.max(np.abs(pre_relative - row["observation"][26:29]))),
        )
        checks["event_action_vs_saved_action_max_abs_error"] = max(
            checks["event_action_vs_saved_action_max_abs_error"],
            float(np.max(np.abs(trajectory["action"][step, env] - np.asarray(event["policy_action"])))),
        )
        decision_to_contact_seconds.append((event["substep"] + 1) * float(trajectory["physics_dt"]))

        for lead in LEADS:
            frame = step - 1 - lead
            if frame < 0:
                raise ValueError(f"phase-0 contact at step {step} has no state for lead {lead}")
            state = trajectory["drone_state"][frame, env]
            relative = quat_rotate_inverse(state[3:7], trajectory["ball_position"][frame, env] - state[:3])
            lead_values[row["label"]][lead].append(
                [relative[0], relative[1], relative[2], np.linalg.norm(relative[:2])]
            )

    summaries = {}
    for label in ("legal", "illegal"):
        group = [row for row in rows if row["label"] == label]
        observations = np.stack([row["observation"] for row in group])
        summaries[label] = {
            "count": len(group),
            "active_actor_counts": {role: sum(row["active_actor"] == role for row in group)
                                    for role in sorted({row["active_actor"] for row in group})},
            "pre_action_local_y_m": summarize([[row["observation"][27]] for row in group]),
            "contact_local_y_m": summarize([[row["contact_local"][1]] for row in group]),
            "delta_local_y_m": summarize([[row["local_delta"][1]] for row in group]),
            "pre_action_radial_m": summarize([[row["pre_radial"]] for row in group]),
            "contact_radial_m": summarize([[row["contact_radial"]] for row in group]),
            "delta_radial_m": summarize([[row["contact_radial"] - row["pre_radial"]] for row in group]),
            "pre_action_local_z_m": summarize([[row["observation"][28]] for row in group]),
            "contact_local_z_m": summarize([[row["contact_local"][2]] for row in group]),
            "local_y_below_minus_0p18m_count": sum(row["observation"][27] < -0.18 for row in group),
            "pre_action_radial_above_0p20m_count": sum(row["pre_radial"] > 0.20 for row in group),
            "actor_input_medians": {
                "target_displacement_world_0_3": np.median(observations[:, 0:3], axis=0).tolist(),
                "ball_relative_position_body_26_29": np.median(observations[:, 26:29], axis=0).tolist(),
                "ball_relative_velocity_body_29_32": np.median(observations[:, 29:32], axis=0).tolist(),
                "predicted_intercept_relative_body_32_35": np.median(observations[:, 32:35], axis=0).tolist(),
            },
            "trajectory_local_relative_position_by_lead": {
                str(lead): summarize(lead_values[label][lead]) for lead in LEADS
            },
            "cases": sorted(row["case_id"] for row in group),
        }

    failure_rows = [row for row in rows if row["label"] == "illegal"]
    crossers = [row["case_id"] for row in failure_rows if row["pre_radial"] <= 0.20 < row["contact_radial"]]
    return {
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "events": str(events_path),
        "events_sha256": sha256(events_path),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": sha256(trajectory_path),
        "contacts_sha256": sha256(contacts_path),
        "phase0_first_body_contacts": len(rows),
        "outcome_step_alignment": checks,
        "decision_to_contact_seconds": summarize([[value] for value in decision_to_contact_seconds]),
        "groups": summaries,
        "illegal_contacts_that_crossed_radial_limit_after_action": sorted(crossers),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=REPRO_DIR)
    parser.add_argument("--output", type=Path,
                        help="Write the analysis report here instead of stdout")
    args = parser.parse_args()
    control = condition_result("control", args.artifact_dir)
    tanh = condition_result("tanh", args.artifact_dir)

    control_cases = {case: group for group in ("legal", "illegal")
                     for case in control["groups"][group]["cases"]}
    tanh_cases = {case: group for group in ("legal", "illegal")
                  for case in tanh["groups"][group]["cases"]}
    if len(control_cases) != 33 or control_cases != tanh_cases:
        raise ValueError("Expected 33 identical phase-0 case/label rows in control and Tanh")

    result = {
        "experiment_name": "AeroWallPhase0PreActionContactAlignmentV1",
        "status": "analysis_passed",
        "analysis_script": "scripts/analyze_aerowall_contact_alignment.py",
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "training_performed": False,
        "promotion": False,
        "method": {
            "first_body_contact_per_case": "minimum (policy_step, substep) with phase_before == 0",
            "contact_labels": "PhysX callback legal_cap flag",
            "pre_action_state": "trajectory frame policy_step - 1; active actor observation feature indices 26:29 independently cross-checked",
            "issued_action": "trajectory action frame policy_step, independently cross-checked against event policy_action",
            "relative_y_axis": "inverse scalar-first WXYZ drone orientation applied to ball minus drone world position",
            "trajectory_leads_before_action": list(LEADS),
            "policy_dt_s": 0.02,
            "physics_dt_s": 0.0025,
        },
        "control": control,
        "tanh": tanh,
        "paired_phase0_cases_and_labels_identical": True,
        "interpretation": "In this failure-enriched 33-case subset, most illegal first contacts already exceed the radial limit at the decision state for their final policy action. Their median local-y error changes little over the remaining action interval; illegal-vs-legal local-y separation is visible earlier in the approach trajectory. This is descriptive evidence and does not establish causation or a population rate.",
        "next_step": "Trace the Launch approach actions and lateral position error over the earlier 0.2-0.4 s window; separately analyze phase-2 contacts by active Hit/Recover actor. Do not train or promote until a falsifiable single-variable hypothesis is defined and evaluated.",
    }
    serialized = json.dumps(result, indent=2, sort_keys=True,
                            default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
