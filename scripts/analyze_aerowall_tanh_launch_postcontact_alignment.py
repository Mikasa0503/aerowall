#!/usr/bin/env python3
"""Descriptively align paired contact and rollout evidence for Tanh Launch follow-up cases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility"
PREFIX = "aerowall-tanh-launch-independent-replication-v1"
PHASE_PATH = BASE / f"{PREFIX}-phase-analysis-s260925.json"
ARMS = ("control", "tanh")
OFFSETS = (0, 1, 2, 5, 10)


def read_json(path: Path):
    return json.loads(path.read_text())


def l2_delta(a, b):
    if a is None or b is None:
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return None
    return float(np.linalg.norm(aa - bb))


def finite_stats(values):
    vals = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def load_arm(arm):
    stem = f"{PREFIX}-{arm}-s260925"
    report = read_json(BASE / f"{stem}.json")
    events_path = BASE / f"{stem}.events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    trajectory = np.load(BASE / f"{stem}.trajectory.npz")
    env_by_case = {item["case_id"]: int(item["env"]) for item in report["outcomes"]}
    outcomes = {item["case_id"]: item for item in report["outcomes"]}
    events_by_env = {}
    for event in events:
        events_by_env.setdefault(int(event["env"]), []).append(event)
    return {
        "report": report,
        "trajectory": trajectory,
        "env_by_case": env_by_case,
        "outcomes": outcomes,
        "events_by_env": events_by_env,
    }


def first_event(events, predicate, *, after=None, strict_step=False):
    selected = []
    for event in events:
        key = (int(event["policy_step"]), int(event.get("substep", 0)))
        if after is not None:
            if strict_step and key[0] <= after[0]:
                continue
            if not strict_step and key <= after:
                continue
        if predicate(event):
            selected.append(event)
    return min(selected, key=lambda e: (int(e["policy_step"]), int(e.get("substep", 0)))) if selected else None


def contact_fields(event):
    if event is None:
        return None
    observation = event.get("actor_observation")
    return {
        "policy_step": int(event["policy_step"]),
        "substep": int(event.get("substep", 0)),
        "phase_before": int(event.get("phase_before", -1)),
        "body": bool(event.get("body", False)),
        "legal_cap": bool(event.get("legal_cap", False)),
        "expected_cap": bool(event.get("expected_cap", False)),
        "actor_role": event.get("actor_observation_role"),
        "actor_observation_version": event.get("actor_observation_version"),
        "actor_observation_dimension": len(observation) if observation is not None else None,
        "radial_error_m": float(event["radial_error"]) if event.get("radial_error") is not None else None,
        "axial_distance_m": float(event["axial_distance"]) if event.get("axial_distance") is not None else None,
        "drone_position_m": event.get("drone_position"),
        "drone_linear_velocity_mps": event.get("drone_linear_velocity"),
        "drone_angular_velocity_rps": event.get("drone_angular_velocity"),
        "ball_position_m": event.get("ball_position"),
        "ball_velocity_before_mps": event.get("ball_velocity_before"),
        "ball_velocity_after_mps": event.get("ball_velocity_after"),
        "actor_observation": observation,
        "policy_action": event.get("policy_action"),
        "event_source": "exact contact callback; policy-step/substep indexed",
    }


def post_contact_trajectory(arm_data, case_id, cap_event):
    trajectory = arm_data["trajectory"]
    env = arm_data["env_by_case"][case_id]
    cap_step = int(cap_event["policy_step"])
    states = {}
    for offset in OFFSETS:
        index = cap_step + offset
        if index < 0 or index >= trajectory["drone_state"].shape[0]:
            continue
        if not bool(trajectory["active"][index, env]):
            continue
        drone = trajectory["drone_state"][index, env]
        states[str(offset)] = {
            "state_index": index,
            "drone_position_m": drone[:3].astype(float).tolist(),
            "drone_linear_velocity_mps": drone[7:10].astype(float).tolist(),
            "drone_angular_velocity_rps": drone[10:13].astype(float).tolist(),
            "ball_position_m": trajectory["ball_position"][index, env].astype(float).tolist(),
            "ball_linear_velocity_mps": trajectory["ball_velocity"][index, env, :3].astype(float).tolist(),
        }
    action_index = cap_step + 1
    post_action = None
    if action_index < trajectory["action"].shape[0] and bool(trajectory["active"][action_index, env]):
        skill_id = int(round(float(trajectory["skill_id"][action_index, env])))
        # Mirrors _selected_actor_role after caps > 0 in the evaluator:
        # skill_id 1 selects Hit; the other post-contact phase selects Recover.
        post_action = {
            "policy_step": action_index,
            "skill_id": skill_id,
            "actor_role_inferred_from_evaluator_routing": "hit" if skill_id == 1 else "recovery",
            "policy_action": trajectory["action"][action_index, env].astype(float).tolist(),
            "active": True,
            "observation_available_in_saved_trajectory": False,
        }
    return {"states_by_policy_step_offset_from_first_cap": states, "first_post_cap_action": post_action}


def main():
    phase = read_json(PHASE_PATH)
    case_ids = phase["safety_failure_case_changes"]["tanh_only_cases_with_first_contact_unchanged"]
    if len(case_ids) != 13:
        raise RuntimeError(f"Expected the previously identified 13 cases, got {len(case_ids)}")

    arms = {arm: load_arm(arm) for arm in ARMS}
    per_case = {}
    first_cap_deltas = []
    post_action_deltas = []
    downstream_observation_deltas = []
    downstream_role_pairs = Counter()
    downstream_phase_pairs = Counter()
    downstream_events_present = Counter()
    downstream_single_arm_roles = {arm: Counter() for arm in ARMS}
    downstream_single_arm_phases = {arm: Counter() for arm in ARMS}
    downstream_single_arm_phase_roles = {arm: Counter() for arm in ARMS}
    downstream_radial_errors = {arm: [] for arm in ARMS}
    downstream_failure_bits = {arm: [] for arm in ARMS}
    horizon_metrics = {str(offset): {key: [] for key in (
        "drone_position_delta_m", "drone_linear_velocity_delta_mps",
        "ball_position_delta_m", "ball_linear_velocity_delta_mps",
    )} for offset in OFFSETS}

    for case_id in case_ids:
        case = {}
        cap_events = {}
        downstream_events = {}
        for arm in ARMS:
            data = arms[arm]
            env = data["env_by_case"][case_id]
            events = data["events_by_env"].get(env, [])
            cap = first_event(events, lambda e: bool(e.get("legal_cap")) and bool(e.get("body")))
            if cap is None:
                raise RuntimeError(f"{arm} is missing a legal first cap for {case_id}")
            after_key = (int(cap["policy_step"]), int(cap.get("substep", 0)))
            downstream = first_event(
                events,
                lambda e: bool(e.get("body")) and not bool(e.get("legal_cap")),
                after=after_key,
                strict_step=True,
            )
            cap_events[arm] = cap
            downstream_events[arm] = downstream
            case[arm] = {
                "outcome": {
                    "rallies": int(data["outcomes"][case_id]["rallies"]),
                    "caps": int(data["outcomes"][case_id]["caps"]),
                    "failure": int(data["outcomes"][case_id]["failure"]),
                    "failure_reason_bits": int(data["outcomes"][case_id]["failure_reason_bits"]),
                },
                "first_legal_cap": contact_fields(cap),
                "first_later_nonlegal_body_contact": contact_fields(downstream),
                "post_contact_trajectory": post_contact_trajectory(data, case_id, cap),
            }
            downstream_events_present[arm] += int(downstream is not None)
            if downstream is not None:
                role = downstream.get("actor_observation_role", "unknown")
                phase_id = str(downstream.get("phase_before", -1))
                downstream_single_arm_roles[arm][role] += 1
                downstream_single_arm_phases[arm][phase_id] += 1
                downstream_single_arm_phase_roles[arm][f"{phase_id}|{role}"] += 1
                if downstream.get("radial_error") is not None:
                    downstream_radial_errors[arm].append(float(downstream["radial_error"]))
                downstream_failure_bits[arm].append(int(data["outcomes"][case_id]["failure_reason_bits"]))

        cc = case["control"]["first_legal_cap"]
        tc = case["tanh"]["first_legal_cap"]
        first_cap_deltas.append({
            "policy_step_tanh_minus_control": tc["policy_step"] - cc["policy_step"],
            "substep_tanh_minus_control": tc["substep"] - cc["substep"],
            "launch_actor_observation_l2": l2_delta(tc["actor_observation"], cc["actor_observation"]),
            "launch_policy_action_l2": l2_delta(tc["policy_action"], cc["policy_action"]),
            "contact_ball_velocity_before_l2_mps": l2_delta(tc["ball_velocity_before_mps"], cc["ball_velocity_before_mps"]),
            "contact_ball_velocity_after_l2_mps": l2_delta(tc["ball_velocity_after_mps"], cc["ball_velocity_after_mps"]),
            "radial_error_tanh_minus_control_m": tc["radial_error_m"] - cc["radial_error_m"],
            "axial_distance_tanh_minus_control_m": tc["axial_distance_m"] - cc["axial_distance_m"],
        })

        cd = case["control"]["first_later_nonlegal_body_contact"]
        td = case["tanh"]["first_later_nonlegal_body_contact"]
        if cd is not None and td is not None:
            downstream_role_pairs[f'{cd["actor_role"]}|{td["actor_role"]}'] += 1
            downstream_phase_pairs[f'{cd["phase_before"]}|{td["phase_before"]}'] += 1
            downstream_observation_deltas.append({
                "same_role": cd["actor_role"] == td["actor_role"],
                "same_version": cd["actor_observation_version"] == td["actor_observation_version"],
                "observation_l2": l2_delta(cd["actor_observation"], td["actor_observation"]),
                "policy_action_l2": l2_delta(cd["policy_action"], td["policy_action"]),
                "radial_error_tanh_minus_control_m": (
                    td["radial_error_m"] - cd["radial_error_m"]
                    if td["radial_error_m"] is not None and cd["radial_error_m"] is not None else None
                ),
            })

        ca = case["control"]["post_contact_trajectory"]["first_post_cap_action"]
        ta = case["tanh"]["post_contact_trajectory"]["first_post_cap_action"]
        if ca and ta:
            post_action_deltas.append({
                "control_actor": ca["actor_role_inferred_from_evaluator_routing"],
                "tanh_actor": ta["actor_role_inferred_from_evaluator_routing"],
                "policy_action_l2": l2_delta(ca["policy_action"], ta["policy_action"]),
            })

        for offset in OFFSETS:
            cstate = case["control"]["post_contact_trajectory"]["states_by_policy_step_offset_from_first_cap"].get(str(offset))
            tstate = case["tanh"]["post_contact_trajectory"]["states_by_policy_step_offset_from_first_cap"].get(str(offset))
            if not cstate or not tstate:
                continue
            horizon_metrics[str(offset)]["drone_position_delta_m"].append(l2_delta(tstate["drone_position_m"], cstate["drone_position_m"]))
            horizon_metrics[str(offset)]["drone_linear_velocity_delta_mps"].append(l2_delta(tstate["drone_linear_velocity_mps"], cstate["drone_linear_velocity_mps"]))
            horizon_metrics[str(offset)]["ball_position_delta_m"].append(l2_delta(tstate["ball_position_m"], cstate["ball_position_m"]))
            horizon_metrics[str(offset)]["ball_linear_velocity_delta_mps"].append(l2_delta(tstate["ball_linear_velocity_mps"], cstate["ball_linear_velocity_mps"]))

        per_case[case_id] = case

    summary = {
        "case_count": len(case_ids),
        "first_cap_policy_step_tanh_minus_control": finite_stats([x["policy_step_tanh_minus_control"] for x in first_cap_deltas]),
        "first_cap_substep_tanh_minus_control": finite_stats([x["substep_tanh_minus_control"] for x in first_cap_deltas]),
        "launch_actor_observation_l2_at_first_cap": finite_stats([x["launch_actor_observation_l2"] for x in first_cap_deltas]),
        "launch_policy_action_l2_at_first_cap": finite_stats([x["launch_policy_action_l2"] for x in first_cap_deltas]),
        "contact_ball_velocity_before_l2_mps": finite_stats([x["contact_ball_velocity_before_l2_mps"] for x in first_cap_deltas]),
        "contact_ball_velocity_after_l2_mps": finite_stats([x["contact_ball_velocity_after_l2_mps"] for x in first_cap_deltas]),
        "radial_error_tanh_minus_control_m": finite_stats([x["radial_error_tanh_minus_control_m"] for x in first_cap_deltas]),
        "axial_distance_tanh_minus_control_m": finite_stats([x["axial_distance_tanh_minus_control_m"] for x in first_cap_deltas]),
        "first_post_cap_action_l2": finite_stats([x["policy_action_l2"] for x in post_action_deltas]),
        "first_post_cap_action_actor_pairs": dict(Counter(f'{x["control_actor"]}|{x["tanh_actor"]}' for x in post_action_deltas)),
        "first_later_nonlegal_body_contact_counts": dict(downstream_events_present),
        "paired_first_later_nonlegal_contact_actor_role_pairs": dict(downstream_role_pairs),
        "paired_first_later_nonlegal_contact_phase_pairs": dict(downstream_phase_pairs),
        "first_later_nonlegal_body_contact_actor_roles_by_arm": {arm: dict(counts) for arm, counts in downstream_single_arm_roles.items()},
        "first_later_nonlegal_body_contact_phases_by_arm": {arm: dict(counts) for arm, counts in downstream_single_arm_phases.items()},
        "first_later_nonlegal_body_contact_phase_role_by_arm": {arm: dict(counts) for arm, counts in downstream_single_arm_phase_roles.items()},
        "first_later_nonlegal_body_contact_radial_error_m_by_arm": {arm: finite_stats(values) for arm, values in downstream_radial_errors.items()},
        "first_later_nonlegal_body_contact_radial_error_exceeds_0p20m_by_arm": {
            arm: sum(value > 0.20 for value in values) for arm, values in downstream_radial_errors.items()
        },
        "outcome_failure_reason_bits_at_first_later_nonlegal_contact_by_arm": {
            arm: dict(Counter(values)) for arm, values in downstream_failure_bits.items()
        },
        "paired_first_later_nonlegal_contact_actor_observation_l2": finite_stats([x["observation_l2"] for x in downstream_observation_deltas]),
        "paired_first_later_nonlegal_contact_policy_action_l2": finite_stats([x["policy_action_l2"] for x in downstream_observation_deltas]),
        "paired_first_later_nonlegal_contact_radial_error_tanh_minus_control_m": finite_stats([x["radial_error_tanh_minus_control_m"] for x in downstream_observation_deltas]),
        "relative_policy_step_horizon_state_delta": {
            offset: {metric: finite_stats(values) for metric, values in metrics.items()}
            for offset, metrics in horizon_metrics.items()
        },
    }
    result = {
        "schema_version": 1,
        "experiment_name": "AeroWallLaunchTanhPostContactAlignmentV1",
        "status": "descriptive_analysis_complete",
        "seed": 260925,
        "case_bank_sha256": arms["control"]["report"]["initial_case_bank_sha256"],
        "source_report_sha256": {
            arm: __import__("hashlib").sha256((BASE / f"{PREFIX}-{arm}-s260925.json").read_bytes()).hexdigest()
            for arm in ARMS
        },
        "selected_cases": case_ids,
        "scope": "post-hoc descriptive alignment of the 13 Tanh-only safety-failure cases that retained first-contact status in both arms; no inferential test, intervention, training, or promotion",
        "alignment_definition": {
            "first_contact": "earliest body event with legal_cap=true; indexed by policy_step and physics substep",
            "post_contact_state_offsets": "trajectory rows at first-cap policy_step plus offsets; each row stores state after its policy step",
            "first_post_cap_action": "trajectory action at first-cap policy_step + 1; actor role inferred from evaluator routing and skill_id",
            "later_contact": "first body event after the first-cap policy step with legal_cap=false",
            "observation_limit": "actor observations are recorded only at contact callbacks; no immediate post-cap actor observation is present in the saved trajectories",
        },
        "summary": summary,
        "cases": per_case,
        "interpretation_limit": "Paired event and trajectory differences locate when the rollouts diverge; they cannot separate altered Launch handoff state from the downstream actor response or identify a causal intervention.",
        "decision": "No new training or promotion; formal C350 unchanged; P2 remains false.",
    }
    output = BASE / f"{PREFIX}-postcontact-alignment-s260925.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "summary": summary,
        "sha256": __import__("hashlib").sha256(output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
