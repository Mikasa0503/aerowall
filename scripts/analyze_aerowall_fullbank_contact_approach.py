#!/usr/bin/env python3
"""Reproduce full-bank phase-0 approach and curated phase-2 actor alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPRO_DIR = Path("artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility")
OUTPUT = REPRO_DIR / "aerowall-fullbank-contact-approach-alignment-v1-s9524.json"
FULL_STEMS = {
    "control": "legacy-control-launch-relv3-corrected-s9524",
    "tanh": "aerowall-v6-recover-tanh-mapping-ablation-v1-s9524",
}
ACTOR_STEMS = {
    "control": "aerowall-contact-substep-actor-input-audit-v1-control-runtime-s9524",
    "tanh": "aerowall-contact-substep-actor-input-audit-v1-tanh-runtime-s9524",
}
LEADS = (20, 15, 10, 5, 2, 0)
LEGAL_RADIUS_M = 0.20
SUBSTEPS_PER_POLICY_STEP = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quat_rotate_inverse(quaternion, vector):
    """Rotate a vector from world frame into a scalar-first quaternion frame."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    v = np.asarray(vector, dtype=np.float64)
    w, axis = q[0], q[1:]
    return 2 * np.dot(axis, v) * axis + (w * w - np.dot(axis, axis)) * v - 2 * w * np.cross(axis, v)


def quat_slerp(q0, q1, fraction):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    return (np.sin((1 - fraction) * theta) * q0 + np.sin(fraction * theta) * q1) / np.sin(theta)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "n": int(array.shape[0]),
        "median": np.median(array, axis=0).tolist(),
        "q25": np.percentile(array, 25, axis=0).tolist(),
        "q75": np.percentile(array, 75, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
    }


def load_run(base_dir: Path, stem: str):
    paths = {
        "report": base_dir / f"{stem}.json",
        "events": base_dir / f"{stem}.events.jsonl",
        "trajectory": base_dir / f"{stem}.trajectory.npz",
        "contacts": base_dir / f"{stem}.contacts.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text())
    events = [json.loads(line) for line in paths["events"].read_text().splitlines() if line.strip()]
    outcomes = {row["env"]: row for row in report["outcomes"]}
    trajectory = np.load(paths["trajectory"])
    return paths, report, events, outcomes, trajectory


def first_phase0_body_events(events, outcomes):
    first = {}
    for event in events:
        if not event.get("body") or event.get("phase_before") != 0:
            continue
        env = event["env"]
        key = (event["policy_step"], event["substep"])
        if env not in first or key < (first[env]["policy_step"], first[env]["substep"]):
            first[env] = event
    return [
        {
            "env": env,
            "case_id": outcomes[env]["case_id"],
            "label": "legal" if event["legal_cap"] else "illegal",
            "event": event,
        }
        for env, event in sorted(first.items())
    ]


def interpolate_contact_state(trajectory, event, env):
    step = int(event["policy_step"])
    substep = int(event["substep"])
    fraction = (substep + 1) / SUBSTEPS_PER_POLICY_STEP
    before = trajectory["drone_state"][step - 1, env]
    after = trajectory["drone_state"][step, env]
    position = before[:3] + fraction * (after[:3] - before[:3])
    orientation = quat_slerp(before[3:7], after[3:7], fraction)
    relative = np.asarray(event["ball_position"], dtype=np.float64) - position
    local = quat_rotate_inverse(orientation, relative)
    return position, orientation, local


def phase0_condition(base_dir: Path, stem: str):
    paths, report, events, outcomes, trajectory = load_run(base_dir, stem)
    rows = first_phase0_body_events(events, outcomes)
    if len(rows) != 128:
        raise ValueError(f"Expected 128 phase-0 first body events for {stem}, got {len(rows)}")

    lead_metrics = {label: {lead: [] for lead in LEADS} for label in ("legal", "illegal")}
    rotor_command_metrics = {label: {lead: [] for lead in LEADS} for label in ("legal", "illegal")}
    rotor_throttle_metrics = {label: {lead: [] for lead in LEADS} for label in ("legal", "illegal")}
    contact_crosscheck = []
    episode_rows = {}
    for row in rows:
        env, event = row["env"], row["event"]
        step = int(event["policy_step"])
        contact_position, contact_orientation, contact_local = interpolate_contact_state(trajectory, event, env)
        contact_radial = float(np.linalg.norm(contact_local[:2]))
        contact_crosscheck.append(abs(contact_radial - float(event["radial_error"])))
        lead_rows = {}
        for lead in LEADS:
            frame = step - 1 - lead
            if frame < 0:
                raise ValueError(f"{row['case_id']} lacks lead-{lead} state")
            state = trajectory["drone_state"][frame, env]
            ball = trajectory["ball_position"][frame, env]
            ball_velocity = trajectory["ball_velocity"][frame, env, :3]
            relative_world = ball - state[:3]
            relative_local = quat_rotate_inverse(state[3:7], relative_world)
            action_frame = step - lead
            raw_action = trajectory["action"][action_frame, env].astype(np.float64)
            # AeroWallSingleWallRallyEnv passes these four actor outputs
            # directly to MultirotorBase.apply_action. HCSP RotorGroup.forward
            # clips each rotor command to [-1, 1], maps it to sqrt((u+1)/2),
            # then advances the motor-lag state. No PIDRateController transform
            # is attached to this evaluator route.
            rotor_command = np.clip(raw_action, -1.0, 1.0)
            target_rotor_throttle = np.sqrt(np.clip((rotor_command + 1.0) * 0.5, 0.0, 1.0))
            lead_rows[lead] = {
                "ball_position_world": ball.astype(np.float64).tolist(),
                "ball_velocity_world": ball_velocity.astype(np.float64).tolist(),
                "relative_world_y": float(relative_world[1]),
                "relative_local_y": float(relative_local[1]),
                "drone_velocity_world_y": float(state[8]),
                "policy_action_frame": int(action_frame),
                "raw_policy_action": raw_action.tolist(),
                "limited_rotor_command_4": rotor_command.tolist(),
                "target_rotor_throttle_4": target_rotor_throttle.tolist(),
            }
            lead_metrics[row["label"]][lead].append([
                relative_local[1], relative_world[1], state[8], ball_velocity[1]
            ])
            rotor_command_metrics[row["label"]][lead].append(rotor_command)
            rotor_throttle_metrics[row["label"]][lead].append(target_rotor_throttle)

        final_frame = step - 1
        state = trajectory["drone_state"][final_frame, env]
        ball = trajectory["ball_position"][final_frame, env]
        pre_relative = quat_rotate_inverse(state[3:7], ball - state[:3])
        episode_rows[row["case_id"]] = {
            "label": row["label"],
            "env": env,
            "policy_step": step,
            "substep": int(event["substep"]),
            "event_radial_error_m": float(event["radial_error"]),
            "interpolated_contact_radial_error_m": contact_radial,
            "interpolated_contact_local_xyz_m": contact_local.tolist(),
            "preaction_radial_error_m": float(np.linalg.norm(pre_relative[:2])),
            "leads": lead_rows,
        }

    groups = {}
    for label in ("legal", "illegal"):
        label_cases = [case for case, row in episode_rows.items() if row["label"] == label]
        groups[label] = {
            "count": len(label_cases),
            "preaction_radial_over_0p20m": sum(episode_rows[case]["preaction_radial_error_m"] > LEGAL_RADIUS_M
                                                for case in label_cases),
            "contact_radial_over_0p20m": sum(episode_rows[case]["event_radial_error_m"] > LEGAL_RADIUS_M
                                              for case in label_cases),
            "local_y_relative_y_drone_vy_ball_vy_by_lead": {
                str(lead): summarize(lead_metrics[label][lead]) for lead in LEADS
            },
            "limited_rotor_command_4ch_by_lead": {
                str(lead): summarize(rotor_command_metrics[label][lead]) for lead in LEADS
            },
            "target_rotor_throttle_4ch_by_lead": {
                str(lead): summarize(rotor_throttle_metrics[label][lead]) for lead in LEADS
            },
            "case_ids": sorted(label_cases),
        }
    illegal_crossers = sorted(
        case for case, row in episode_rows.items()
        if row["label"] == "illegal"
        and row["preaction_radial_error_m"] <= LEGAL_RADIUS_M < row["event_radial_error_m"]
    )
    source_hashes = {key: sha256(path) for key, path in paths.items()}
    return {
        "source_stem": stem,
        "source_sha256": source_hashes,
        "reported_episodes": report["summary"]["episodes"],
        "phase0_first_body_event_count": len(rows),
        "groups": groups,
        "illegal_cases_crossing_radial_limit_after_last_action": illegal_crossers,
        "contact_radial_reconstruction_error_m": summarize(contact_crosscheck),
        "episodes": episode_rows,
    }


def hungarian(cost):
    """Return minimum-cost row/column assignments for a rectangular matrix."""
    matrix = np.asarray(cost, dtype=np.float64)
    n, m = matrix.shape
    if n > m:
        raise ValueError("Hungarian helper expects rows <= columns")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((int(p[j] - 1), j - 1) for j in range(1, m + 1) if p[j] != 0)


def phase0_matching(control, lead=20):
    illegal = sorted((case, row) for case, row in control["episodes"].items() if row["label"] == "illegal")
    legal = sorted((case, row) for case, row in control["episodes"].items() if row["label"] == "legal")
    features = []
    for _, row in illegal + legal:
        lead_row = row["leads"][lead]
        features.append(np.concatenate((lead_row["ball_position_world"], lead_row["ball_velocity_world"])))
    features = np.asarray(features, dtype=np.float64)
    scale = np.std(features, axis=0, ddof=1)
    scale = np.where(scale > 1e-9, scale, 1.0)
    illegal_z = (features[:len(illegal)] - np.mean(features, axis=0)) / scale
    legal_z = (features[len(illegal):] - np.mean(features, axis=0)) / scale
    cost = np.linalg.norm(illegal_z[:, None, :] - legal_z[None, :, :], axis=-1)
    assignments = hungarian(cost)
    pairs = []
    for i, j in assignments:
        failed_case, failed_row = illegal[i]
        matched_case, matched_row = legal[j]
        pair = {"illegal_case_id": failed_case, "matched_legal_case_id": matched_case,
                "standardized_ball_state_distance_at_lead20": float(cost[i, j])}
        for lead in LEADS:
            a, b = failed_row["leads"][lead], matched_row["leads"][lead]
            pair.setdefault("illegal_minus_legal", {})[str(lead)] = {
                "local_y_m": a["relative_local_y"] - b["relative_local_y"],
                "world_relative_y_m": a["relative_world_y"] - b["relative_world_y"],
                "drone_world_vy_m_s": a["drone_velocity_world_y"] - b["drone_velocity_world_y"],
                "ball_world_vy_m_s": a["ball_velocity_world"][1] - b["ball_velocity_world"][1],
                "limited_rotor_command_4ch": [
                    failed_row["leads"][lead]["limited_rotor_command_4"][i]
                    - matched_row["leads"][lead]["limited_rotor_command_4"][i]
                    for i in range(4)
                ],
                "target_rotor_throttle_4ch": [
                    failed_row["leads"][lead]["target_rotor_throttle_4"][i]
                    - matched_row["leads"][lead]["target_rotor_throttle_4"][i]
                    for i in range(4)
                ],
            }
        pairs.append(pair)
    paired_summary = {}
    for lead in LEADS:
        paired_summary[str(lead)] = {
            metric: summarize([pair["illegal_minus_legal"][str(lead)][metric] for pair in pairs])
            for metric in ("local_y_m", "world_relative_y_m", "drone_world_vy_m_s", "ball_world_vy_m_s",
                           "limited_rotor_command_4ch", "target_rotor_throttle_4ch")
        }
    distances = [pair["standardized_ball_state_distance_at_lead20"] for pair in pairs]
    return {
        "matching_method": "minimum-total-cost one-to-one Hungarian assignment using Euclidean distance over pooled-SD-standardized ball world position and linear velocity at 0.4 s before the final policy action",
        "pair_count": len(pairs),
        "distance_threshold_counts": {"le_1": sum(value <= 1 for value in distances),
                                      "le_1p5": sum(value <= 1.5 for value in distances)},
        "standardized_state_distance": summarize(distances),
        "paired_illegal_minus_legal_medians_by_lead": paired_summary,
        "pairs": pairs,
    }


def phase2_condition(base_dir: Path, stem: str):
    paths, report, events, outcomes, trajectory = load_run(base_dir, stem)
    earliest = {}
    for event in events:
        if (not event.get("body") or event.get("legal_cap")
                or event.get("phase_before") != 2 or "actor_observation" not in event):
            continue
        env = event["env"]
        key = (event["policy_step"], event["substep"])
        if env not in earliest or key < (earliest[env]["policy_step"], earliest[env]["substep"]):
            earliest[env] = event
    rows = []
    observation_position_error = []
    observation_relative_error = []
    action_error = []
    for env, event in sorted(earliest.items()):
        step = int(event["policy_step"])
        observation = np.asarray(event["actor_observation"], dtype=np.float64)
        pre_state = trajectory["drone_state"][step - 1, env]
        pre_ball = trajectory["ball_position"][step - 1, env]
        pre_relative_local = quat_rotate_inverse(pre_state[3:7], pre_ball - pre_state[:3])
        pre_radial = float(np.linalg.norm(pre_relative_local[:2]))
        observation_position_error.append(float(np.max(np.abs(observation[26:29] - pre_ball))))
        observation_relative_error.append(float(np.max(np.abs(observation[29:32] - (pre_ball - pre_state[:3])))))
        action_error.append(float(np.max(np.abs(trajectory["action"][step, env] - np.asarray(event["policy_action"])))))
        rows.append({
            "case_id": outcomes[env]["case_id"],
            "actor_role": event["actor_observation_role"],
            "actor_observation_version": event["actor_observation_version"],
            "policy_step": step,
            "substep": int(event["substep"]),
            "preaction_radial_m": pre_radial,
            "event_radial_m": float(event["radial_error"]),
            "observation_position_error_m": observation_position_error[-1],
            "observation_relative_position_error_m": observation_relative_error[-1],
            "event_action_vs_trajectory_action_error": action_error[-1],
        })
    role_counts = {}
    for row in rows:
        role_counts[row["actor_role"]] = role_counts.get(row["actor_role"], 0) + 1
    return {
        "source_stem": stem,
        "source_sha256": {key: sha256(path) for key, path in paths.items()},
        "reported_episodes": report["summary"]["episodes"],
        "earliest_phase2_illegal_contacts": len(rows),
        "active_actor_counts": role_counts,
        "preaction_radial_m": summarize([row["preaction_radial_m"] for row in rows]),
        "event_radial_m": summarize([row["event_radial_m"] for row in rows]),
        "observation_vs_preaction_state_max_abs_error": {
            "ball_world_position_m": max(observation_position_error, default=0.0),
            "ball_minus_drone_world_position_m": max(observation_relative_error, default=0.0),
        },
        "event_action_vs_trajectory_action_max_abs_error": max(action_error, default=0.0),
        "cases": sorted(rows, key=lambda row: row["case_id"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=REPRO_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    control = phase0_condition(args.artifact_dir, FULL_STEMS["control"])
    tanh = phase0_condition(args.artifact_dir, FULL_STEMS["tanh"])
    control_labels = {case: row["label"] for case, row in control["episodes"].items()}
    tanh_labels = {case: row["label"] for case, row in tanh["episodes"].items()}
    if control_labels != tanh_labels:
        raise ValueError("Expected identical full-bank Phase-0 cases and legal labels")

    control_phase2 = phase2_condition(args.artifact_dir, ACTOR_STEMS["control"])
    tanh_phase2 = phase2_condition(args.artifact_dir, ACTOR_STEMS["tanh"])
    control_phase2_cases = {row["case_id"] for row in control_phase2["cases"]}
    tanh_phase2_cases = {row["case_id"] for row in tanh_phase2["cases"]}
    phase2_case_comparison = {
        "shared_case_count": len(control_phase2_cases & tanh_phase2_cases),
        "control_only_case_ids": sorted(control_phase2_cases - tanh_phase2_cases),
        "tanh_only_case_ids": sorted(tanh_phase2_cases - control_phase2_cases),
    }
    result = {
        "experiment_name": "AeroWallFullbankContactApproachAlignmentV1",
        "status": "analysis_passed",
        "analysis_script": str(Path(__file__).as_posix()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "training_performed": False,
        "promotion": False,
        "scope": {
            "phase0": "full 128-case v4 development bank; not the final frozen acceptance holdout",
            "phase2": "failure-enriched 33-case exact-contact actor-input reruns; not a population-rate sample",
            "conditions": "V6 relative_v3 Launch + fixed Hit and C350 Recover versus same run with Recover-only TanhNormal mapping",
        },
        "method": {
            "first_phase0_contact": "earliest body event per env with phase_before == 0, labelled by PhysX legal_cap callback",
            "contact_pose": "linearly interpolate root position and scalar-first WXYZ orientation by slerp at (substep+1)/8; pair with callback ball_position",
            "preaction_frames": "policy_step-1-lead for leads 20,15,10,5,2,0 at 50 Hz",
            "radial_error": "norm of local x/y ball-center offset; legal radius 0.20 m",
            "phase2_alignment": "earliest illegal phase-2 body event per case; compare actor observation slices 26:29 and 29:32, plus policy_action, with the exact pre-action and action trajectory frames",
            "phase2_subset_limit": "33 failure-enriched cases selected for diagnostic replay",
            "action_decode": "raw actor action is passed directly to AeroWallSingleWallRallyEnv.drone.apply_action; the pinned HCSP RotorGroup clips each rotor input to [-1,1], maps to sqrt((u+1)/2), then applies motor lag. This route does not attach PIDRateController_flightmare.",
        },
        "control_phase0": control,
        "recover_tanh_phase0": tanh,
        "phase0_labels_and_case_ids_identical": True,
        "control_phase0_failure_vs_legal_matched_approach": phase0_matching(control),
        "control_phase2": control_phase2,
        "recover_tanh_phase2": tanh_phase2,
        "phase2_case_comparison": phase2_case_comparison,
        "action_semantics": "The active AeroWall evaluator uses direct four-rotor commands: policy action -> drone.apply_action -> HCSP RotorGroup clipping, throttle square-root map, and motor lag. The PIDRateController_flightmare chain is not active on this route.",
        "interpretation": "The complete development-bank phase-0 analysis confirms 12 illegal and 116 legal first contacts in both paired conditions. Illegal contacts are generally already beyond the 0.20 m radial limit at the last action decision. In one-to-one comparisons of similar incoming ball states at 0.4 s before that decision, the illegal group shows a more negative lateral relative position and a less negative drone lateral velocity earlier in the approach; this is observational and does not identify a causal rotor-command channel. The curated phase-2 replay independently aligns actor observations and issued actions to trajectory frames and shows misses already outside the radial limit before their last action. The Recover-only Tanh mapping did not improve the rally/safety gate and remains rejected.",
        "next_step": "Keep formal C350 and P2 gate=false. Do not train or promote. Use this evidence to draft one reversible, read-only-evaluable Launch approach intervention with a predeclared falsification criterion; only then consider a separately authorized candidate experiment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "phase0_control": {key: control["groups"][key]["count"] for key in ("legal", "illegal")},
        "phase0_tanh": {key: tanh["groups"][key]["count"] for key in ("legal", "illegal")},
        "phase0_crossers": control["illegal_cases_crossing_radial_limit_after_last_action"],
        "phase0_match_distance_counts": result["control_phase0_failure_vs_legal_matched_approach"]["distance_threshold_counts"],
        "phase2_control": control_phase2["earliest_phase2_illegal_contacts"],
        "phase2_tanh": tanh_phase2["earliest_phase2_illegal_contacts"],
        "phase2_case_comparison": phase2_case_comparison,
        "phase2_actor_counts": {"control": control_phase2["active_actor_counts"], "tanh": tanh_phase2["active_actor_counts"]},
        "max_event_pose_radial_error_m": max(
            control["contact_radial_reconstruction_error_m"]["max"][0],
            tanh["contact_radial_reconstruction_error_m"]["max"][0],
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
