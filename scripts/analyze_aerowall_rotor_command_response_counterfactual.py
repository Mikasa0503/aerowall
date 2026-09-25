"""Analyze AeroWallRotorCommandResponseCounterfactualV1 without training."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility"
PREFIX = "aerowall-rotor-command-response-counterfactual-v1"
OUT = ARTIFACTS / (PREFIX + "-analysis-s9524.json")
BANK_SHA256 = "cc9d93b5f3ecb2b367e1f3c209b080963ef535b38be63f2d004b38b4c35af0ec"
CHECKPOINTS = {
    "launch": ("artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-bounded-intercept-v1-causal-v6-s6101.launch.pt",
               "bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50"),
    "hit": ("artifacts/wall-skill-upgrade-v3/aerowall-bounded-goal-hit-v1-tanh-s6201-resume20.pt",
            "d970228e9c2fdc662fc6fa9b0aead0672628ffd0da947dd04a93685a31968dd7"),
    "recovery": ("checkpoints/c-u350.pt",
                 "4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659"),
}
EXPECTED_CASE_IDS = (
    "handoff-heldout-0008-env-0008", "handoff-heldout-0012-env-0012",
    "handoff-heldout-0018-env-0018", "handoff-heldout-0033-env-0033",
    "handoff-heldout-0051-env-0051", "handoff-heldout-0059-env-0059",
    "handoff-heldout-0073-env-0073", "handoff-heldout-0082-env-0082",
    "handoff-heldout-0099-env-0099", "handoff-heldout-0107-env-0107",
    "handoff-heldout-0112-env-0112", "handoff-heldout-0113-env-0113",
)
ARMS = (
    ("control", None, None),
    ("r0plus", 0, 0.10), ("r0minus", 0, -0.10),
    ("r1plus", 1, 0.10), ("r1minus", 1, -0.10),
    ("r2plus", 2, 0.10), ("r2minus", 2, -0.10),
    ("r3plus", 3, 0.10), ("r3minus", 3, -0.10),
)
SAFETY_FAILURE_CODES = {2, 4, 5}
FAILURE_NAMES = {
    0: "none", 1: "ball_ground", 2: "drone_ground",
    3: "out_of_bounds", 4: "illegal_contact", 5: "drone_wall",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def read_json(path):
    return json.loads(path.read_text())


def paths(arm):
    stem = ARTIFACTS / (PREFIX + "-" + arm + "-s9524")
    return {
        "report": stem.with_suffix(".json"),
        "trajectory": stem.with_suffix(".trajectory.npz"),
        "events": stem.with_suffix(".events.jsonl"),
        "contacts": stem.with_suffix(".contacts.json"),
        "run_log": stem.with_suffix(".run.log"),
    }


def first_body_events(event_path):
    first = {}
    rows = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
    for row in rows:
        if not row.get("body"):
            continue
        env = int(row["env"])
        marker = (int(row["policy_step"]), int(row["substep"]))
        if env not in first or marker < (int(first[env]["policy_step"]), int(first[env]["substep"])):
            first[env] = row
    return first, rows


def audit_first_contacts(first, contact_rows):
    result = {}
    for env, event in first.items():
        base_link = "/World/envs/env_{}/IrisTest_0/base_link".format(env)
        ball = "/World/envs/env_{}/ball".format(env)
        result[env] = any(
            int(contact["step"]) == int(event["policy_step"])
            and int(contact["substep"]) == int(event["substep"])
            and int(contact["contact_count"]) > 0
            and base_link in contact["actors"] and ball in contact["actors"]
            for contact in contact_rows
        )
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    prereg_path = ROOT / "docs/plans/2026-09-25-aerowall-rotor-command-response-counterfactual-v1.md"
    evaluator_path = ROOT / "scripts/evaluate_aerowall_wall_rl.py"
    env_path = ROOT / "scripts/aerowall_wall_rally_env.py"
    reward_path = ROOT / "scripts/aerowall_wall_reward_logic.py"
    launch_report_path = ROOT / CHECKPOINTS["launch"][0].replace(".launch.pt", ".json")
    hit_report_path = ROOT / "artifacts/wall-skill-upgrade-v3/aerowall-bounded-goal-hit-v1-tanh-s6201-resume20.json"
    analysis_script_path = Path(__file__).resolve()
    bank_path = ARTIFACTS / "aerowall-launch-lateral-momentum-counterfactual-v1-control-s9524.json"
    pair_analysis_path = ARTIFACTS / "aerowall-launch-lateral-momentum-counterfactual-v1-analysis-s9524.json"
    source_handoffs_path = ARTIFACTS / "aerowall-launch-lateral-momentum-counterfactual-v1-source-handoffs-s9524.json"

    bank = read_json(bank_path)
    require(digest(bank_path) == BANK_SHA256, "initial handoff bank SHA256 changed")
    require(bank.get("experiment_name") == "AeroWallLaunchLateralMomentumCounterfactualV1", "wrong initial bank experiment")
    require(bank.get("condition") == "control" and bank.get("count") == 12, "wrong control handoff bank")
    cases = bank["cases"]
    case_ids = tuple(case["case_id"] for case in cases)
    require(case_ids == EXPECTED_CASE_IDS, "initial handoff case order/IDs changed")
    case_index = {case_id: i for i, case_id in enumerate(case_ids)}

    pairing_report = read_json(pair_analysis_path)
    pairing_rows = pairing_report["per_case"]
    require(len(pairing_rows) == 12, "matched legal reference map must contain 12 pairs")
    source = read_json(source_handoffs_path)
    source_by_id = {case["source_case_id"]: case for case in source["cases"]}
    reference_by_failed_id = {}
    for row in pairing_rows:
        failed_id = row["source_failed_case_id"]
        legal_id = row["matched_legal_reference_case_id"]
        require(failed_id in {case["source_case_id"] for case in cases} and legal_id in source_by_id, "pair mapping refers to an unknown handoff case")
        reference_by_failed_id[failed_id] = {
            "case_id": legal_id,
            "world_y_velocity_m_s": float(source_by_id[legal_id]["drone_velocity"][1]),
        }
    require(set(reference_by_failed_id) == {case["source_case_id"] for case in cases},
            "pair mapping does not cover the registered failure cases")

    source_hashes = {
        relative(path): digest(path)
        for path in (prereg_path, evaluator_path, env_path, reward_path, bank_path,
                     pair_analysis_path, source_handoffs_path, launch_report_path, hit_report_path, analysis_script_path)
    }
    for role, (checkpoint_rel, expected_sha) in CHECKPOINTS.items():
        checkpoint_path = ROOT / checkpoint_rel
        actual_sha = digest(checkpoint_path)
        require(actual_sha == expected_sha, "{} checkpoint SHA256 changed".format(role))
        source_hashes[relative(checkpoint_path)] = actual_sha

    run_data = {}
    for arm, channel, delta in ARMS:
        arm_paths = paths(arm)
        report = read_json(arm_paths["report"])
        require(report.get("status") == "passed", "{} report did not complete".format(arm))
        require(report.get("num_envs") == 12 and report.get("seed") == 9524, "{} run used wrong bank size/seed".format(arm))
        require(report.get("protocol") == "natural" and report.get("stage") == "RALLY", "{} run was not natural RALLY".format(arm))
        require(report.get("initial_case_bank_sha256") == BANK_SHA256, "{} report has wrong initial bank".format(arm))
        require(report.get("candidate_name") == "AeroWallBoundedInterceptV1-CausalV6", "{} report has wrong Launch candidate".format(arm))
        require(report.get("hit_candidate_name") == "AeroWallBoundedGoalHitV1-Tanh-S6201", "{} report has wrong Hit candidate".format(arm))
        require(report.get("candidate_actor_checkpoint_sha256") == CHECKPOINTS["launch"][1], "{} Launch checkpoint mismatch".format(arm))
        require(report.get("hit_checkpoint_sha256") == CHECKPOINTS["hit"][1], "{} Hit checkpoint mismatch".format(arm))
        require(report.get("recovery_checkpoint_sha256") == CHECKPOINTS["recovery"][1], "{} Recover checkpoint mismatch".format(arm))
        require(report.get("train_skill") is None, "{} unexpectedly trained a skill".format(arm))
        require(report.get("observation_version") == "relative_v3", "{} environment observation mismatch".format(arm))
        require(report.get("skill_observation_version") == "relative_v2", "{} skill observation mismatch".format(arm))
        require(report.get("reward_design") == "aerowall_causal_v6", "{} reward mismatch".format(arm))
        require(report.get("launch_action_distribution") == "tanh"
                and report.get("hit_action_distribution") == "tanh"
                and report.get("recovery_action_distribution") == "default", "{} action mapping mismatch".format(arm))
        require(report.get("gate", {}).get("applicable") is False, "{} conditional handoff run was incorrectly a formal gate".format(arm))
        require(report.get("summary", {}).get("episodes") == 12, "{} report is missing episode outcomes".format(arm))
        require(tuple(row["case_id"] for row in report["outcomes"]) == EXPECTED_CASE_IDS,
                "{} outcome case order/IDs changed".format(arm))

        reset_state = report.get("reproducibility", {}).get("post_reset_state", {})
        require(reset_state.get("capture_point") == "after_env_reset_before_first_policy_action",
                "{} reset fingerprint is missing".format(arm))
        trajectory = np.load(arm_paths["trajectory"])
        require(trajectory["drone_state"].shape[0] >= 10, "{} trajectory is shorter than the step-9 endpoint".format(arm))
        require(trajectory["active"][9].astype(bool).all(), "{} has a case inactive before step 9".format(arm))
        require(trajectory["drone_state"].shape[1] == 12, "{} trajectory has wrong environment count".format(arm))
        first, all_events = first_body_events(arm_paths["events"])
        require(set(first) == set(range(12)), "{} does not have a first body contact for all 12 cases".format(arm))
        contact_rows = read_json(arm_paths["contacts"])
        physx = audit_first_contacts(first, contact_rows)
        require(all(physx.values()), "{} has a first contact without PhysX corroboration".format(arm))
        audit = report.get("contact_audit", {})
        require(not audit.get("callback_errors"), "{} has contact callback errors".format(arm))
        require(audit.get("legal_events") == audit.get("corroborated_events"), "{} legal contact audit mismatch".format(arm))
        require(audit.get("wall_events") == audit.get("corroborated_wall_events"), "{} wall contact audit mismatch".format(arm))

        intervention = report.get("action_intervention")
        if channel is None:
            require(intervention is None, "control arm unexpectedly has an action override")
        else:
            require(intervention and intervention.get("name") == "AeroWallRotorCommandResponseCounterfactualV1",
                    "{} missing registered action intervention metadata".format(arm))
            require(intervention.get("channel") == channel and abs(intervention.get("delta") - delta) < 1e-12,
                    "{} intervention channel/sign mismatch".format(arm))
            trace = intervention["per_step_effective_command_trace"]
            require(intervention.get("applied_env_step_count") == 120 and len(trace) == 120,
                    "{} did not apply all 10 intervention steps to all 12 cases".format(arm))
            counts = {}
            for row in trace:
                counts[int(row["env"])] = counts.get(int(row["env"]), 0) + 1
                require(int(row["rotor_index"]) == channel and int(row["policy_step"]) in range(10),
                        "{} has an intervention outside its registered channel/horizon".format(arm))
                require(abs(float(row["delta"]) - delta) < 1e-12, "{} trace has an unexpected signed offset".format(arm))
                expected_raw = float(row["policy_action_before_override"]) + delta
                expected_limited = max(-1.0, min(1.0, expected_raw))
                require(abs(expected_raw - float(row["effective_action_before_rotor_limit"])) < 1e-6,
                        "{} raw command trace is inconsistent".format(arm))
                require(abs(expected_limited - float(row["rotor_command_after_limit"])) < 1e-6,
                        "{} clipped command trace is inconsistent".format(arm))
                env_i = int(row["env"])
                step_i = int(row["policy_step"])
                require(abs(float(trajectory["action"][step_i, env_i, channel])
                            - float(row["policy_action_before_override"])) < 1e-6,
                        "{} trajectory raw action disagrees with its command trace".format(arm))
                require(abs(float(trajectory["executed_action"][step_i, env_i, channel])
                            - expected_raw) < 1e-6, "{} trajectory effective action disagrees with its command trace".format(arm))
                require(abs(float(trajectory["actuator_command"][step_i, env_i, channel])
                            - expected_limited) < 1e-6, "{} actuator command disagrees with its command trace".format(arm))
            require(counts == {i: 10 for i in range(12)}, "{} did not cover each case for 10 steps".format(arm))

        outcome_by_id = {outcome["case_id"]: outcome for outcome in report["outcomes"]}
        failure_counts = {name: 0 for name in FAILURE_NAMES.values()}
        for outcome in report["outcomes"]:
            failure_counts[FAILURE_NAMES[int(outcome["failure"])]] += 1
        require(failure_counts == report["summary"]["failure_counts"],
                "{} numeric failure codes do not match evaluator failure counts".format(arm))

        boundary_cases = []
        boundary_extrema = {}
        trajectory_active = trajectory["active"].astype(bool)
        for index, case_id in enumerate(case_ids):
            active_y = trajectory["ball_position"][trajectory_active[:, index], index, 1]
            min_y = float(active_y.min())
            max_y = float(active_y.max())
            boundary_extrema[case_id] = {"min_world_y_m": min_y, "max_world_y_m": max_y}
            if np.any(np.abs(active_y) > 3.0):
                boundary_cases.append(case_id)

        per_case_contact = {}
        for env, event in first.items():
            case_id = case_ids[env]
            per_case_contact[case_id] = {
                "policy_step": int(event["policy_step"]),
                "substep": int(event["substep"]),
                "legal": bool(event["legal_cap"]),
                "radial_error_m": float(event["radial_error"]),
                "physx_audited": bool(physx[env]),
            }

        report_hashes = {}
        for key in ("report", "trajectory", "events", "contacts"):
            file_path = arm_paths[key]
            report_hashes[relative(file_path)] = digest(file_path)
        source_hashes.update(report_hashes)

        run_data[arm] = {
            "report": report,
            "trajectory": trajectory,
            "first_contacts": per_case_contact,
            "outcomes": outcome_by_id,
            "failure_counts": failure_counts,
            "physx_audited_first_contacts": sum(physx.values()),
            "all_first_contacts_physx_audited": all(physx.values()),
            "boundary_crossing_cases": boundary_cases,
            "ball_world_y_extrema_by_case": boundary_extrema,
            "contact_event_count": len(all_events),
            "reset_state_sha256": reset_state["state_sha256"],
            "rng_state_sha256": reset_state["rng_state_sha256"],
            "command_saturation_rows": (
                sum(abs(float(row["effective_limited_command_delta"]) - float(row["delta"])) > 1e-8
                    for row in intervention["per_step_effective_command_trace"])
                if intervention else 0
            ),
        }

    control = run_data["control"]
    reset_hashes = {row["reset_state_sha256"] for row in run_data.values()}
    require(len(reset_hashes) == 1, "the 9 arms did not start from an identical recorded state")
    rng_consistent = {}
    for rng_name in ("torch_cpu", "numpy", "torch_cuda"):
        hashes = {json.dumps(row["rng_state_sha256"].get(rng_name), sort_keys=True) for row in run_data.values()}
        rng_consistent[rng_name] = len(hashes) == 1
        require(rng_consistent[rng_name], "{} RNG fingerprints differ across arms".format(rng_name))
    python_rng_hashes = {arm: row["rng_state_sha256"].get("python") for arm, row in run_data.items()}
    control_first_actions = control["trajectory"]["action"][0].astype(float)
    source_action_errors = [
        float(np.max(np.abs(control_first_actions[i] - np.asarray(cases[i]["source_policy_action"], dtype=float))))
        for i in range(12)
    ]
    first_action_match_errors = {
        arm: float(np.max(np.abs(data["trajectory"]["action"][0].astype(float) - control_first_actions)))
        for arm, data in run_data.items()
    }
    require(max(source_action_errors) <= 1e-5, "fresh control first actions do not reproduce the frozen source actions")
    require(max(first_action_match_errors.values()) <= 1e-6, "first actor actions differ across paired arms")
    control_vy_step9 = control["trajectory"]["drone_state"][9, :, 8].astype(float)
    control_safety_cases = {
        case_id for case_id, outcome in control["outcomes"].items()
        if int(outcome["failure"]) in SAFETY_FAILURE_CODES
    }
    control_boundary_cases = set(control["boundary_crossing_cases"])
    ref_per_case = {}
    for index, case in enumerate(cases):
        failed_id = case["source_case_id"]
        ref_per_case[case["case_id"]] = reference_by_failed_id[failed_id]

    baseline_first_contacts = [control["first_contacts"][case_id] for case_id in case_ids]
    baseline_summary = {
        "episodes": 12,
        "legal_first_contacts": sum(int(row["legal"]) for row in baseline_first_contacts),
        "median_first_contact_radial_error_m": float(np.median([row["radial_error_m"] for row in baseline_first_contacts])),
        "safety_failure_case_ids": sorted(control_safety_cases),
        "boundary_crossing_case_ids": sorted(control_boundary_cases),
        "failure_counts": control["failure_counts"],
        "all_12_first_contacts_physx_audited": control["all_first_contacts_physx_audited"],
        "mean_rallies": float(control["report"]["summary"]["mean_rallies"]),
    }

    treatment_results = {}
    response_matrix = {}
    for arm, channel, delta in ARMS:
        if channel is None:
            continue
        treatment = run_data[arm]
        trajectory = treatment["trajectory"]
        treatment_vy = trajectory["drone_state"][9, :, 8].astype(float)
        per_case = []
        effects = []
        toward_count = 0
        for index, case_id in enumerate(case_ids):
            reference = ref_per_case[case_id]
            control_vy = float(control_vy_step9[index])
            treated_vy = float(treatment_vy[index])
            ref_vy = float(reference["world_y_velocity_m_s"])
            effect = treated_vy - control_vy
            control_gap = abs(control_vy - ref_vy)
            treatment_gap = abs(treated_vy - ref_vy)
            toward = treatment_gap < control_gap
            toward_count += int(toward)
            effects.append(abs(effect))
            per_case.append({
                "case_id": case_id,
                "matched_legal_reference_case_id": reference["case_id"],
                "reset_world_y_velocity_m_s": float(cases[index]["drone_velocity"][1]),
                "matched_legal_reference_world_y_velocity_m_s": ref_vy,
                "control_world_y_velocity_after_step9_m_s": control_vy,
                "treatment_world_y_velocity_after_step9_m_s": treated_vy,
                "treatment_minus_control_world_y_velocity_m_s": effect,
                "absolute_treatment_effect_m_s": abs(effect),
                "reference_gap_before_m_s": control_gap,
                "reference_gap_after_m_s": treatment_gap,
                "moves_toward_matched_reference": bool(toward),
                "contact": treatment["first_contacts"][case_id],
                "terminal_failure": FAILURE_NAMES[int(treatment["outcomes"][case_id]["failure"])],
                "ball_world_y_extrema_m": treatment["ball_world_y_extrema_by_case"][case_id],
            })

        median_abs_effect = float(np.median(np.asarray(effects)))
        local_signal = toward_count >= 9 and median_abs_effect >= 0.02
        safety_cases = {
            case_id for case_id, outcome in treatment["outcomes"].items()
            if int(outcome["failure"]) in SAFETY_FAILURE_CODES
        }
        new_safety = sorted(safety_cases - control_safety_cases)
        boundary_cases = set(treatment["boundary_crossing_cases"])
        new_boundary = sorted(boundary_cases - control_boundary_cases)
        rejected = bool(new_safety or new_boundary)
        treatment_contacts = [treatment["first_contacts"][case_id] for case_id in case_ids]
        results = {
            "rotor_index": channel,
            "signed_actor_output_delta": delta,
            "intervention_decisions_per_case": 10,
            "intervention_command_rows": 120,
            "command_saturation_rows": treatment["command_saturation_rows"],
            "toward_reference_case_count": toward_count,
            "median_absolute_world_y_velocity_effect_m_s": median_abs_effect,
            "median_signed_world_y_velocity_effect_m_s": float(np.median([row["treatment_minus_control_world_y_velocity_m_s"] for row in per_case])),
            "mean_signed_world_y_velocity_effect_m_s": float(np.mean([row["treatment_minus_control_world_y_velocity_m_s"] for row in per_case])),
            "preregistered_local_signal": {
                "toward_reference_at_least_9_of_12": toward_count >= 9,
                "median_absolute_effect_at_least_0p02_m_s": median_abs_effect >= 0.02,
                "passed": bool(local_signal),
            },
            "legal_first_contacts": sum(int(row["legal"]) for row in treatment_contacts),
            "median_first_contact_radial_error_m": float(np.median([row["radial_error_m"] for row in treatment_contacts])),
            "first_contact_physx_audited_count": treatment["physx_audited_first_contacts"],
            "all_12_first_contacts_physx_audited": treatment["all_first_contacts_physx_audited"],
            "failure_counts": treatment["failure_counts"],
            "new_safety_failure_case_ids": new_safety,
            "boundary_crossing_case_ids": sorted(boundary_cases),
            "new_boundary_crossing_case_ids": new_boundary,
            "rejected_for_candidate_use_by_preregistered_safety_rule": rejected,
            "per_case": per_case,
        }
        treatment_results[arm] = results
        response_matrix[arm] = [
            {
                "case_id": row["case_id"],
                "effect_m_s": row["treatment_minus_control_world_y_velocity_m_s"],
                "toward_reference": row["moves_toward_matched_reference"],
            }
            for row in per_case
        ]

    local_arms = [arm for arm, result in treatment_results.items()
                  if result["preregistered_local_signal"]["passed"]]
    safety_eligible_local_arms = [
        arm for arm in local_arms
        if not treatment_results[arm]["rejected_for_candidate_use_by_preregistered_safety_rule"]
    ]

    result = {
        "schema_version": 1,
        "experiment_name": "AeroWallRotorCommandResponseCounterfactualV1",
        "status": "local_action_signal_found" if local_arms else "no_preregistered_local_action_signal",
        "interpretation": "no_training_local_action_sensitivity_screen_on_12_selected_heldout_cases",
        "does_not_pass_p2": True,
        "authorizes_training_or_promotion": False,
        "formal_policy_unchanged": "C350",
        "scope": {
            "seed": 9524,
            "case_count": 12,
            "arms": [name for name, _, _ in ARMS],
            "total_case_continuations": 108,
            "action_intervention_steps": list(range(10)),
            "measure": "world-y drone velocity after policy step 9, compared with same-seed control and each case's matched legal reference",
            "world_y_velocity_state_index": 8,
            "actor_action_channels": "direct rotor command indices 0-3",
            "source_bank_sha256": BANK_SHA256,
            "checkpoints": {role: {"path": path, "sha256": sha} for role, (path, sha) in CHECKPOINTS.items()},
        },
        "criteria": {
            "shared_rotor_index_and_sign_moves_toward_reference_in_at_least_9_of_12": True,
            "median_absolute_velocity_change_threshold_m_s": 0.02,
            "training_or_promotion_permitted": False,
            "candidate_use_rejects_any_new_ball_boundary_or_drone_ground_drone_wall_illegal_contact_failure": True,
        },
        "control_summary": baseline_summary,
        "paired_reproducibility": {
            "identical_post_reset_state_sha256": next(iter(reset_hashes)),
            "torch_numpy_cuda_rng_fingerprints_match_across_arms": rng_consistent,
            "python_stdlib_rng_fingerprint_match_across_arms": len(set(python_rng_hashes.values())) == 1,
            "python_stdlib_rng_fingerprints_by_arm": python_rng_hashes,
            "control_step0_actor_action_max_abs_error_vs_frozen_source_actions": max(source_action_errors),
            "step0_actor_action_max_abs_difference_vs_control_by_arm": first_action_match_errors,
            "note": "Evaluator seeds Torch and NumPy but not Python stdlib random; post-reset state and first actor actions match, while the stdlib RNG fingerprint varies across fresh processes.",
        },
        "local_signal_arms": local_arms,
        "local_signal_arms_not_rejected_by_preregistered_safety_rule": safety_eligible_local_arms,
        "treatment_results": treatment_results,
        "response_matrix": response_matrix,
        "artifact_sha256": dict(sorted(source_hashes.items())),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temp = OUT.with_suffix(OUT.suffix + ".tmp")
    temp.write_text(encoded)
    temp.replace(OUT)
    print(json.dumps({
        "analysis_path": relative(OUT),
        "sha256": digest(OUT),
        "status": result["status"],
        "local_signal_arms": local_arms,
        "safety_eligible_local_arms": safety_eligible_local_arms,
        "control_summary": baseline_summary,
        "paired_reproducibility": {
            "identical_post_reset_state_sha256": next(iter(reset_hashes)),
            "torch_numpy_cuda_rng_fingerprints_match_across_arms": rng_consistent,
            "python_stdlib_rng_fingerprint_match_across_arms": len(set(python_rng_hashes.values())) == 1,
            "python_stdlib_rng_fingerprints_by_arm": python_rng_hashes,
            "control_step0_actor_action_max_abs_error_vs_frozen_source_actions": max(source_action_errors),
            "step0_actor_action_max_abs_difference_vs_control_by_arm": first_action_match_errors,
            "note": "Evaluator seeds Torch and NumPy but not Python stdlib random; post-reset state and first actor actions match, while the stdlib RNG fingerprint varies across fresh processes.",
        },
        "treatment_summary": {
            arm: {key: value for key, value in values.items()
                  if key not in ("per_case",)}
            for arm, values in treatment_results.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
