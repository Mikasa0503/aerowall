"""Analyze the paired V6 same-bank AeroWall Launch Tanh mapping replication."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility"
PREFIX = "aerowall-v6-samebank-launch-tanh-replication-v1"
BANK = ROOT / "configs/wall_cases/heldout-128-v4.json"
BANK_SHA = "666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80"
EXPECTED_CHECKPOINTS = {
    "launch": "165fd2a0eabf43082df62a1f5d458cc4f528f51d02cffa389f32e8bd6671dd8b",
    "hit": "4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659",
    "recovery": "4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659",
}
FAILURE_NAMES = {
    0: "none", 1: "ball_ground", 2: "drone_ground",
    3: "out_of_bounds", 4: "illegal_contact", 5: "drone_wall",
}
SAFETY_CODES = {2, 4, 5}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load_arm(name):
    stem = ART / (PREFIX + "-" + name + "-s9524")
    report_path = stem.with_suffix(".json")
    report = json.loads(report_path.read_text())
    require(report.get("status") == "passed", name + " report did not pass")
    require(report.get("seed") == 9524, name + " seed mismatch")
    require(report.get("num_envs") == 128, name + " env count mismatch")
    require(report.get("initial_case_bank_sha256") == BANK_SHA, name + " bank hash mismatch")
    require(report.get("summary", {}).get("episodes") == 128, name + " episode count mismatch")
    require(report.get("summary", {}).get("all_outcomes_contact_audited") is True,
            name + " outcomes are not all contact-audited")
    inputs = report.get("reproducibility", {}).get("input_files", {})
    for role, expected in EXPECTED_CHECKPOINTS.items():
        key = "launch_checkpoint" if role == "launch" else role + "_checkpoint"
        require(inputs.get(key, {}).get("sha256") == expected, name + " " + role + " checkpoint mismatch")
    outcomes = {row["case_id"]: row for row in report.get("outcomes", [])}
    require(len(outcomes) == 128, name + " outcome count mismatch")
    require(len(outcomes) == len(report.get("outcomes", [])), name + " duplicate case id")
    failure_counts = {label: 0 for label in FAILURE_NAMES.values()}
    for row in outcomes.values():
        failure_counts[FAILURE_NAMES[int(row["failure"])]] += 1
    require(failure_counts == report["summary"]["failure_counts"], name + " failure count mismatch")
    artifacts = {}
    for suffix in (".json", ".trajectory.npz", ".events.jsonl", ".contacts.json"):
        path = stem.with_suffix(suffix)
        require(path.is_file(), name + " missing artifact " + suffix)
        artifacts[str(path.relative_to(ROOT))] = digest(path)
    report["artifact_hashes"] = artifacts
    return report, outcomes


def count_pair(control, treatment, predicate):
    c_yes = {case for case, row in control.items() if predicate(row)}
    t_yes = {case for case, row in treatment.items() if predicate(row)}
    return {
        "control": len(c_yes),
        "tanh": len(t_yes),
        "delta_tanh_minus_control": len(t_yes) - len(c_yes),
        "both": len(c_yes & t_yes),
        "control_only": len(c_yes - t_yes),
        "tanh_only": len(t_yes - c_yes),
        "neither": 128 - len(c_yes | t_yes),
        "control_only_case_ids": sorted(c_yes - t_yes),
        "tanh_only_case_ids": sorted(t_yes - c_yes),
    }


def main():
    require(BANK.is_file() and digest(BANK) == BANK_SHA, "development bank hash mismatch")
    control, c = load_arm("control")
    tanh, t = load_arm("tanh")
    require(set(c) == set(t), "case IDs differ across arms")
    require(control["launch_action_distribution"] == "default", "control Launch mapping mismatch")
    require(tanh["launch_action_distribution"] == "tanh", "treatment Launch mapping mismatch")
    for arm, report in (("control", control), ("tanh", tanh)):
        require(report["hit_action_distribution"] == "default", arm + " Hit mapping changed")
        require(report["recovery_action_distribution"] == "default", arm + " Recover mapping changed")
        require(report["evaluation_observation_versions_by_role"] ==
                {"launch": "relative_v3", "hit": "relative_v3", "recovery": "relative_v3"},
                arm + " role observation mismatch")
        require(report["reward_design"] == "legacy", arm + " reward mismatch")
    require(tanh.get("action_mapping_experiment") == "AeroWallLaunchTanhSameBankReplicationV1",
            "treatment experiment identity mismatch")
    require(tanh.get("action_mapping_ablation_roles") == ["launch"], "treatment changed wrong roles")

    c_rep, t_rep = control["reproducibility"], tanh["reproducibility"]
    c_post, t_post = c_rep["post_reset_state"], t_rep["post_reset_state"]
    state_match = c_post["state_sha256"] == t_post["state_sha256"]
    rng_match = c_post["rng_state_sha256"] == t_post["rng_state_sha256"]
    fields_match = c_post["fields_by_env"] == t_post["fields_by_env"]
    c_trace, t_trace = control["initial_policy_input_trace"], tanh["initial_policy_input_trace"]
    observation_match = (
        c_trace["actor_observation_views_by_role"] == t_trace["actor_observation_views_by_role"]
        and c_trace["observation_versions_by_role"] == t_trace["observation_versions_by_role"]
    )
    source_match = c_rep["source_files"] == t_rep["source_files"]
    env_match = control["env_sha256"] == tanh["env_sha256"]
    reward_match = control["reward_logic_sha256"] == tanh["reward_logic_sha256"]
    require(all((state_match, rng_match, fields_match, observation_match, source_match, env_match, reward_match)),
            "paired reset, RNG, observation, environment, reward, or source checks differ")

    audits = {}
    for name, report in (("control", control), ("tanh", tanh)):
        audit = report["contact_audit"]
        require(not audit["callback_errors"], name + " contact callback errors")
        require(audit["legal_events"] == audit["corroborated_events"], name + " legal contact audit mismatch")
        require(audit["wall_events"] == audit["corroborated_wall_events"], name + " wall contact audit mismatch")
        audits[name] = audit

    predicates = {
        "legal_first_contact_caps_ge_1": lambda row: int(row["caps"]) >= 1,
        "legal_second_hit_caps_ge_2": lambda row: int(row["caps"]) >= 2,
        "three_rallies": lambda row: int(row["rallies"]) >= 3,
        "five_rallies": lambda row: int(row["rallies"]) >= 5,
        "safety_failure": lambda row: int(row["failure"]) in SAFETY_CODES,
        "illegal_contact": lambda row: int(row["failure"]) == 4,
        "out_of_bounds": lambda row: int(row["failure"]) == 3,
    }
    paired = {name: count_pair(c, t, predicate) for name, predicate in predicates.items()}
    rally_delta = {case: int(t[case]["rallies"]) - int(c[case]["rallies"]) for case in c}
    rally_changes = {
        "improved_cases": sorted(case for case, delta in rally_delta.items() if delta > 0),
        "unchanged_cases": sorted(case for case, delta in rally_delta.items() if delta == 0),
        "worsened_cases": sorted(case for case, delta in rally_delta.items() if delta < 0),
    }

    c_clip, t_clip = control["actuator_trace_summary"], tanh["actuator_trace_summary"]
    task_signal = (
        paired["legal_second_hit_caps_ge_2"]["delta_tanh_minus_control"] > 0
        and paired["three_rallies"]["delta_tanh_minus_control"] > 0
        and paired["safety_failure"]["delta_tanh_minus_control"] <= 0
    )
    analysis = {
        "schema_version": 1,
        "experiment_name": "AeroWallLaunchTanhSameBankReplicationV1",
        "status": "complete",
        "scope": "paired same-bank 128-case development replication; not final test or P2 acceptance",
        "seed": 9524,
        "case_bank": {"path": str(BANK.relative_to(ROOT)), "sha256": BANK_SHA, "count": 128, "kind": "development"},
        "checkpoints_sha256": EXPECTED_CHECKPOINTS,
        "reproducibility": {
            "post_reset_state_sha256": c_post["state_sha256"],
            "post_reset_state_matches": state_match,
            "all_rng_state_fingerprints_match": rng_match,
            "rng_state_sha256": c_post["rng_state_sha256"],
            "post_reset_fields_match": fields_match,
            "initial_actor_observations_match": observation_match,
            "source_files_match": source_match,
            "environment_hashes_match": env_match,
            "reward_logic_hashes_match": reward_match,
        },
        "arm_reports": {
            "control": {
                "path": str((ART / (PREFIX + "-control-s9524.json")).relative_to(ROOT)),
                "sha256": digest(ART / (PREFIX + "-control-s9524.json")),
                "summary": control["summary"],
                "actuator_trace_summary": c_clip,
                "contact_audit": audits["control"],
                "artifacts": control["artifact_hashes"],
            },
            "tanh": {
                "path": str((ART / (PREFIX + "-tanh-s9524.json")).relative_to(ROOT)),
                "sha256": digest(ART / (PREFIX + "-tanh-s9524.json")),
                "summary": tanh["summary"],
                "actuator_trace_summary": t_clip,
                "contact_audit": audits["tanh"],
                "artifacts": tanh["artifact_hashes"],
            },
        },
        "paired_case_metrics": paired,
        "rally_case_changes": rally_changes,
        "decision_rule": {
            "requires_more_legal_second_hits_and_three_rallies_and_no_more_safety_failures": True,
            "passed": task_signal,
            "decision": "development_signal_only" if task_signal else "no_task_level_replication_signal",
            "additional_training_from_this_replication": False,
            "promotion_authorized": False,
            "formal_c350_changed": False,
            "p2_passed": False,
        },
        "interpretation": (
            "Launch Tanh reduced actuator clipping, increased legal first contacts, and reduced total safety failures. "
            "Legal second hits were unchanged and neither arm reached three rallies, so the preregistered task-level "
            "continuation rule did not pass. This does not justify another training run or promotion."
        ),
    }
    out = ART / (PREFIX + "-analysis-s9524.json")
    encoded = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text() != encoded:
        raise SystemExit("refusing to overwrite a different analysis result")
    out.write_text(encoded)
    print(json.dumps({
        "analysis": str(out.relative_to(ROOT)),
        "analysis_sha256": digest(out),
        "reproducibility_checks_passed": True,
        "task_signal": task_signal,
        "metrics": {
            name: {"control": value["control"], "tanh": value["tanh"],
                   "delta": value["delta_tanh_minus_control"]}
            for name, value in paired.items()
        },
        "clip_total": [c_clip["fraction_action_values_limited"], t_clip["fraction_action_values_limited"]],
        "physx_contact_audit": {
            arm: {"legal_events": audit["legal_events"], "corroborated_events": audit["corroborated_events"],
                  "wall_events": audit["wall_events"], "corroborated_wall_events": audit["corroborated_wall_events"]}
            for arm, audit in audits.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
