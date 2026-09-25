"""Analyze the paired AeroWall Launch Tanh independent development replication."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility"
PREFIX = "aerowall-tanh-launch-independent-replication-v1"
BANK = ROOT / "configs/wall_cases/devrep-aerowall-tanh-launch-v1-seed-260925.json"
BANK_SHA = "b6ebf13209ba42b1ad77eb452248b6f743c0f0fbcc32805d250fbc162b3804da"
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


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load_arm(name):
    stem = ART / (PREFIX + "-" + name + "-s260925")
    report_path = stem.with_suffix(".json")
    report = json.loads(report_path.read_text())
    require(report.get("status") == "passed", name + " report did not pass")
    require(report.get("seed") == 260925, name + " seed mismatch")
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
    artefacts = {}
    for suffix in (".json", ".trajectory.npz", ".events.jsonl", ".contacts.json"):
        path = stem.with_suffix(suffix)
        require(path.is_file(), name + " missing artifact " + suffix)
        artefacts[str(path.relative_to(ROOT))] = digest(path)
    report["artifact_hashes"] = artefacts
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
    require(control["initial_case_bank_sha256"] == tanh["initial_case_bank_sha256"], "bank mismatch")
    require(control["launch_action_distribution"] == "default", "control Launch mapping mismatch")
    require(tanh["launch_action_distribution"] == "tanh", "treatment Launch mapping mismatch")
    for arm, report in (("control", control), ("tanh", tanh)):
        require(report["hit_action_distribution"] == "default", arm + " Hit mapping changed")
        require(report["recovery_action_distribution"] == "default", arm + " Recover mapping changed")
        require(report["observation_version"] == "relative_v3", arm + " observation mismatch")
        require(report["reward_design"] == "legacy", arm + " reward mismatch")
    require(tanh.get("action_mapping_experiment") == "AeroWallLaunchTanhIndependentReplicationV1",
            "treatment experiment identity mismatch")
    require(tanh.get("action_mapping_ablation_roles") == ["launch"], "treatment changed wrong roles")
    c_rep = control["reproducibility"]
    t_rep = tanh["reproducibility"]
    c_post = c_rep["post_reset_state"]
    t_post = t_rep["post_reset_state"]
    state_match = c_post["state_sha256"] == t_post["state_sha256"]
    rng_match = c_post["rng_state_sha256"] == t_post["rng_state_sha256"]
    fields_match = c_post["fields_by_env"] == t_post["fields_by_env"]
    c_trace = control["initial_policy_input_trace"]
    t_trace = tanh["initial_policy_input_trace"]
    observation_match = (
        c_trace["actor_observation_views_by_role"] == t_trace["actor_observation_views_by_role"]
        and c_trace["observation_versions_by_role"] == t_trace["observation_versions_by_role"]
    )
    require(state_match and rng_match and fields_match and observation_match,
            "paired reset, RNG, or initial observations differ")
    for left, right in (("env_sha256", "environment"), ("reward_logic_sha256", "reward logic")):
        require(control[left] == tanh[left], "environment/reward code mismatch: " + right)
    require(c_rep["source_files"] == t_rep["source_files"], "source code hashes differ across arms")

    predicates = {
        "legal_first_contact_proxy_caps_ge_1": lambda x: int(x["caps"]) >= 1,
        "legal_second_hit_proxy_caps_ge_2": lambda x: int(x["caps"]) >= 2,
        "three_rallies": lambda x: int(x["rallies"]) >= 3,
        "five_rallies": lambda x: int(x["rallies"]) >= 5,
        "safety_failure": lambda x: int(x["failure"]) in SAFETY_CODES,
        "out_of_bounds": lambda x: int(x["failure"]) == 3,
    }
    paired = {name: count_pair(c, t, predicate) for name, predicate in predicates.items()}
    rally_delta = {case: int(t[case]["rallies"]) - int(c[case]["rallies"]) for case in c}
    rally_changes = {
        "improved_cases": sorted(case for case, delta in rally_delta.items() if delta > 0),
        "unchanged_cases": sorted(case for case, delta in rally_delta.items() if delta == 0),
        "worsened_cases": sorted(case for case, delta in rally_delta.items() if delta < 0),
    }
    audits = {}
    for name, report in (("control", control), ("tanh", tanh)):
        audit = report["contact_audit"]
        require(not audit["callback_errors"], name + " contact callback errors")
        require(audit["legal_events"] == audit["corroborated_events"], name + " legal contact audit mismatch")
        require(audit["wall_events"] == audit["corroborated_wall_events"], name + " wall contact audit mismatch")
        audits[name] = audit
    c_clip = control["actuator_trace_summary"]
    t_clip = tanh["actuator_trace_summary"]
    protocol_passed = (
        paired["legal_second_hit_proxy_caps_ge_2"]["delta_tanh_minus_control"] > 0
        and paired["three_rallies"]["delta_tanh_minus_control"] > 0
        and paired["safety_failure"]["delta_tanh_minus_control"] <= 0
    )
    analysis = {
        "schema_version": 1,
        "experiment_name": "AeroWallLaunchTanhIndependentReplicationV1",
        "status": "complete",
        "scope": "paired 128-case development replication; not final test or P2 acceptance",
        "seed": 260925,
        "case_bank": {
            "path": str(BANK.relative_to(ROOT)),
            "sha256": BANK_SHA,
            "count": 128,
            "kind": "development_replication",
        },
        "checkpoints_sha256": EXPECTED_CHECKPOINTS,
        "reproducibility": {
            "post_reset_state_sha256": c_post["state_sha256"],
            "post_reset_state_matches": state_match,
            "all_rng_state_fingerprints_match": rng_match,
            "rng_state_sha256": c_post["rng_state_sha256"],
            "post_reset_fields_match": fields_match,
            "initial_observations_match": observation_match,
            "source_files_match": True,
            "environment_and_reward_hashes_match": True,
        },
        "arm_reports": {
            "control": {"path": str((ART / (PREFIX + "-control-s260925.json")).relative_to(ROOT)),
                        "sha256": digest(ART / (PREFIX + "-control-s260925.json")),
                        "summary": control["summary"],
                        "actuator_trace_summary": c_clip,
                        "contact_audit": audits["control"],
                        "artifacts": control["artifact_hashes"]},
            "tanh": {"path": str((ART / (PREFIX + "-tanh-s260925.json")).relative_to(ROOT)),
                     "sha256": digest(ART / (PREFIX + "-tanh-s260925.json")),
                     "summary": tanh["summary"],
                     "actuator_trace_summary": t_clip,
                     "contact_audit": audits["tanh"],
                     "artifacts": tanh["artifact_hashes"]},
        },
        "paired_case_metrics": paired,
        "rally_case_changes": rally_changes,
        "decision_rule": {
            "requires_higher_legal_second_hits_and_three_rallies_and_no_increase_in_safety_failures": True,
            "passed": protocol_passed,
            "decision": "replication_rejected_for_task_level_candidate_use" if not protocol_passed else "development_signal_only",
            "training_authorized_by_this_replication": False,
            "promotion_authorized": False,
            "formal_c350_changed": False,
            "p2_passed": False,
        },
        "interpretation": (
            "Tanh mapping lowered raw actuator clipping and raised first-contact count, "
            "but legal second hits fell, three-rally count did not improve, and safety "
            "failures increased. This independent development replication rejects a "
            "task-level Tanh Launch signal for candidate use; it does not pass P2."
        ),
    }
    out = ART / (PREFIX + "-analysis-s260925.json")
    encoded = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text() != encoded:
        raise SystemExit("refusing to overwrite a different analysis result")
    out.write_text(encoded)
    print(json.dumps({"analysis": str(out.relative_to(ROOT)), "sha256": digest(out),
                      "paired_state_rng_observation_match": True,
                      "protocol_passed": protocol_passed,
                      "metrics": {name: {"control": value["control"],
                                         "tanh": value["tanh"],
                                         "delta": value["delta_tanh_minus_control"]}
                                  for name, value in paired.items()},
                      "clip_total": [c_clip["fraction_action_values_limited"],
                                     t_clip["fraction_action_values_limited"]]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
