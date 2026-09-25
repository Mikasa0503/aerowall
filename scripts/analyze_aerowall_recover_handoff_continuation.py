"""Create a hash-backed paired summary for Recover handoff continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility"
BANK = REPRO / "aerowall-recover-handoff-bank-v1-s9524.json"
RESET = REPRO / "aerowall-recover-handoff-reset-audit-v1-s9524.json"
CONTROL = REPRO / "aerowall-recover-handoff-continuation-control-v1-s9524.json"
TANH = REPRO / "aerowall-recover-handoff-continuation-tanh-v1-s9524.json"
OUTPUT = REPRO / "aerowall-recover-handoff-continuation-comparison-v1-s9524.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_outcomes(report: dict, name: str) -> dict[str, dict]:
    rows = report.get("outcomes", [])
    result = {row["case_id"]: row for row in rows}
    require(len(result) == len(rows), f"{name} has duplicate case_id values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()

    bank = load(BANK)
    reset = load(RESET)
    control = load(CONTROL)
    tanh = load(TANH)
    cases = {case["case_id"]: case for case in bank["cases"]}
    require(len(cases) == len(bank["cases"]), "Recover handoff bank has duplicate case_id values")
    require(bank.get("bank_kind") == "aerowall_recover_handoff_v1", "unexpected handoff bank kind")
    require(bank.get("handoff_actor_role") == "recover", "handoff bank does not identify Recover")
    require(all((case.get("handoff_state_version"), case.get("handoff_actor_role"),
                 case.get("skill_id"), case.get("phase")) == (2, "recover", 2, 2)
                for case in cases.values()), "bank contains a non-Recover handoff state")

    bank_hash = sha256(BANK)
    reset_audit = reset.get("handoff_reset_audit", {})
    require(reset.get("status") == "passed", "Recover reset audit did not pass")
    require(reset.get("initial_case_bank_sha256") == bank_hash, "reset audit used a different bank")
    require(reset_audit.get("passed") is True and reset_audit.get("case_count") == len(cases),
            "reset audit does not cover every Recover handoff case")
    require(reset_audit.get("field_count") == 20, "unexpected reset audit field count")
    require(reset_audit.get("absolute_tolerance") == 1e-4, "unexpected reset audit tolerance")
    require(max(reset_audit["max_abs_error_by_field"].values()) <= 1e-4,
            "Recover reset audit exceeds its declared tolerance")

    for name, report in (("default Recover", control), ("Tanh Recover", tanh)):
        require(report.get("status") == "passed", f"{name} continuation did not pass")
        require(report.get("handoff_evaluation") is True, f"{name} is not a conditional handoff run")
        require(report.get("handoff_actor_role") == "recover", f"{name} used a different handoff role")
        require(report.get("initial_case_bank_sha256") == bank_hash,
                f"{name} used a different Recover handoff bank")
        summary = report.get("summary", {})
        audit = report.get("contact_audit", {})
        require(summary.get("all_outcomes_contact_audited") is True,
                f"{name} outcomes are not fully contact-audited")
        require(not audit.get("callback_errors"), f"{name} has contact callback errors")
        require(audit.get("legal_events") == audit.get("corroborated_events"),
                f"{name} legal contacts lack PhysX corroboration")
        require(audit.get("wall_events") == audit.get("corroborated_wall_events"),
                f"{name} wall events lack PhysX corroboration")
        require(audit.get("inherited_source_prefix_physx_audited") is True,
                f"{name} inherited prefix is not PhysX-audited")

    require(control.get("recovery_action_distribution") == "default",
            "control continuation did not use the default Recover mapping")
    require(tanh.get("recovery_action_distribution") == "tanh",
            "candidate continuation did not use the Tanh Recover mapping")
    require(tanh.get("action_mapping_experiment") == "AeroWallRecoverTanhMappingAblationV1",
            "Tanh Recover candidate identity changed")
    require(tanh.get("action_mapping_ablation_roles") == ["recovery"],
            "Tanh Recover run changed more than the Recover role")
    shared_route_fields = (
        "seed", "num_envs", "launch_checkpoint_sha256", "hit_checkpoint_sha256",
        "recovery_checkpoint_sha256", "launch_action_distribution",
        "hit_action_distribution", "reward_design", "observation_version",
        "skill_observation_version", "evaluation_observation_versions_by_role",
        "hit_window", "hit_exit_window", "initial_case_bank_sha256",
    )
    for field in shared_route_fields:
        require(control.get(field) == tanh.get(field),
                f"paired continuations differ in shared route field {field}")
    require(control.get("seed") == 9524 and control.get("num_envs") == len(cases),
            "unexpected seed or case count in paired continuations")

    control_rows = load_outcomes(control, "control continuation")
    tanh_rows = load_outcomes(tanh, "Tanh continuation")
    require(set(cases) == set(control_rows) == set(tanh_rows),
            "paired continuations do not cover the exact same cases")

    rally_delta_bins: Counter[str] = Counter()
    rally_transitions: Counter[str] = Counter()
    safety_transitions: Counter[str] = Counter()
    failure_code_transitions: Counter[str] = Counter()
    continued_cap = {"control": 0, "tanh": 0}
    improved_cases: list[str] = []
    regressed_cases: list[str] = []
    safety_became_true: list[str] = []
    safety_became_false: list[str] = []
    paired_rows = []

    def safety_failure(row: dict) -> bool:
        # Match the project safety tally: drone ground, illegal contact, or drone wall.
        return int(row["failure"]) in (2, 4, 5)

    for case_id in sorted(cases):
        case = cases[case_id]
        control_row = control_rows[case_id]
        tanh_row = tanh_rows[case_id]
        control_caps = int(control_row["caps"]) - int(case["caps"])
        tanh_caps = int(tanh_row["caps"]) - int(case["caps"])
        control_rallies = int(control_row["rallies"]) - int(case["rallies"])
        tanh_rallies = int(tanh_row["rallies"]) - int(case["rallies"])
        require(control_caps >= 0 and tanh_caps >= 0, f"negative post-handoff cap count for {case_id}")
        require(control_rallies >= 0 and tanh_rallies >= 0,
                f"negative post-handoff rally count for {case_id}")
        rally_delta = tanh_rallies - control_rallies
        rally_delta_bins[str(rally_delta)] += 1
        rally_transitions["tanh_higher" if rally_delta > 0 else
                          "equal" if rally_delta == 0 else "tanh_lower"] += 1
        if rally_delta > 0:
            improved_cases.append(case_id)
        elif rally_delta < 0:
            regressed_cases.append(case_id)
        continued_cap["control"] += control_caps > 0
        continued_cap["tanh"] += tanh_caps > 0
        control_safety = safety_failure(control_row)
        tanh_safety = safety_failure(tanh_row)
        key = f"control_{'safety_failure' if control_safety else 'no_safety_failure'}__tanh_{'safety_failure' if tanh_safety else 'no_safety_failure'}"
        safety_transitions[key] += 1
        if not control_safety and tanh_safety:
            safety_became_true.append(case_id)
        elif control_safety and not tanh_safety:
            safety_became_false.append(case_id)
        failure_code_transitions[f"{int(control_row['failure'])}_to_{int(tanh_row['failure'])}"] += 1
        paired_rows.append({
            "case_id": case_id,
            "control_rallies_after_handoff": control_rallies,
            "tanh_rallies_after_handoff": tanh_rallies,
            "tanh_minus_control_rallies": rally_delta,
            "control_failure_code": int(control_row["failure"]),
            "tanh_failure_code": int(tanh_row["failure"]),
        })

    n = len(cases)
    cmean = sum(row["control_rallies_after_handoff"] for row in paired_rows) / n
    tmean = sum(row["tanh_rallies_after_handoff"] for row in paired_rows) / n
    source_audit = bank.get("source_contact_audit", {})
    expected_source_ok = (
        not source_audit.get("callback_errors")
        and source_audit.get("legal_cap_events") == source_audit.get("corroborated_cap_events")
        and source_audit.get("wall_events") == source_audit.get("corroborated_wall_events")
    )
    require(expected_source_ok, "source rollout contact audit is incomplete")

    input_paths = {
        "handoff_source_report": REPRO / "aerowall-recover-handoff-source-v1-s9524.json",
        "handoff_bank": BANK,
        "reset_audit_report": RESET,
        "default_recover_continuation_report": CONTROL,
        "tanh_recover_continuation_report": TANH,
        "source_event_log": Path(bank["source_event_log"]),
        "source_contact_report": Path(bank["source_contact_report"]),
        "control_event_log": CONTROL.with_suffix(".events.jsonl"),
        "control_contact_report": CONTROL.with_suffix(".contacts.json"),
        "tanh_event_log": TANH.with_suffix(".events.jsonl"),
        "tanh_contact_report": TANH.with_suffix(".contacts.json"),
        "evaluator": ROOT / "scripts/evaluate_aerowall_wall_rl.py",
        "policy_wrapper": ROOT / "scripts/aerowall_skill_policies.py",
        "rally_environment": ROOT / "scripts/aerowall_wall_rally_env.py",
        "handoff_validator": ROOT / "aerowall/wall_rl/curriculum.py",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    require(not missing, f"missing hash input artifacts: {missing}")
    result = {
        "analysis_name": "AeroWallRecoverHandoffContinuationComparisonV1",
        "status": "passed",
        "seed": 9524,
        "handoff_actor_role": "recover",
        "diagnostic_only": True,
        "training_performed": False,
        "candidate_promotion": False,
        "protocol": "paired conditional continuation from the same audited Recover handoff bank",
        "handoff_case_count": n,
        "hcsp_commit": control.get("hcsp_commit"),
        "single_variable_route_match": list(shared_route_fields),
        "source_initial_case_bank_sha256": bank.get("source_initial_case_bank_sha256"),
        "source_contact_audit": source_audit,
        "reset_audit": {
            "passed": True,
            "field_count": reset_audit["field_count"],
            "absolute_tolerance": reset_audit["absolute_tolerance"],
            "max_error": max(reset_audit["max_abs_error_by_field"].values()),
            "case_count": reset_audit["case_count"],
        },
        "conditions": {
            "control": "AeroWall frozen route with HCSP default Recover IndependentNormal mapping",
            "candidate": tanh.get("action_mapping_experiment"),
            "changed_role": "recovery",
            "launch_action_distribution": control.get("launch_action_distribution"),
            "hit_action_distribution": control.get("hit_action_distribution"),
            "reward_design": control.get("reward_design"),
            "observation_versions_by_role": control.get("evaluation_observation_versions_by_role"),
            "checkpoint_sha256_by_role": {
                "launch": control.get("launch_checkpoint_sha256"),
                "hit": control.get("hit_checkpoint_sha256"),
                "recovery": control.get("recovery_checkpoint_sha256"),
            },
            "same_initial_bank_sha256": bank_hash,
        },
        "paired_outcomes": {
            "case_count": n,
            "first_legal_hit_after_handoff": continued_cap,
            "mean_rallies_after_handoff": {"control": cmean, "tanh": tmean},
            "three_rally_rate": {
                "control": control["summary"]["handoff_conditional"]["three_rallies_after_handoff_rate"],
                "tanh": tanh["summary"]["handoff_conditional"]["three_rallies_after_handoff_rate"],
            },
            "five_rally_rate": {
                "control": control["summary"]["handoff_conditional"]["five_rallies_after_handoff_rate"],
                "tanh": tanh["summary"]["handoff_conditional"]["five_rallies_after_handoff_rate"],
            },
            "tanh_minus_control_rallies": {
                "higher": rally_transitions["tanh_higher"],
                "equal": rally_transitions["equal"],
                "lower": rally_transitions["tanh_lower"],
                "delta_bins": dict(sorted(rally_delta_bins.items(), key=lambda item: int(item[0]))),
                "higher_case_ids": improved_cases,
                "lower_case_ids": regressed_cases,
            },
            "safety_failure_cases": {
                "codes": [2, 4, 5],
                "control_count": sum(safety_failure(row) for row in control_rows.values()),
                "tanh_count": sum(safety_failure(row) for row in tanh_rows.values()),
                "paired_transitions": dict(safety_transitions),
                "new_cases": safety_became_true,
                "resolved_cases": safety_became_false,
            },
            "failure_code_transitions": dict(sorted(failure_code_transitions.items())),
            "contact_audit": {
                "control_new_cap_events": control["contact_audit"]["legal_events"],
                "control_new_wall_events": control["contact_audit"]["wall_events"],
                "tanh_new_cap_events": tanh["contact_audit"]["legal_events"],
                "tanh_new_wall_events": tanh["contact_audit"]["wall_events"],
                "inherited_source_prefix_events_per_condition": control["contact_audit"]["inherited_source_prefix_events"],
                "all_new_events_physx_corroborated": True,
                "callback_errors": [],
            },
        },
        "limitations": [
            "Cases are conditioned on reaching Recover under the default C350 source route.",
            "The source states come from the v4 development challenge bank, not an independent frozen final test set.",
            "These conditional continuation outcomes do not replace the natural full-chain P2 gate.",
        ],
        "source_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "case_outcomes": paired_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "output": str(output), "sha256": sha256(output),
                      "cases": n, "tanh_minus_control_rallies": dict(rally_transitions),
                      "safety_failures": {"control": result["paired_outcomes"]["safety_failure_cases"]["control_count"],
                                          "tanh": result["paired_outcomes"]["safety_failure_cases"]["tanh_count"]}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
