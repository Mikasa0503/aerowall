"""Audit the preregistered AeroWall Launch lateral-momentum screen."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-launch-lateral-momentum-counterfactual-v1-analysis-s9524.json"
PREFIX = "aerowall-launch-lateral-momentum-counterfactual-v1"
ARTIFACTS = ROOT / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility"
EXPECTED_PAIRS = (
    ("heldout-0008", "heldout-0095"), ("heldout-0012", "heldout-0100"),
    ("heldout-0018", "heldout-0004"), ("heldout-0033", "heldout-0096"),
    ("heldout-0051", "heldout-0029"), ("heldout-0059", "heldout-0075"),
    ("heldout-0073", "heldout-0070"), ("heldout-0082", "heldout-0037"),
    ("heldout-0099", "heldout-0068"), ("heldout-0107", "heldout-0124"),
    ("heldout-0112", "heldout-0103"), ("heldout-0113", "heldout-0058"),
)
EXPECTED_INITIAL_BANK_SHA256 = "666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80"
EXPECTED_SOURCE_TRAJECTORY_SHA256 = "bf0f2aee9e7f065acad26b2121680487995ad01dfb58702ccc6e41dca1401c06"
SAFETY_FAILURE_CODES = {2, 4, 5}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def report_paths(arm):
    return {
        "report": ARTIFACTS / f"{PREFIX}-{arm}-continuation-s9524.json",
        "reset_audit": ARTIFACTS / f"{PREFIX}-{arm}-reset-audit-s9524.json",
        "bank": ARTIFACTS / f"{PREFIX}-{arm}-s9524.json",
    }


def first_body_events(report):
    path = Path(report["first_episode_events"])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    first = {}
    for row in rows:
        if not row.get("body"):
            continue
        env = int(row["env"])
        marker = (int(row["policy_step"]), int(row["substep"]))
        if env not in first or marker < (int(first[env]["policy_step"]), int(first[env]["substep"])):
            first[env] = row
    return first, path


def physx_audit_first_events(report, first):
    trajectory = Path(report["trajectory"])
    sidecar = trajectory.with_name(trajectory.name.replace(".trajectory.npz", ".contacts.json"))
    contacts = read_json(sidecar)
    result = {}
    for env, event in first.items():
        base = f"/World/envs/env_{env}/IrisTest_0/base_link"
        ball = f"/World/envs/env_{env}/ball"
        result[env] = any(
            int(contact["step"]) == int(event["policy_step"])
            and int(contact["substep"]) == int(event["substep"])
            and int(contact["contact_count"]) > 0
            and base in contact["actors"] and ball in contact["actors"]
            for contact in contacts
        )
    return result, sidecar


def main():
    prereg = ROOT / "docs/plans/2026-09-25-aerowall-launch-lateral-momentum-counterfactual-v1.md"
    source_report_path = ARTIFACTS / f"{PREFIX}-source-s9524.json"
    source_bank_path = ARTIFACTS / f"{PREFIX}-source-handoffs-s9524.json"
    pair_report_path = ARTIFACTS / "aerowall-fullbank-contact-approach-alignment-v1-s9524.json"
    control = report_paths("control")
    intervention = report_paths("intervention")
    inputs = [prereg, source_report_path, source_bank_path, pair_report_path,
              control["report"], control["reset_audit"], control["bank"],
              intervention["report"], intervention["reset_audit"], intervention["bank"]]
    input_hashes = {relative(path): digest(path) for path in inputs}

    source_report = read_json(source_report_path)
    source_bank = read_json(source_bank_path)
    pair_report = read_json(pair_report_path)
    assert source_report["status"] == "passed"
    assert source_report["summary"]["all_outcomes_contact_audited"]
    assert source_report["initial_case_bank_sha256"] == EXPECTED_INITIAL_BANK_SHA256
    assert source_bank["bank_kind"] == "aerowall_launch_handoff_v1"
    assert source_bank["handoff_actor_role"] == "launch"
    assert len(source_bank["cases"]) == 24
    assert not source_bank["source_contact_audit"]["callback_errors"]
    assert source_bank["source_contact_audit"]["legal_cap_events"] == source_bank["source_contact_audit"]["corroborated_cap_events"]
    assert source_bank["source_contact_audit"]["wall_events"] == source_bank["source_contact_audit"]["corroborated_wall_events"]
    source_by_id = {case["source_case_id"]: case for case in source_bank["cases"]}
    assert len(source_by_id) == 24
    expected_ids = {case_id for pair in EXPECTED_PAIRS for case_id in pair}
    assert set(source_by_id) == expected_ids
    source_pairs = pair_report["control_phase0_failure_vs_legal_matched_approach"]["pairs"]
    assert tuple((r["illegal_case_id"], r["matched_legal_case_id"]) for r in source_pairs) == EXPECTED_PAIRS
    source_trajectory = Path(source_report["trajectory"])
    source_trajectory_sha = digest(source_trajectory)
    assert source_trajectory_sha == EXPECTED_SOURCE_TRAJECTORY_SHA256

    arms = {}
    for name, paths in (("control", control), ("intervention", intervention)):
        report = read_json(paths["report"])
        reset = read_json(paths["reset_audit"])
        bank = read_json(paths["bank"])
        assert report["status"] == "passed"
        assert report["summary"]["all_outcomes_contact_audited"]
        audit = report["contact_audit"]
        assert not audit["callback_errors"]
        assert audit["legal_events"] == audit["corroborated_events"]
        assert audit["wall_events"] == audit["corroborated_wall_events"]
        ra = reset["handoff_reset_audit"]
        assert reset["status"] == "passed" and ra["passed"]
        assert ra["case_count"] == 12 and ra["field_count"] == 20
        assert max(ra["max_abs_error_by_field"].values()) <= ra["absolute_tolerance"] <= 1e-4
        cases = bank["cases"]
        assert len(cases) == 12
        assert tuple((cases[i]["source_case_id"], EXPECTED_PAIRS[i][0]) for i in range(12)) == tuple((bad, bad) for bad, _ in EXPECTED_PAIRS)
        first, event_path = first_body_events(report)
        physx, contacts_path = physx_audit_first_events(report, first)
        assert set(first) == set(range(12))
        assert all(physx.values())
        assert len(report["outcomes"]) == 12
        arms[name] = {"paths": paths, "report": report, "reset": reset, "bank": bank,
                      "first": first, "physx": physx, "event_path": event_path,
                      "contacts_path": contacts_path}

    control_cases = arms["control"]["bank"]["cases"]
    intervention_cases = arms["intervention"]["bank"]["cases"]
    signed_deltas = []
    for left, right in zip(control_cases, intervention_cases):
        assert left["source_case_id"] == right["source_case_id"]
        changed = {key for key in left if left[key] != right.get(key)}
        assert changed == {"drone_velocity"}
        assert all(left["drone_velocity"][j] == right["drone_velocity"][j] for j in range(6) if j != 1)
        signed_deltas.append(float(left["drone_velocity"][1]) - float(right["drone_velocity"][1]))

    source_action_errors = []
    control_traj = Path(arms["control"]["report"]["trajectory"])
    with np.load(control_traj) as trajectory:
        first_actions = trajectory["action"][0]
        assert first_actions.shape == (12, 4)
        for index, case in enumerate(control_cases):
            expected = source_by_id[case["source_case_id"]]["source_policy_action"]
            source_action_errors.append(float(np.max(np.abs(np.asarray(expected) - first_actions[index]))))
    action_match_max = max(source_action_errors)

    rows = []
    improvements = []
    new_safety = []
    newly_legal = []
    code3_cases = []
    control_legal = 0
    intervention_legal = 0
    for index, pair in enumerate(EXPECTED_PAIRS):
        bad, legal = pair
        c_event = arms["control"]["first"][index]
        i_event = arms["intervention"]["first"][index]
        c_out = arms["control"]["report"]["outcomes"][index]
        i_out = arms["intervention"]["report"]["outcomes"][index]
        assert control_cases[index]["source_case_id"] == bad
        assert intervention_cases[index]["source_case_id"] == bad
        c_legal = bool(c_event["legal_cap"])
        i_legal = bool(i_event["legal_cap"])
        control_legal += int(c_legal)
        intervention_legal += int(i_legal)
        if not c_legal and i_legal:
            newly_legal.append(bad)
        improvement = float(c_event["radial_error"]) - float(i_event["radial_error"])
        improvements.append(improvement)
        if int(i_out["failure"]) in SAFETY_FAILURE_CODES and int(c_out["failure"]) not in SAFETY_FAILURE_CODES:
            new_safety.append(bad)
        if int(i_out["failure"]) == 3 and int(c_out["failure"]) != 3:
            code3_cases.append(bad)
        rows.append({
            "source_failed_case_id": bad,
            "matched_legal_reference_case_id": legal,
            "control_first_contact": {
                "step": int(c_event["policy_step"]), "substep": int(c_event["substep"]),
                "legal": c_legal, "radial_error_m": float(c_event["radial_error"]),
                "physx_audited": bool(arms["control"]["physx"][index]),
            },
            "intervention_first_contact": {
                "step": int(i_event["policy_step"]), "substep": int(i_event["substep"]),
                "legal": i_legal, "radial_error_m": float(i_event["radial_error"]),
                "physx_audited": bool(arms["intervention"]["physx"][index]),
            },
            "paired_radial_improvement_m": improvement,
            "terminal_failure_code": {"control": int(c_out["failure"]), "intervention": int(i_out["failure"])},
        })

    median_improvement = float(np.median(np.asarray(improvements)))
    all_first_contacts_audited = all(
        all(arm["physx"].values()) and len(arm["first"]) == 12 for arm in arms.values()
    )
    no_callback_errors = all(not arm["report"]["contact_audit"]["callback_errors"] for arm in arms.values())
    no_new_safety = not new_safety
    reset_ok = all(
        arm["reset"]["handoff_reset_audit"]["passed"]
        and max(arm["reset"]["handoff_reset_audit"]["max_abs_error_by_field"].values()) <= 1e-4
        for arm in arms.values()
    )
    screen_supported = (
        intervention_legal >= 6 and len(improvements) >= 6 and median_improvement >= 0.02
        and no_new_safety and all_first_contacts_audited and no_callback_errors and reset_ok
    )

    file_hashes = dict(input_hashes)
    for name, arm in arms.items():
        report = arm["report"]
        for kind, path in (("report", arm["paths"]["report"]),
                           ("trajectory", Path(report["trajectory"])),
                           ("events", arm["event_path"]),
                           ("contacts", arm["contacts_path"]),
                           ("reset_audit", arm["paths"]["reset_audit"])):
            file_hashes[relative(path)] = digest(path)
    for path in (source_trajectory,):
        file_hashes[relative(path)] = digest(path)

    result = {
        "schema_version": 1,
        "experiment_name": "AeroWallLaunchLateralMomentumCounterfactualV1",
        "status": "passed" if screen_supported else "screen_not_supported",
        "interpretation": "supports_only_the_preregistered_mechanism_screen_on_12_selected_heldout-128-v4_development_cases",
        "does_not_pass_p2": True,
        "authorizes_training_or_promotion": False,
        "formal_policy_unchanged": "C350",
        "protocol": {
            "seed": 9524,
            "initial_bank": "configs/wall_cases/heldout-128-v4.json",
            "initial_bank_sha256": EXPECTED_INITIAL_BANK_SHA256,
            "pairs": [list(pair) for pair in EXPECTED_PAIRS],
            "intervention": "set only failed-state drone_velocity[1] to its paired legal-state snapshot value",
            "source_trajectory_matches_corrected_control": source_trajectory_sha == EXPECTED_SOURCE_TRAJECTORY_SHA256,
            "control_first_action_max_abs_error_vs_source": action_match_max,
        },
        "metrics": {
            "control_legal_first_contacts": control_legal,
            "intervention_legal_first_contacts": intervention_legal,
            "net_legal_first_contact_gain_vs_control": intervention_legal - control_legal,
            "newly_legal_vs_control_case_ids": newly_legal,
            "paired_first_contact_radial_improvement_m": {
                "n_pairs": len(improvements), "median": median_improvement,
                "min": float(min(improvements)), "max": float(max(improvements)),
                "threshold_m": 0.02,
            },
            "signed_intervened_world_y_velocity_delta_m_s": {
                "min": float(min(signed_deltas)), "median": float(np.median(signed_deltas)),
                "max": float(max(signed_deltas)), "negative_delta_pair_count": sum(x < 0 for x in signed_deltas),
            },
            "failure_counts": {
                arm: arms[arm]["report"]["summary"]["failure_counts"] for arm in ("control", "intervention")
            },
            "new_registered_safety_failure_case_ids": new_safety,
            "new_out_of_bounds_case_ids": code3_cases,
            "first_contact_physx_audit": {
                "control": {"audited": sum(arms["control"]["physx"].values()), "total": 12},
                "intervention": {"audited": sum(arms["intervention"]["physx"].values()), "total": 12},
            },
            "callback_errors": {
                arm: arms[arm]["report"]["contact_audit"]["callback_errors"] for arm in ("control", "intervention")
            },
            "reset_audit": {
                arm: arms[arm]["reset"]["handoff_reset_audit"] for arm in ("control", "intervention")
            },
            "preregistered_screen_criteria": {
                "at_least_six_intervention_legal_first_contacts": intervention_legal >= 6,
                "at_least_six_complete_pairs_and_median_improvement_ge_0p02m": len(improvements) >= 6 and median_improvement >= 0.02,
                "no_new_drone_ground_illegal_or_drone_wall_failures": no_new_safety,
                "all_first_contacts_physx_audited_and_no_callback_errors": all_first_contacts_audited and no_callback_errors,
                "all_reset_fields_within_1e-4": reset_ok,
                "screen_supported": screen_supported,
            },
        },
        "per_case": rows,
        "evaluator_summary": {arm: arms[arm]["report"]["summary"] for arm in ("control", "intervention")},
        "artifact_sha256": file_hashes,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError(f"refusing to overwrite different analysis summary: {OUT}")
    if not OUT.exists():
        temp = OUT.with_suffix(OUT.suffix + ".tmp")
        temp.write_text(encoded)
        temp.replace(OUT)
    print(json.dumps({"analysis_path": relative(OUT), "sha256": digest(OUT), "status": result["status"], "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
