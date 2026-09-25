"""Build paired AeroWall Launch momentum counterfactual banks from audited snapshots."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

EXPECTED_PAIRS = (
    ("heldout-0008", "heldout-0095"),
    ("heldout-0012", "heldout-0100"),
    ("heldout-0018", "heldout-0004"),
    ("heldout-0033", "heldout-0096"),
    ("heldout-0051", "heldout-0029"),
    ("heldout-0059", "heldout-0075"),
    ("heldout-0073", "heldout-0070"),
    ("heldout-0082", "heldout-0037"),
    ("heldout-0099", "heldout-0068"),
    ("heldout-0107", "heldout-0124"),
    ("heldout-0112", "heldout-0103"),
    ("heldout-0113", "heldout-0058"),
)
EXPECTED_CASE_BANK_SHA256 = "666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80"
EXPECTED_LAUNCH_SHA256 = "bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50"
EXPECTED_HIT_SHA256 = "d970228e9c2fdc662fc6fa9b0aead0672628ffd0da947dd04a93685a31968dd7"
EXPECTED_RECOVER_SHA256 = "4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable_bytes(path, encoded):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise ValueError(f"refusing to overwrite a different frozen bank: {path}")


def write_pair(control_path, control, intervention_path, intervention):
    control_text = json.dumps(control, indent=2, sort_keys=True) + "\n"
    intervention_text = json.dumps(intervention, indent=2, sort_keys=True) + "\n"
    immutable_bytes(control_path, control_text)
    immutable_bytes(intervention_path, intervention_text)
    for path, encoded in ((control_path, control_text), (intervention_path, intervention_text)):
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(encoded)
            temp.replace(path)
    return sha256(control_path), sha256(intervention_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-bank", type=Path, required=True)
    parser.add_argument("--pair-report", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, required=True)
    parser.add_argument("--intervention-output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot_bank.read_text())
    pairing = json.loads(args.pair_report.read_text())
    if snapshot.get("bank_kind") != "aerowall_launch_handoff_v1":
        raise ValueError("snapshot input must be an AeroWall Launch precontact handoff bank")
    if snapshot.get("handoff_actor_role") != "launch":
        raise ValueError("snapshot bank must explicitly identify Launch actor role")
    if snapshot.get("source_initial_case_bank_sha256") != EXPECTED_CASE_BANK_SHA256:
        raise ValueError("source snapshots are not from the preregistered heldout-128-v4 bank")
    hashes = snapshot.get("source_checkpoint_sha256", {})
    for role, expected in (("launch", EXPECTED_LAUNCH_SHA256),
                           ("hit", EXPECTED_HIT_SHA256),
                           ("recovery", EXPECTED_RECOVER_SHA256)):
        if hashes.get(role) != expected:
            raise ValueError(f"source {role} checkpoint hash differs from the preregistered route")
    if snapshot.get("source_contact_audit", {}).get("callback_errors"):
        raise ValueError("source rollout contains PhysX callback errors")

    analysis = pairing["control_phase0_failure_vs_legal_matched_approach"]
    actual_pairs = tuple((row["illegal_case_id"], row["matched_legal_case_id"])
                         for row in analysis["pairs"])
    if actual_pairs != EXPECTED_PAIRS:
        raise ValueError("matched case pairs differ from the preregistration")
    source_cases = snapshot.get("cases", [])
    by_source_id = {case.get("source_case_id"): case for case in source_cases}
    if len(by_source_id) != len(source_cases):
        raise ValueError("snapshot bank contains duplicate source case IDs")

    control_cases = []
    intervention_cases = []
    vy_changes = []
    for failed_id, legal_id in EXPECTED_PAIRS:
        if failed_id not in by_source_id or legal_id not in by_source_id:
            raise ValueError(f"missing preregistered source snapshots for {failed_id}/{legal_id}")
        failed = by_source_id[failed_id]
        legal = by_source_id[legal_id]
        for case in (failed, legal):
            lead = float(case.get("source_time_to_contact_s", float("nan")))
            if not 0.38 <= lead <= 0.42 or case.get("source_event_prefix") != []:
                raise ValueError("source snapshot is outside the preregistered phase-0 time window")
        before = copy.deepcopy(failed)
        after = copy.deepcopy(failed)
        reference_vy = float(legal["drone_velocity"][1])
        previous_vy = float(failed["drone_velocity"][1])
        # The preregistered intervention copies the paired legal-state value;
        # its direction is not constrained, so retain pairs where that value is higher.
        after["drone_velocity"][1] = reference_vy
        changed = {key for key in before if before[key] != after[key]}
        if changed != {"drone_velocity"}:
            raise ValueError(f"intervention changed unexpected fields for {failed_id}: {sorted(changed)}")
        if any(before["drone_velocity"][i] != after["drone_velocity"][i]
               for i in range(6) if i != 1):
            raise ValueError(f"intervention changed a non-y velocity component for {failed_id}")
        control_cases.append(before)
        intervention_cases.append(after)
        vy_changes.append(previous_vy - reference_vy)

    common = {
        "handoff_bank_version": 1,
        "bank_kind": "aerowall_launch_handoff_v1",
        "handoff_actor_role": "launch",
        "experiment_name": "AeroWallLaunchLateralMomentumCounterfactualV1",
        "source_snapshot_bank": str(args.snapshot_bank.resolve()),
        "source_snapshot_bank_sha256": sha256(args.snapshot_bank),
        "source_pair_report": str(args.pair_report.resolve()),
        "source_pair_report_sha256": sha256(args.pair_report),
        "source_initial_case_bank_sha256": EXPECTED_CASE_BANK_SHA256,
        "source_checkpoint_sha256": hashes,
        "source_contact_audit": snapshot["source_contact_audit"],
        "evaluation_seed": 9524,
        "count": len(EXPECTED_PAIRS),
    }
    control = {**common, "condition": "control", "cases": control_cases}
    intervention = {
        **common,
        "condition": "world_y_velocity_set_to_matched_legal_case",
        "intervention_field": "drone_velocity[1]",
        "cases": intervention_cases,
    }
    control_sha, intervention_sha = write_pair(
        args.control_output, control, args.intervention_output, intervention,
    )
    ordered = sorted(vy_changes)
    median = (ordered[5] + ordered[6]) / 2.0
    print(json.dumps({
        "experiment_name": common["experiment_name"],
        "count": len(EXPECTED_PAIRS),
        "control_bank_sha256": control_sha,
        "intervention_bank_sha256": intervention_sha,
        "intervened_velocity_delta_m_s": {
            "min": min(vy_changes), "median": median, "max": max(vy_changes),
        },
        "only_changed_case_field": "drone_velocity[1]",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
