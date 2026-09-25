"""Describe paired phase/contact changes in the seeded Tanh Launch replication."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility"
PREFIX = "aerowall-tanh-launch-independent-replication-v1"
BANK_SHA = "b6ebf13209ba42b1ad77eb452248b6f743c0f0fbcc32805d250fbc162b3804da"
REPORT_SHA = {
    "control": "5ad16aef081ff2db5c469a78eaa1519218ac9e217f63cfdbb62e58afd8834c61",
    "tanh": "e027e86fb3446a0da05863058007c074f2280eb1a85f6f7d1e9db549cd003935",
}
SAFETY = {2, 4, 5}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arm(name):
    stem = ART / (PREFIX + "-" + name + "-s260925")
    report_path = stem.with_suffix(".json")
    require(digest(report_path) == REPORT_SHA[name], name + " report hash mismatch")
    report = json.loads(report_path.read_text())
    require(report["initial_case_bank_sha256"] == BANK_SHA, name + " bank mismatch")
    outcomes = {row["case_id"]: row for row in report["outcomes"]}
    require(len(outcomes) == 128, name + " case count mismatch")
    events = [json.loads(line) for line in stem.with_suffix(".events.jsonl").read_text().splitlines() if line]
    event_by_env = {}
    for event in events:
        event_by_env.setdefault(int(event["env"]), []).append(event)
    case_by_env = {int(row["env"]): row["case_id"] for row in report["outcomes"]}
    require(set(event_by_env).issubset(set(case_by_env)), name + " event has unknown env")
    return report, outcomes, event_by_env, case_by_env


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def final_contact_roles(report, event_by_env, case_by_env):
    counts = {}
    for row in report["outcomes"]:
        if int(row["failure"]) not in SAFETY:
            continue
        events = event_by_env.get(int(row["env"]), [])
        if not events:
            label = ("no_contact_event", "none", "none")
        else:
            event = events[-1]
            label = (str(event.get("phase_before")), str(event.get("actor_observation_role")),
                     "legal_cap" if event.get("legal_cap") else "nonlegal_cap")
        counts["|".join(label)] = counts.get("|".join(label), 0) + 1
    return counts


def short_event(event):
    return {
        "policy_step": event.get("policy_step"),
        "substep": event.get("substep"),
        "phase_before": event.get("phase_before"),
        "actor_observation_role": event.get("actor_observation_role"),
        "legal_cap": event.get("legal_cap"),
        "wall": event.get("wall"),
        "radial_error_m": event.get("radial_error"),
        "racket_radius_m": event.get("racket_radius"),
        "body": event.get("body"),
    }


def main():
    c_report, c, c_events, c_env = load_arm("control")
    t_report, t, t_events, t_env = load_arm("tanh")
    require(set(c) == set(t), "paired case IDs differ")
    c_first = {case for case, row in c.items() if int(row["caps"]) >= 1}
    t_first = {case for case, row in t.items() if int(row["caps"]) >= 1}
    c_safety = {case for case, row in c.items() if int(row["failure"]) in SAFETY}
    t_safety = {case for case, row in t.items() if int(row["failure"]) in SAFETY}
    c_rallies = {case: int(row["rallies"]) for case, row in c.items()}
    t_rallies = {case: int(row["rallies"]) for case, row in t.items()}
    rally_worse = sorted(case for case in c if t_rallies[case] < c_rallies[case])
    rally_better = sorted(case for case in c if t_rallies[case] > c_rallies[case])
    tanh_only_safety_same_first = sorted((t_safety - c_safety) & c_first & t_first)
    downstream_details = []
    downstream_phase_roles = {}
    downstream_radials = []
    for case in tanh_only_safety_same_first:
        row = t[case]
        events = t_events.get(int(row["env"]), [])
        require(int(row["failure"]) == 4, case + " is not an illegal-contact failure")
        require(int(row["caps"]) >= 1 and int(c[case]["caps"]) >= 1,
                case + " first-contact status is not shared")
        require(events, case + " has no recorded contact event")
        last = events[-1]
        phase_role = str(last.get("phase_before")) + "|" + str(last.get("actor_observation_role"))
        downstream_phase_roles[phase_role] = downstream_phase_roles.get(phase_role, 0) + 1
        radial = float(last["radial_error"])
        radius = float(last["racket_radius"])
        downstream_radials.append(radial)
        downstream_details.append({
            "case_id": case,
            "control": {
                "caps": c[case]["caps"], "rallies": c[case]["rallies"],
                "failure": c[case]["failure"], "policy_steps": c[case]["policy_steps"]
            },
            "tanh": {
                "caps": row["caps"], "rallies": row["rallies"],
                "failure": row["failure"], "policy_steps": row["policy_steps"],
                "last_contact": short_event(last)
            }
        })
    case_id = "heldout-0063"
    require(c[case_id]["rallies"] == 1 and t[case_id]["rallies"] == 0,
            "frozen exemplar outcome changed")
    c0063 = [event for event in c_events[int(c[case_id]["env"])] if event["policy_step"] <= 31]
    t0063 = [event for event in t_events[int(t[case_id]["env"])] if event["policy_step"] <= 31]
    result = {
        "schema_version": 2,
        "experiment_name": "AeroWallLaunchTanhIndependentReplicationV1PhaseDifferences",
        "status": "complete",
        "scope": "descriptive paired event review; no new policy intervention",
        "seed": 260925,
        "bank_sha256": BANK_SHA,
        "source_reports_sha256": REPORT_SHA,
        "first_contact_proxy": {
            "tanh_only_case_ids": sorted(t_first - c_first),
            "control_only_case_ids": sorted(c_first - t_first),
        },
        "safety_failure_case_changes": {
            "tanh_only_case_ids": sorted(t_safety - c_safety),
            "control_only_case_ids": sorted(c_safety - t_safety),
            "tanh_only_cases_that_lost_first_contact": sorted((t_safety - c_safety) & (c_first - t_first)),
            "tanh_only_cases_with_first_contact_unchanged": sorted((t_safety - c_safety) & (c_first & t_first)),
            "tanh_only_cases_that_gained_first_contact": sorted((t_safety - c_safety) & (t_first - c_first)),
        },
        "tanh_only_safety_after_shared_first_contact": {
            "case_count": len(downstream_details),
            "cases": downstream_details,
            "last_contact_phase_role_counts": downstream_phase_roles,
            "last_contact_radial_error_m": {
                "min": min(downstream_radials), "max": max(downstream_radials),
                "all_exceed_racket_radius": all(
                    detail["tanh"]["last_contact"]["radial_error_m"] >
                    detail["tanh"]["last_contact"]["racket_radius_m"]
                    for detail in downstream_details
                )
            },
            "interpretation_limit": "The first-contact count is shared within each pair; later last-contact locations are descriptive and do not isolate the causal source of the state difference."
        },
        "rally_case_changes": {
            "worse_case_ids": rally_worse,
            "better_case_ids": rally_better,
            "unchanged_count": 128 - len(rally_worse) - len(rally_better),
        },
        "last_logged_contact_phase_role_for_safety_failure_episodes": {
            "control": final_contact_roles(c_report, c_events, c_env),
            "tanh": final_contact_roles(t_report, t_events, t_env),
            "interpretation_limit": "Descriptive last logged contact classification; not a causal attribution for aggregate failure changes.",
        },
        "heldout_0063_contact_sequence": {
            "control_outcome": c[case_id],
            "tanh_outcome": t[case_id],
            "control_contacts_through_step_31": [short_event(event) for event in c0063],
            "tanh_contacts_through_step_31": [short_event(event) for event in t0063],
            "interpretation": "One paired case changed from a legal phase-0 Launch cap at step 31 to a nonlegal phase-0 Launch contact at the same policy step. This is a case-level illustration, not evidence of population generalization.",
        },
        "decision": "No new candidate training or promotion; formal C350 unchanged; P2=false.",
    }
    out = ART / (PREFIX + "-phase-analysis-s260925.json")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    out.write_text(encoded)
    print(json.dumps({
        "analysis": str(out.relative_to(ROOT)),
        "sha256": digest(out),
        "first_contact_tanh_only": len(t_first-c_first),
        "first_contact_control_only": len(c_first-t_first),
        "safety_tanh_only": len(t_safety-c_safety),
        "safety_control_only": len(c_safety-t_safety),
        "same_first_tanh_only_safety": len(downstream_details),
        "downstream_phase_roles": downstream_phase_roles,
        "all_downstream_radials_exceed_radius": all(x > 0.2 for x in downstream_radials),
        "rally_worse": rally_worse,
        "rally_better": rally_better
    }, sort_keys=True))


if __name__ == "__main__":
    main()
