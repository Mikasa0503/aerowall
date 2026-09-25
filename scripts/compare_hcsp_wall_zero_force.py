"""Compare matched natural rollouts with and without the zero-force API call."""

import argparse
import json
from pathlib import Path


def compare(normal, zero_force):
    for report in (normal, zero_force):
        assert report["status"] == "passed"
        assert report["protocol"] == "natural"
        assert all(row["contact_audit_passed"] for row in report["outcomes"])
    assert normal["zero_force_control"] is False
    assert zero_force["zero_force_control"] is True
    for key in ("checkpoint_sha256", "num_envs", "seed", "experiment_sha256"):
        assert normal[key] == zero_force[key], key
    left = {row["env"]: row for row in normal["outcomes"]}
    right = {row["env"]: row for row in zero_force["outcomes"]}
    assert set(left) == set(right) == set(range(normal["num_envs"]))
    rows = []
    for env in sorted(left):
        a, b = left[env], right[env]
        paired = min(len(a["wall_y_sequence"]), len(b["wall_y_sequence"]))
        differences = [abs(x - y) for x, y in zip(a["wall_y_sequence"][:paired],
                                                    b["wall_y_sequence"][:paired])]
        rows.append({
            "env": env,
            "normal_rallies": a["rallies"],
            "zero_force_rallies": b["rallies"],
            "normal_centered_prefix": a["centered_prefix_rallies"],
            "zero_force_centered_prefix": b["centered_prefix_rallies"],
            "normal_failure": a["failure_reason"],
            "zero_force_failure": b["failure_reason"],
            "normal_wall_count": len(a["wall_y_sequence"]),
            "zero_force_wall_count": len(b["wall_y_sequence"]),
            "max_paired_wall_y_difference_m": max(differences, default=0.0),
            "rallies_differ": a["rallies"] != b["rallies"],
            "failure_differs": a["failure_reason"] != b["failure_reason"],
        })
    return {
        "checkpoint_sha256": normal["checkpoint_sha256"],
        "seed": normal["seed"],
        "num_envs": normal["num_envs"],
        "rally_count_differences": sum(row["rallies_differ"] for row in rows),
        "failure_reason_differences": sum(row["failure_differs"] for row in rows),
        "max_paired_wall_y_difference_m": max(
            (row["max_paired_wall_y_difference_m"] for row in rows), default=0.0),
        "per_env": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--zero-force", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.normal.read_text()),
                     json.loads(args.zero_force.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_env"}))


if __name__ == "__main__":
    main()
