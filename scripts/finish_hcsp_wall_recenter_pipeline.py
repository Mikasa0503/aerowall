"""Run challenge, authentic demo, and report after formal repeats finish."""

import json
import subprocess
import time

from project_paths import ROOT, RUNS, PYTHON, runtime_environment


def wait_for_formal():
    deadline = time.monotonic() + 7200
    summary = RUNS / "formal-summary.json"
    while time.monotonic() < deadline:
        if summary.exists():
            report = json.loads(summary.read_text())
            if report.get("status") == "passed":
                group = report["selection"]["selected_demo_group"].lower()
                repeats = [json.loads((RUNS / f"{group}-formal-repeat-{index:02d}.json").read_text())
                           for index in range(20)]
                assert all(item["status"] == "passed" and
                           item["outcomes"][0]["contact_audit_passed"] for item in repeats)
                assert sum(item["outcomes"][0]["safe10"] for item in repeats) == report["repeat_safe10_count"]
                return report
        time.sleep(10)
    raise TimeoutError("Formal 20-process evaluation did not finish within two hours")


def run(name, script, *arguments):
    environment = runtime_environment()
    environment["OMP_NUM_THREADS"] = environment["MKL_NUM_THREADS"] = "4"
    log_path = RUNS / (name + ".log")
    started = time.monotonic()
    with log_path.open("w") as log:
        result = subprocess.run([str(PYTHON), "--plain", str(ROOT / "scripts" / script), *arguments],
                                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    print(json.dumps({"stage": name, "returncode": result.returncode, "seconds": elapsed,
                      "log": str(log_path)}), flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {log_path}")


def main():
    formal = wait_for_formal()
    assert formal["selection"]["selected_demo_group"] in ("B", "C")
    run("challenge-driver", "evaluate_hcsp_wall_challenge.py")
    run("demo-driver", "render_hcsp_wall_recenter_demo.py")
    run("summary-driver", "summarize_hcsp_wall_recenter_results.py",
        "--output", str(ROOT / "artifacts/hcsp-wall-recenter-results.md"))


if __name__ == "__main__":
    main()
