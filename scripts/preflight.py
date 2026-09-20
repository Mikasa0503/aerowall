"""Read-only deployment inventory; this never certifies a physics/training gate."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "docs/upstream-lock.json").read_text())
    repositories = []
    for entry in lock["repositories"]:
        path = root / entry["path"]
        head = run(["git", "-C", str(path), "rev-parse", "HEAD"])
        dirty = run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"])
        links = run(["git", "-C", str(path), "ls-tree", "-r", "HEAD"])
        actual_links = {}
        for line in links.get("stdout", "").splitlines():
            if line.startswith("160000 "):
                meta, relpath = line.split("\t", 1)
                actual_links[relpath] = meta.split()[2]
        required_submodules = []
        for relpath, commit in entry["submodules"].items():
            subhead = run(["git", "-C", str(path / relpath), "rev-parse", "HEAD"])
            required_submodules.append({"path": relpath, "expected": commit,
                                        "head": subhead.get("stdout"),
                                        "matches": subhead.get("stdout") == commit})
        repositories.append({"path": entry["path"], "expected": entry["commit"],
                             "head": head, "worktree": dirty,
                             "gitlinks_match": actual_links == entry["submodules"],
                             "materialized_submodules": required_submodules,
                             "matches": head.get("stdout") == entry["commit"]
                             and dirty.get("returncode") == 0 and not dirty.get("stdout")
                             and actual_links == entry["submodules"]
                             and (not entry["runtime_dependency"] or
                                  all(x["matches"] for x in required_submodules))})
    runtime = Path(os.environ.get("AEROWALL_ISAACSIM_PATH", str(root / "third_party/isaac-sim-2023.1.0-hotfix.1")))
    usage = shutil.disk_usage(root)
    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(),
        "python": {"executable": sys.executable, "version": platform.python_version(),
                   "user_site_disabled": os.environ.get("PYTHONNOUSERSITE") == "1"},
        "project": str(root), "disk_bytes": dict(zip(("total", "used", "free"), usage)),
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv"]),
        "gpu_processes": run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]),
        "repositories": repositories,
        "runtime": {"expected_version": "2023.1.0-hotfix.1", "path": str(runtime),
                    "setup_script_exists": (runtime / "setup_conda_env.sh").is_file()},
        "gates": {"gpu_physics": "not_run", "16_envs_10000_steps": "not_run",
                  "ppo_update_save_reload": "not_run", "collision_calibration": "not_run"},
    }
    report["deployment_ready_for_probe"] = (
        all(repo["matches"] for repo in repositories)
        and sys.version_info[:2] == (3, 10)
        and report["python"]["user_site_disabled"]
        and report["runtime"]["setup_script_exists"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(args.output), "source_locks_match": all(r["matches"] for r in repositories),
                      "deployment_ready_for_probe": report["deployment_ready_for_probe"],
                      "physics_training_gates": "not_run"}))
    return 0 if report["deployment_ready_for_probe"] else 2


if __name__ == "__main__":
    sys.exit(main())
