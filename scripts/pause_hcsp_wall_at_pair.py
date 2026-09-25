"""Pause a live experiment driver at a completed matched pair for review.

This does not alter checkpoints or reports. Resume with SIGCONT on the recorded
PIDs, or terminate them after explicitly recording a stop reason.
"""

import argparse
import json
import os
import signal
import time
from pathlib import Path


def command_line(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return ""


def descendants(pid):
    parents = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            rows = status.read_text().splitlines()
            parent = next(int(row.split()[1]) for row in rows if row.startswith("PPid:"))
            parents.setdefault(parent, []).append(int(status.parent.name))
        except (FileNotFoundError, StopIteration):
            continue
    result, frontier = [], [pid]
    while frontier:
        children = parents.get(frontier.pop(), [])
        result.extend(children)
        frontier.extend(children)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-pid", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--pair", type=int, required=True)
    args = parser.parse_args()
    assert "run_hcsp_wall_recenter_experiment.py" in command_line(args.driver_pid)
    while True:
        if not command_line(args.driver_pid):
            raise RuntimeError("Experiment driver exited before the target pair")
        state = json.loads(args.state.read_text())
        if args.pair in state.get("completed_pairs", []):
            os.kill(args.driver_pid, signal.SIGSTOP)
            time.sleep(0.05)
            children = descendants(args.driver_pid)
            for pid in children:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
            payload = {
                "pair": args.pair,
                "driver_pid": args.driver_pid,
                "paused_descendant_pids": children,
                "completed_pairs": state["completed_pairs"],
                "pair_metrics": state["pair_metrics"].get(str(args.pair)),
                "observed_at_unix": time.time(),
            }
            args.marker.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(payload), flush=True)
            return
        time.sleep(0.2)


if __name__ == "__main__":
    main()
