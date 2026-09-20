#!/usr/bin/env bash
# Kit fast shutdown may return zero after a Python failure; require the JSON verdict.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -lt 2 ]]; then
    echo "Usage: scripts/run_probe.sh SCRIPT OUTPUT_JSON [probe arguments...]" >&2
    exit 2
fi
probe_script="$1"
probe_output="$2"
shift 2
if [[ -e "$probe_output" ]]; then
    echo "Preserving existing report: $probe_output" >&2
    exit 3
fi
timeout --signal=TERM --kill-after=15s "${AEROWALL_PROBE_TIMEOUT:-900}s" \
    ./scripts/python.sh "$probe_script" --output "$probe_output" "$@"
probe_exit=$?
./scripts/python.sh --plain - "$probe_output" "$probe_exit" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
process_exit = int(sys.argv[2])
report = json.loads(path.read_text()) if path.exists() else {'status': 'missing_report'}
verdict = {'process_exit': process_exit, 'reported_status': report['status'],
           'passed': process_exit == 0 and report['status'] == 'passed'}
path.with_suffix('.exit.json').write_text(json.dumps(verdict, indent=2) + '\n')
print(json.dumps(verdict), flush=True)
sys.exit(0 if verdict['passed'] else 1)
PY
