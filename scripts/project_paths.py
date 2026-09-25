"""Resolve project-local runtime paths with environment overrides."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/recenter-recovery"
PYTHON = ROOT / "scripts/python.sh"
CHECKPOINT_DIR = Path(
    os.environ.get("AEROWALL_CHECKPOINT_DIR", ROOT / "checkpoints")
).expanduser()
ISAACSIM_PATH = Path(
    os.environ.get(
        "AEROWALL_ISAACSIM_PATH",
        ROOT / "third_party/isaac-sim-2023.1.0-hotfix.1",
    )
).expanduser()


def runtime_environment():
    """Return a copy of the current environment with the default simulator path."""
    environment = os.environ.copy()
    environment.setdefault("AEROWALL_ISAACSIM_PATH", str(ISAACSIM_PATH))
    return environment
