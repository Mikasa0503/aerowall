#!/usr/bin/env bash
# Project-local launcher. Do not source this script into a user's shell.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export AEROWALL_ROOT="$ROOT"
export CONDA_PREFIX="${AEROWALL_ENV:-/home/public/Workspace/boweiy/mambaforge/envs/aerowall}"
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export XDG_CONFIG_HOME="$ROOT/.cache/config"
export XDG_DATA_HOME="$ROOT/.cache/data"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export TORCH_HOME="$ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions"
export CUDA_CACHE_PATH="$ROOT/.cache/cuda"
export TMPDIR="$ROOT/.cache/tmp"
export WANDB_DIR="$ROOT/runs"
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_MODE=offline
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$WANDB_DIR"
if [[ ! -x "$CONDA_PREFIX/bin/python" ]]; then
    echo "Missing isolated Python environment: $CONDA_PREFIX" >&2
    exit 20
fi
# Supply --plain for inventory tools that must run before Isaac Sim exists.
if [[ "${1:-}" == --plain ]]; then
    shift
else
    export ISAACSIM_PATH="${AEROWALL_ISAACSIM_PATH:-$ROOT/third_party/isaac-sim-2023.1.0-hotfix.1}"
    if [[ ! -f "$ISAACSIM_PATH/setup_conda_env.sh" ]]; then
        echo "Missing Isaac Sim runtime: $ISAACSIM_PATH/setup_conda_env.sh" >&2
        exit 21
    fi
    # Preserve Isaac Sim's library and Python setup order, without upstream SSH DISPLAY changes.
    source "$ISAACSIM_PATH/setup_conda_env.sh"
fi
exec "$CONDA_PREFIX/bin/python" "$@"
