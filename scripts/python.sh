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
export __GL_SHADER_DISK_CACHE_PATH="$ROOT/.cache/gl"
export OMNI_CONFIG_PATH="$ROOT/.cache/omniverse-config"
export TMPDIR="$ROOT/.cache/tmp"
export WANDB_DIR="$ROOT/runs"
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_MODE=offline
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$WANDB_DIR"
mkdir -p "$OMNI_CONFIG_PATH" "$ROOT/.cache/ov/data" "$ROOT/.cache/ov/cache" "$ROOT/runs/kit" "$__GL_SHADER_DISK_CACHE_PATH"
cat > "$OMNI_CONFIG_PATH/omniverse.toml" <<EOF
[paths]
data_root = "$ROOT/.cache/ov/data"
cache_root = "$ROOT/.cache/ov/cache"
logs_root = "$ROOT/runs/kit"
EOF
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
    # This author archive adds empty and ../../../ entries. Keep the runtime's
    # library order, but do not search the working directory or sibling projects.
    runtime_paths_only() {
        local entry normalized result=""
        local -a entries
        IFS=: read -r -a entries <<< "$1"
        for entry in "${entries[@]}"; do
            [[ -n "$entry" ]] || continue
            normalized="$(realpath -m -- "$entry")"
            case "$normalized" in
                "$ISAACSIM_PATH"|"$ISAACSIM_PATH"/*)
                    result="${result:+$result:}$normalized" ;;
            esac
        done
        printf '%s' "$result"
    }
    export PYTHONPATH="$(runtime_paths_only "$PYTHONPATH")"
    export LD_LIBRARY_PATH="$(runtime_paths_only "$LD_LIBRARY_PATH")"
    # Importing upstream Torch before Kit can load Ubuntu's older libstdc++.
    # The isolated Python sqlite/ICU stack needs its own newer C++ ABI.
    # Pin only this library, without exposing all Conda libraries to Kit.
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
fi
exec "$CONDA_PREFIX/bin/python" "$@"
