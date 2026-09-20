#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA=/home/public/Workspace/boweiy/mambaforge/bin/conda
PREFIX="${AEROWALL_ENV:-/home/public/Workspace/boweiy/mambaforge/envs/aerowall}"
export CONDA_PKGS_DIRS="$ROOT/.cache/conda/pkgs"
export TMPDIR="$ROOT/.cache/tmp"
export PYTHONNOUSERSITE=1
export CONDARC="$ROOT/.cache/condarc"
mkdir -p "$CONDA_PKGS_DIRS" "$TMPDIR" "$ROOT/runs"
# Override inherited Conda proxy entries only for this project.
cat > "$CONDARC" <<EOF
channels:
  - conda-forge
proxy_servers:
  http: ""
  https: ""
pkgs_dirs:
  - $CONDA_PKGS_DIRS
EOF
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
if [[ -d "$PREFIX/conda-meta" ]]; then
    "$PREFIX/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version; print(sys.version)'
    echo "Existing Python 3.10 environment preserved: $PREFIX"
elif [[ -e "$PREFIX" ]]; then
    echo "Refusing to overwrite incomplete or unrelated directory: $PREFIX" >&2
    exit 1
else
    "$CONDA" create --prefix "$PREFIX" --override-channels -c conda-forge python=3.10 pip -y
fi
"$CONDA" list --prefix "$PREFIX" --explicit > "$ROOT/docs/conda-linux-64-explicit.txt"
