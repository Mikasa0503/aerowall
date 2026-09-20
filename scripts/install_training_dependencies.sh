#!/usr/bin/env bash
# Install only into the project environment; preserve bundled Torch and system GCC.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 MAX_JOBS=4
for compiler in "$CC" "$CXX"; do test -x "$compiler"; done
# Avoid inherited invalid proxy settings without editing shared configuration.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
./scripts/python.sh --plain -m pip --isolated install --index-url https://pypi.org/simple \
    setuptools==68.2.2 wheel==0.41.3 cloudpickle==2.2.1 ninja==1.11.1.1
./scripts/python.sh --plain -m pip --isolated install --index-url https://pypi.org/simple \
    --only-binary=av -r docs/training-python-requirements.txt
for source in \
    third_party/JuggleRL_train/third_party/tensordict \
    third_party/JuggleRL_train/third_party/rl \
    third_party/JuggleRL_train/third_party/orbit/source/extensions/omni.isaac.orbit \
    third_party/JuggleRL_train; do
    ./scripts/python.sh -m pip install --no-build-isolation --no-deps -e "$source"
done
./scripts/python.sh --plain scripts/prepare_assets.py
./scripts/python.sh -c 'import torch,tensordict,torchrl; print(torch.__version__,tensordict.__version__,torchrl.__version__)'
./scripts/python.sh --plain -m pip freeze > docs/training-env-freeze.txt
