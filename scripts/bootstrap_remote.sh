#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-/workspace/hongjia/envs/vla-jepa/bin/python}"

cd "${ROOT}"
if [[ ! -x .venv/bin/python ]]; then
    "${BASE_PYTHON}" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install pytest

mkdir -p third_party
if [[ ! -d third_party/FACT/.git ]]; then
    git clone --depth 1 https://github.com/Bariona/FACT.git third_party/FACT
fi
if [[ ! -d third_party/flux2/.git ]]; then
    git clone --depth 1 https://github.com/black-forest-labs/flux2.git third_party/flux2
fi
