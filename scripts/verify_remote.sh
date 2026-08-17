#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    DEFAULT_PYTHON="${ROOT}/.venv/bin/python"
else
    DEFAULT_PYTHON="/workspace/hongjia/envs/vla-jepa/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/FACT:${ROOT}/third_party/flux2/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m pytest -q "${ROOT}/tests"
