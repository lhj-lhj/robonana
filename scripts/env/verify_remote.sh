#!/usr/bin/env bash
# 中文：环境检查：用指定解释器运行仓库测试。
# English: Environment check: run repository tests with the selected interpreter.
# 调用 / Invocation: 通过 PYTHON_BIN 指定已配置环境；不启动训练。 / Set PYTHON_BIN to the configured environment; does not start training.
# 导航 / Guide: scripts/README.md (env)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    DEFAULT_PYTHON="${ROOT}/.venv/bin/python"
else
    DEFAULT_PYTHON="/workspace/hongjia/envs/vla-jepa/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/FACT:${ROOT}/third_party/flux2/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m pytest -q "${ROOT}/tests"
