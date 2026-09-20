#!/usr/bin/env bash
# 中文：兼容旧 shell 参数；所有任务调度、seed、重试及仿真均走唯一 Python 管线。
# English: Compatibility launcher; no independent evaluation loop lives here.
# 调用 / Invocation: bash script [demo_clean|demo_randomized] [episodes].
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2/src:${repo_root}/third_party/flux2_official/src:${PYTHONPATH:-}"
exec "${ROBONANA_MODEL_PYTHON:-/data3/hongjia/conda/envs/robonana/bin/python}" \
  "${repo_root}/scripts/internal/eval_legacy_args.py" "${1:-${TASK_CONFIG:-demo_clean}}" "${2:-${TEST_NUM:-50}}"
