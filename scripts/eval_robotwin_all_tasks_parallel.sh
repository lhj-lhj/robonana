#!/usr/bin/env bash
# 中文：不再翻译旧环境变量，直接传递统一配置。
# English: Direct invocation of the single evaluation entrypoint.
# 调用 / Invocation: bash scripts/eval_robotwin_all_tasks_parallel.sh --config configs/eval.json --execute
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2_official/src:${PYTHONPATH:-}"
exec python "${repo_root}/scripts/run_multitask_mbrl.py" eval "$@"
