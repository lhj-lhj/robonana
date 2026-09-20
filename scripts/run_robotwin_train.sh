#!/usr/bin/env bash
# 中文：唯一配置入口的便捷调用；使用当前激活的 Python，不读取旧实验环境变量。
# English: Thin launcher for the single explicit training config.
# 调用 / Invocation: bash scripts/run_robotwin_train.sh --config configs/train.json --execute
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2/src:${repo_root}/third_party/flux2_official/src:${PYTHONPATH:-}"
exec python "${repo_root}/scripts/run_multitask_mbrl.py" train "$@"
