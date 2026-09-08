#!/usr/bin/env bash
# 中文：环境辅助：用指定的 RoboTwin Python 启动仿真适配。
# English: Environment helper: launch simulation bootstrap with the selected RoboTwin Python.
# 调用 / Invocation: 由评测/采集间接调用，不安装软件。 / Called by eval/collection; does not install packages.
# 导航 / Guide: scripts/README.md (env)
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
robotwin_python=${ROBONANA_ROBOTWIN_PYTHON:?set ROBONANA_ROBOTWIN_PYTHON to the RoboTwin interpreter}
exec "${robotwin_python}" "${script_dir}/robotwin_eval_bootstrap.py" "$@"
