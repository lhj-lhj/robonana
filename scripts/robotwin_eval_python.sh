#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
robotwin_python=${ROBONANA_ROBOTWIN_PYTHON:?set ROBONANA_ROBOTWIN_PYTHON to the RoboTwin interpreter}
exec "${robotwin_python}" "${script_dir}/robotwin_eval_bootstrap.py" "$@"
