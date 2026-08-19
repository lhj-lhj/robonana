#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="${ROBONANA_PROJECT_DIR:-${repo_root}/experiments/robotwin_flux2}"
log_dir="${ROBONANA_LOG_DIR:-${project_dir}/logs}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${log_dir}/train_${run_stamp}.log"

mkdir -p "${log_dir}"
exec > >(tee -a "${log_file}") 2>&1
trap 'status=$?; echo "[$(date -u +%FT%TZ)] training launcher exit=${status} log=${log_file}"; exit ${status}' EXIT

echo "[$(date -u +%FT%TZ)] repo=${repo_root}"
echo "[$(date -u +%FT%TZ)] persistent stdout/stderr=${log_file}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2/src${PYTHONPATH:+:${PYTHONPATH}}"
export ROBONANA_PROJECT_DIR="${project_dir}"

"${repo_root}/.venv/bin/python" scripts/train_robotwin.py "$@"
