#!/usr/bin/env bash
# 中文：正式入口：启动单次 MAC 训练；阶段由配置指定。
# English: Public entry: launch one MAC training phase selected by config.
# 调用 / Invocation: 通过 bash 调用；创建训练日志和 checkpoint。 / Run with bash; writes logs and checkpoints.
# 导航 / Guide: scripts/README.md (public / 正式入口)
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="${ROBONANA_PROJECT_DIR:-${repo_root}/experiments/hanging_mug_mac}"
python_bin="${ROBONANA_PYTHON:-${repo_root}/.venv/bin/python}"
log_dir="${ROBONANA_LOG_DIR:-${project_dir}/logs}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${log_dir}/train_${run_stamp}.log"

mkdir -p "${log_dir}"
exec > >(tee -a "${log_file}") 2>&1
trap 'status=$?; echo "[$(date -u +%FT%TZ)] training launcher exit=${status} log=${log_file}"; exit ${status}' EXIT

echo "[$(date -u +%FT%TZ)] repo=${repo_root}"
echo "[$(date -u +%FT%TZ)] persistent stdout/stderr=${log_file}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2/src:${repo_root}/third_party/flux2_official/src${PYTHONPATH:+:${PYTHONPATH}}"
export ROBONANA_PROJECT_DIR="${project_dir}"

"${python_bin}" scripts/internal/train_robotwin.py "$@"
