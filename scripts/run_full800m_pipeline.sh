#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${ROBONANA_PYTHON:-/data3/hongjia/conda/envs/robonana/bin/python}"
dataset_root="${ROBONANA_DATASET_ROOT:-/workspace/datasets/fact-robotwin-v2/RoboTwin}"
checkpoint="${ROBONANA_FLUX_CHECKPOINT_DIR:-${repo_root}/checkpoints/FLUX.2-klein-base-4B}"
job_dir="${ROBONANA_PIPELINE_DIR:-/data3/hongjia/robonana-jobs/full800m_bs256_120k}"
status_file="${job_dir}/status.txt"
pipeline_log="${job_dir}/pipeline.log"
preprocess_gpus="${ROBONANA_PREPROCESS_GPUS:-1,2,3,4,5,6,7}"

mkdir -p "${job_dir}"
exec >>"${pipeline_log}" 2>&1

stage="starting"
write_status() {
  printf 'state=running\nstage=%s\nupdated_at=%s\npid=%s\n' \
    "${stage}" "$(date -u +%FT%TZ)" "$$" >"${status_file}.tmp"
  mv "${status_file}.tmp" "${status_file}"
}
finish() {
  code=$?
  if [[ "${code}" -eq 0 ]]; then
    state="complete"
  else
    state="failed"
  fi
  printf 'state=%s\nstage=%s\nupdated_at=%s\npid=%s\nexit_code=%s\n' \
    "${state}" "${stage}" "$(date -u +%FT%TZ)" "$$" "${code}" >"${status_file}"
  echo "[$(date -u +%FT%TZ)] pipeline exit=${code} stage=${stage}"
}
trap finish EXIT

export PYTHONUNBUFFERED=1
export CUDA_HOME="/usr/local/cuda-12.8"
export PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2/src:${repo_root}/third_party/flux2_official/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="/data3/hongjia/hf-cache"
export TORCH_EXTENSIONS_DIR="/data3/hongjia/torch-extensions"
export TRITON_CACHE_DIR="/data3/hongjia/triton-cache"

cd "${repo_root}"

stage="verify_archive_and_capacity"
write_status
"${python_bin}" - <<'PY'
import json
import shutil
from pathlib import Path

status = json.loads(Path("/data3/hongjia/logs/robotwin_v2_archive_verification.json").read_text())
if not status.get("complete"):
    raise SystemExit("robotwin-v2 archive SHA256 verification is not complete")
free = shutil.disk_usage("/data3").free
print(f"archive_sha256={status['local_sha256']} free_tib={free / 1024**4:.3f}")
if free < 1024**4:
    raise SystemExit("less than 1 TiB free on /data3; refusing full preprocessing")
PY

stage="metadata"
write_status
"${python_bin}" scripts/compute_robotwin_lerobot_metadata.py \
  --dataset-root "${dataset_root}" \
  --task-glob 'Clean/*' \
  --task-glob 'Randomized/*'

stage="qwen3_and_flux_cache"
write_status
IFS=',' read -r -a preprocess_gpu_ids <<<"${preprocess_gpus}"
preprocess_world_size="${#preprocess_gpu_ids[@]}"
CUDA_VISIBLE_DEVICES="${preprocess_gpus}" \
  "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${preprocess_world_size}" \
  scripts/preprocess_robotwin_lerobot_flux.py \
  --dataset-root "${dataset_root}" \
  --checkpoint "${checkpoint}" \
  --stage all \
  --batch-size "${ROBONANA_CACHE_BATCH_SIZE:-64}" \
  --language-batch-size "${ROBONANA_LANGUAGE_BATCH_SIZE:-4}" \
  --pyav-thread-count "${ROBONANA_PYAV_THREADS:-2}"

stage="validate_full_cache"
write_status
"${python_bin}" scripts/validate_robotwin_lerobot_flux.py --dataset-root "${dataset_root}" \
  | tee "${job_dir}/cache_validation.json"

stage="waiting_for_8_gpus"
write_status
while ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
  | awk '$1 > 1024 {busy=1} END {exit busy ? 1 : 0}'; do
  echo "[$(date -u +%FT%TZ)] waiting for all 8 GPUs; current memory MiB:"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  sleep 300
done

stage="waiting_for_wandb_auth"
write_status
until "${python_bin}" -m wandb login --verify >/dev/null 2>&1; do
  echo "[$(date -u +%FT%TZ)] waiting for W&B authentication in ${HOME}/.netrc"
  sleep 300
done

stage="training_800m_bs256_120k"
write_status
export ROBONANA_PYTHON="${python_bin}"
export ROBONANA_DATASET_ROOT="${dataset_root}"
export ROBONANA_PROJECT_DIR="${repo_root}/experiments/robotwin_flux2_800m_full_bs256_120k"
export ROBONANA_GPU_IDS="0,1,2,3,4,5,6,7"
export ROBONANA_BATCH_SIZE="32"
export ROBONANA_MAX_STEPS="120000"
export ROBONANA_PIXEL_EVAL_INTERVAL="1000"
export ROBONANA_CHECKPOINT_INTERVAL="1000"
export ROBONANA_NUM_WORKERS="4"
export ROBONANA_RESUME="1"
bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_800m.config
