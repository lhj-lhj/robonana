#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dataset_root="${ROBOTWIN_ROOT:-/workspace/datasets/RoboTwin/hf_dataset}"
checkpoint="${FLUX2_CHECKPOINT:-$repo_root/checkpoints/FLUX.2-klein-base-4B}"
gpu_list="${ROBONANA_CACHE_GPUS:-0,2,5,7}"
batch_size="${ROBONANA_CACHE_BATCH_SIZE:-16}"
log_path="${ROBONANA_CACHE_LOG:-$repo_root/logs/preprocess_robotwin_flux_full.log}"

IFS=',' read -r -a gpu_ids <<< "$gpu_list"
nproc="${#gpu_ids[@]}"
if [[ "$nproc" -lt 1 ]]; then
  echo "ROBONANA_CACHE_GPUS must contain at least one GPU" >&2
  exit 2
fi

mkdir -p "$(dirname "$log_path")"
exec >>"$log_path" 2>&1

export CUDA_VISIBLE_DEVICES="$gpu_list"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_root/src:$repo_root/third_party/FACT:$repo_root/third_party/flux2_official/src:$repo_root/third_party/flux2/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting RoboTwin FLUX cache: GPUs=$gpu_list workers=$nproc batch_size=$batch_size"
exec .venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node "$nproc" \
  scripts/preprocess_robotwin_flux.py \
  --dataset-root "$dataset_root" \
  --checkpoint "$checkpoint" \
  --stage images \
  --batch-size "$batch_size"
