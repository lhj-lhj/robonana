#!/usr/bin/env bash
# Wait for the step-1000 evaluation capture, annotate 3x50 episodes, and upload.
set -euo pipefail

PROJECT_ROOT="${ROBONANA_PROJECT_ROOT:-/data3/hongjia/robonana}"
INITIAL_ROOT="${ROBONANA_INITIAL_ROOT:-/workspace/datasets/fact-robotwin-v2/RoboTwin}"
PRE_ROOT="${ROBONANA_PRE_ROOT:-/data3/hongjia/robonana_rollouts/hanging_mug_round0_from160k}"
POST_ROOT="${ROBONANA_POST_ROOT:-/data3/hongjia/robonana_rollouts/hanging_mug_posttrain_step1000_eval50_capture}"
CAPTURE_OUTPUT="${ROBONANA_CAPTURE_OUTPUT:-${PROJECT_ROOT}/outputs/hanging_mug_posttrain_round0_from160k/posttrain_capture_eval50}"
CHECKPOINT="${ROBONANA_CHECKPOINT:-${PROJECT_ROOT}/experiments/hanging_mug_posttrain_round0_from160k/models/checkpoint_epoch_1_step_1000/transformer/diffusion_pytorch_model.bin}"
MODEL_CONFIG="${ROBONANA_MODEL_CONFIG:-${PROJECT_ROOT}/experiments/hanging_mug_posttrain_round0_from160k/config.json}"
FLUX_CHECKPOINT="${ROBONANA_FLUX_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/FLUX.2-klein-base-4B}"
PYTHON="${ROBONANA_PYTHON:-/data3/hongjia/conda/envs/robonana/bin/python}"
HF="${ROBONANA_HF:-/data3/hongjia/conda/envs/robonana/bin/hf}"
HF_REPO="${ROBONANA_HF_REPO:-AvaX1/robonana-eval-videos}"
RUN_NAME="${ROBONANA_RUN_NAME:-hanging_mug_return_overlay150_step1000_20260902}"
OUTPUT_ROOT="${ROBONANA_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/${RUN_NAME}}"
PIPELINE_LOG="${OUTPUT_ROOT}/pipeline_status.txt"

# Never inherit a stale editable install from a migrated environment.  The
# pipeline must resolve RoboNana, FACT, and official FLUX from this checkout.
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/FACT:${PROJECT_ROOT}/third_party/flux2_official/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}/logs"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

capture_ledger="${CAPTURE_OUTPUT}/workers/rank_0_server_6_sim_7/task_runs/hanging_mug/episodes.jsonl"
echo "status=waiting_for_capture started_at=$(date --iso-8601=seconds)"
while true; do
  ledger_count=0
  hdf5_count=0
  [[ -f "${capture_ledger}" ]] && ledger_count="$(wc -l < "${capture_ledger}")"
  [[ -d "${POST_ROOT}" ]] && hdf5_count="$(find "${POST_ROOT}" -type f -name '*.hdf5' | wc -l)"
  echo "capture ledger=${ledger_count}/50 hdf5=${hdf5_count}/50 at=$(date --iso-8601=seconds)"
  if [[ "${ledger_count}" -eq 50 && "${hdf5_count}" -eq 50 && -f "${POST_ROOT}/robonana_index.json" ]]; then
    break
  fi
  if ! pgrep -f "collect_prepare_robotwin_rollouts.sh hanging_mug demo_clean hanging_mug_posttrain_step1000_eval50_capture" >/dev/null; then
    echo "status=failed reason=capture_process_exited_before_50"
    exit 1
  fi
  sleep 60
done

common_args=(
  --checkpoint "${CHECKPOINT}"
  --model-config "${MODEL_CONFIG}"
  --flux-checkpoint-dir "${FLUX_CHECKPOINT}"
  --stats-path "${INITIAL_ROOT}/robonana_norm_stats.json"
  --task-name hanging_mug
  --output-dir "${OUTPUT_ROOT}"
  --expected-episodes 50
  --action-chunk 48
  --num-inference-steps 20
  --model-device cuda:0
  --vae-device cuda:0
  --num-shards 2
)

run_group() {
  local group_name="$1"
  local dataset_format="$2"
  local dataset_root="$3"
  echo "status=annotating group=${group_name} started_at=$(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES=6 "${PYTHON}" scripts/annotate_recorded_robotwin_returns.py \
    "${common_args[@]}" --group-name "${group_name}" --dataset-format "${dataset_format}" \
    --dataset-root "${dataset_root}" --shard-id 0 \
    > "${OUTPUT_ROOT}/logs/${group_name}_shard0.log" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=7 "${PYTHON}" scripts/annotate_recorded_robotwin_returns.py \
    "${common_args[@]}" --group-name "${group_name}" --dataset-format "${dataset_format}" \
    --dataset-root "${dataset_root}" --shard-id 1 \
    > "${OUTPUT_ROOT}/logs/${group_name}_shard1.log" 2>&1 &
  local pid1=$!
  local status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "status=failed group=${group_name} exit_code=${status}"
    exit "${status}"
  fi
  echo "status=annotated group=${group_name} completed_at=$(date --iso-8601=seconds)"
}

cd "${PROJECT_ROOT}"
run_group expert_clean lerobot "${INITIAL_ROOT}"
run_group collected_pre_5of50 hdf5 "${PRE_ROOT}"
run_group posttrain_eval_low_success hdf5 "${POST_ROOT}"

cat "${OUTPUT_ROOT}"/manifest_*_shard_*.jsonl > "${OUTPUT_ROOT}/manifest.jsonl"
video_count="$(find "${OUTPUT_ROOT}/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
telemetry_count="$(find "${OUTPUT_ROOT}/telemetry" -maxdepth 1 -type f -name '*.json' | wc -l)"
manifest_count="$(wc -l < "${OUTPUT_ROOT}/manifest.jsonl")"
if [[ "${video_count}" -ne 150 || "${telemetry_count}" -ne 150 || "${manifest_count}" -ne 150 ]]; then
  echo "status=failed reason=count_mismatch videos=${video_count} telemetry=${telemetry_count} manifest=${manifest_count}"
  exit 1
fi

(
  cd "${OUTPUT_ROOT}"
  find . -type f \
    ! -name SHA256SUMS \
    ! -name pipeline_status.txt \
    ! -name hf_upload_status.json \
    -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
archive="${OUTPUT_ROOT}.tar.zst"
archive_sha="${archive}.sha256"
tar --zstd -cf "${archive}" -C "$(dirname "${OUTPUT_ROOT}")" "$(basename "${OUTPUT_ROOT}")"
sha256sum "${archive}" > "${archive_sha}"

echo "status=uploading_hf started_at=$(date --iso-8601=seconds)"
"${HF}" upload "${HF_REPO}" "${archive}" "${RUN_NAME}/$(basename "${archive}")" --repo-type dataset
"${HF}" upload "${HF_REPO}" "${archive_sha}" "${RUN_NAME}/$(basename "${archive_sha}")" --repo-type dataset
cat > "${OUTPUT_ROOT}/hf_upload_status.json" <<EOF
{
  "status": "complete",
  "repo": "${HF_REPO}",
  "remote_dir": "${RUN_NAME}",
  "archive": "$(basename "${archive}")",
  "archive_sha256": "$(cut -d ' ' -f 1 "${archive_sha}")",
  "videos": ${video_count},
  "telemetry": ${telemetry_count},
  "completed_at": "$(date --iso-8601=seconds)"
}
EOF
echo "status=complete videos=${video_count} hf=${HF_REPO}/${RUN_NAME} completed_at=$(date --iso-8601=seconds)"
