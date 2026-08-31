#!/usr/bin/env bash
set -Eeuo pipefail

# pretrained evaluation -> rollout replay -> one posttraining round -> evaluation

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
pretrain_project=${ROBONANA_PRETRAIN_PROJECT:-${repo_root}/experiments/robotwin_flux2_4b_dino_reward_success_q_from150k_plus10k}
pretrain_checkpoint=${ROBONANA_PRETRAIN_CHECKPOINT:-${pretrain_project}/models/checkpoint_epoch_1_step_160000/transformer/diffusion_pytorch_model.bin}
pretrain_config=${ROBONANA_PRETRAIN_MODEL_CONFIG:-${pretrain_project}/config.json}
model_python=${ROBONANA_MODEL_PYTHON:-/data3/hongjia/conda/envs/robonana/bin/python}
robotwin_path=${ROBOTWIN_PATH:-/workspace/hongjia/RoboTwin}
robotwin_env=${ROBOTWIN_CONDA_ENV:-/data3/hongjia/conda/envs/robotwin2}
robotwin_python=${ROBONANA_ROBOTWIN_PYTHON:-${robotwin_env}/bin/python}
dataset_root=${ROBONANA_INITIAL_DATASET_ROOT:-/workspace/datasets/fact-robotwin-v2/RoboTwin}
rollout_base=${ROBONANA_ROLLOUT_BASE:-/data3/hongjia/robonana_rollouts}
collection=${ROBONANA_ROLLOUT_COLLECTION:-hanging_mug_round0_from160k}
replay_root=${rollout_base}/${collection}
posttrain_steps=${ROBONANA_POSTTRAIN_STEPS:-1000}
posttrain_batch_size=${ROBONANA_POSTTRAIN_BATCH_SIZE:-4}
test_num=${ROBONANA_HANGING_MUG_TEST_NUM:-50}
run_root=${ROBONANA_HANGING_MUG_RUN_ROOT:-${repo_root}/outputs/hanging_mug_posttrain_round0_from160k}
pre_eval_dir=${run_root}/pretrain_eval50
posttrain_project=${ROBONANA_POSTTRAIN_PROJECT:-${repo_root}/experiments/hanging_mug_posttrain_round0_from160k}
post_eval_dir=${run_root}/posttrain_eval50
state_dir=${run_root}/state

if ! [[ ${posttrain_steps} =~ ^[1-9][0-9]*$ && ${posttrain_batch_size} =~ ^[1-9][0-9]*$ \
  && ${test_num} =~ ^[1-9][0-9]*$ ]]; then
  echo "posttrain steps, batch size, and test count must be positive integers" >&2
  exit 2
fi
for required in "${pretrain_checkpoint}" "${pretrain_config}" "${model_python}" \
  "${robotwin_env}/bin/python" "${dataset_root}/robonana_norm_stats.json"; do
  if [[ ! -f ${required} ]]; then
    echo "required file does not exist: ${required}" >&2
    exit 2
  fi
done
mkdir -p "${state_dir}" "${run_root}"

run_eval() {
  local checkpoint=$1
  local model_config=$2
  local output_dir=$3
  env \
    ROBONANA_TRAINED_CHECKPOINT="${checkpoint}" \
    ROBONANA_MODEL_CONFIG="${model_config}" \
    ROBONANA_MODEL_PYTHON="${model_python}" \
    ROBOTWIN_PATH="${robotwin_path}" \
    ROBOTWIN_CONDA_ENV="${robotwin_env}" \
    ROBONANA_ROBOTWIN_PYTHON="${robotwin_python}" \
    ROBONANA_DATASET_ROOT="${dataset_root}" \
    ROBONANA_EVAL_TASKS=hanging_mug \
    ROBONANA_EVAL_SERVER_GPUS=6 \
    ROBONANA_EVAL_SIM_GPUS=7 \
    ROBONANA_EVAL_JOBS_PER_GPU=1 \
    ROBONANA_EPISODE_CPU_FALLBACK=0 \
    ROBONANA_ROBOTWIN_STATIC_CAMERAS=head_camera \
    ROBONANA_EVAL_RUN_DIR="${output_dir}" \
    ROBONANA_PORT_BASE=18700 \
    bash "${repo_root}/scripts/eval_robotwin_all_tasks_parallel.sh" demo_clean "${test_num}"
}

if [[ ! -f ${state_dir}/pretrain_eval.done ]]; then
  run_eval "${pretrain_checkpoint}" "${pretrain_config}" "${pre_eval_dir}"
  touch "${state_dir}/pretrain_eval.done"
fi

if [[ ! -f ${state_dir}/rollout_replay.done ]]; then
  env \
    ROBONANA_MODEL_PYTHON="${model_python}" \
    ROBONANA_MODEL_CONFIG="${pretrain_config}" \
    ROBOTWIN_PATH="${robotwin_path}" \
    ROBOTWIN_CONDA_ENV="${robotwin_env}" \
    ROBONANA_ROBOTWIN_PYTHON="${robotwin_python}" \
    ROBONANA_ROLLOUT_BASE="${rollout_base}" \
    ROBONANA_INITIAL_DATASET_ROOT="${dataset_root}" \
    ROBONANA_STATS_SOURCE="${dataset_root}/robonana_norm_stats.json" \
    ROBONANA_TRAINED_CHECKPOINT="${pretrain_checkpoint}" \
    ROBONANA_COLLECTION_ROUND=0 \
    ROBONANA_POLICY_VERSION=reward_success_q_160k \
    ROBONANA_SERVER_GPU_ID=6 \
    ROBONANA_SIM_GPU_ID=7 \
    ROBONANA_PREPARE_GPU_ID=7 \
    ROBONANA_ROBOTWIN_STATIC_CAMERAS=head_camera \
    TEST_NUM="${test_num}" \
    PORT=18720 \
    bash "${repo_root}/scripts/collect_prepare_robotwin_rollouts.sh" \
      hanging_mug demo_clean "${collection}" 0
  touch "${state_dir}/rollout_replay.done"
fi

if [[ ! -f ${state_dir}/posttrain.done ]]; then
  env \
    ROBONANA_PYTHON="${model_python}" \
    ROBONANA_GPU_IDS=6,7 \
    ROBONANA_BATCH_SIZE="${posttrain_batch_size}" \
    ROBONANA_NUM_WORKERS=4 \
    ROBONANA_MAX_STEPS="${posttrain_steps}" \
    ROBONANA_REPLAY_ROOT="${replay_root}" \
    ROBONANA_POSTTRAIN_CHECKPOINT="${pretrain_checkpoint}" \
    ROBONANA_POSTTRAIN_MODEL_CONFIG="${pretrain_config}" \
    ROBONANA_COLLECTION_ROUND=0 \
    ROBONANA_PROJECT_DIR="${posttrain_project}" \
    ROBONANA_RESUME=0 \
    ROBONANA_PIXEL_EVAL_INTERVAL=0 \
    ROBONANA_CHECKPOINT_INTERVAL=100 \
    ROBONANA_EARLY_CHECKPOINT_STEPS=10 \
    WANDB_MODE=online \
    WANDB_NAME=hanging-mug-posttrain-round0-from160k \
    bash "${repo_root}/scripts/run_robotwin_train.sh" \
      --config robonana.configs.robotwin_flux2_4b_dino_posttrain.config
  touch "${state_dir}/posttrain.done"
fi

posttrain_checkpoint=$(find "${posttrain_project}/models" -path \
  "*/checkpoint_*_step_${posttrain_steps}/transformer/diffusion_pytorch_model.bin" \
  -type f -print -quit)
if [[ -z ${posttrain_checkpoint} ]]; then
  echo "posttraining finished without step ${posttrain_steps} checkpoint" >&2
  exit 1
fi
if [[ ! -f ${state_dir}/posttrain_eval.done ]]; then
  run_eval "${posttrain_checkpoint}" "${posttrain_project}/config.json" "${post_eval_dir}"
  touch "${state_dir}/posttrain_eval.done"
fi

"${model_python}" - "${pre_eval_dir}" "${post_eval_dir}" "${run_root}/comparison.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

def read_rate(root: str) -> dict[str, float | int]:
    rows = list(csv.DictReader((Path(root) / "results.csv").open(encoding="utf-8")))
    row = next(item for item in rows if item["task"] == "hanging_mug")
    return {
        "success": int(row["success"]),
        "total": int(row["total"]),
        "success_rate": float(row["success_rate"]),
    }

before = read_rate(sys.argv[1])
after = read_rate(sys.argv[2])
payload = {
    "after_posttrain": after,
    "before_posttrain": before,
    "success_rate_delta": after["success_rate"] - before["success_rate"],
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
