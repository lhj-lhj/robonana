#!/usr/bin/env bash
set -Eeuo pipefail

# Run the two serialized fixed-48 MAC phases, compare M=1 versus M=32, and
# append selected-policy trajectories to replay.  There is one FLUX checkpoint:
# phase 1 updates policy/world; phase 2 freezes it and updates only V/Q experts.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
round_id=${ROBONANA_COLLECTION_ROUND:-0}
world_policy_steps=${ROBONANA_MAC_WORLD_POLICY_STEPS:-${ROBONANA_MAC_TRAIN_STEPS:-1000}}
critic_steps=${ROBONANA_MAC_CRITIC_STEPS:-1000}
train_batch_size=${ROBONANA_MAC_BATCH_SIZE:-4}
test_num=${ROBONANA_HANGING_MUG_TEST_NUM:-50}
model_python=${ROBONANA_MODEL_PYTHON:-/data3/hongjia/conda/envs/robonana/bin/python}
robotwin_path=${ROBOTWIN_PATH:-/workspace/hongjia/RoboTwin}
robotwin_env=${ROBOTWIN_CONDA_ENV:-/data3/hongjia/conda/envs/robotwin2}
robotwin_python=${ROBONANA_ROBOTWIN_PYTHON:-${robotwin_env}/bin/python}
initial_dataset_root=${ROBONANA_INITIAL_DATASET_ROOT:-/workspace/datasets/fact-robotwin-v2/RoboTwin}
replay_root=${ROBONANA_REPLAY_ROOT:-/data3/hongjia/robonana_rollouts/hanging_mug_round0_from160k}
project_dir=${ROBONANA_PROJECT_DIR:-${repo_root}/experiments/hanging_mug_mac_round${round_id}}
run_root=${ROBONANA_MAC_RUN_ROOT:-${repo_root}/outputs/hanging_mug_mac_round${round_id}}
state_dir=${run_root}/state
train_gpu_ids=${ROBONANA_GPU_IDS:-6,7}
server_gpu_id=${ROBONANA_SERVER_GPU_ID:-6}
sim_gpu_id=${ROBONANA_SIM_GPU_ID:-7}
prepare_gpu_id=${ROBONANA_PREPARE_GPU_ID:-7}
seed_group=${ROBONANA_EVAL_SEED_GROUP:-${round_id}}

if ! [[ ${round_id} =~ ^[0-9]+$ && ${world_policy_steps} =~ ^[1-9][0-9]*$ \
  && ${critic_steps} =~ ^[1-9][0-9]*$ \
  && ${train_batch_size} =~ ^[1-9][0-9]*$ && ${test_num} =~ ^[1-9][0-9]*$ ]]; then
  echo "round must be non-negative; steps, batch size, and test count must be positive" >&2
  exit 2
fi
next_round=$((round_id + 1))

if [[ -n ${ROBONANA_MAC_INITIALIZATION:-} ]]; then
  initialization=${ROBONANA_MAC_INITIALIZATION}
elif (( round_id == 0 )); then
  initialization=mac_from_legacy
else
  initialization=trained
fi

if [[ ${initialization} == mac_from_legacy ]]; then
  source_run=${ROBONANA_SOURCE_RUN:-${repo_root}/experiments/robotwin_flux2_4b_dino_grouped_lr_A_bidir_G_causal_bs256_120k}
  source_checkpoint=${ROBONANA_MAC_SOURCE_CHECKPOINT:-${source_run}/models/checkpoint_epoch_6_step_120000/transformer/diffusion_pytorch_model.bin}
  source_config=${ROBONANA_MAC_SOURCE_CONFIG:-${source_run}/config.json}
elif [[ ${initialization} == trained ]]; then
  source_checkpoint=${ROBONANA_MAC_SOURCE_CHECKPOINT:?set the previous MAC checkpoint for round ${round_id}}
  source_config=${ROBONANA_MAC_SOURCE_CONFIG:?set the previous MAC run config.json for round ${round_id}}
else
  echo "ROBONANA_MAC_INITIALIZATION must be mac_from_legacy or trained" >&2
  exit 2
fi

# A v2 critic checkpoint stores the Value-only target beside transformer/.
# Carry it across the next round's frozen-expert world phase. The first legacy
# migration has no target file and intentionally starts from an exact online-V
# copy when its critic phase begins.
source_checkpoint_root=$(dirname "$(dirname "${source_checkpoint}")")
target_value_checkpoint=${ROBONANA_MAC_TARGET_VALUE_CHECKPOINT:-}
target_value_state=${ROBONANA_MAC_TARGET_VALUE_STATE:-}
if [[ -z ${target_value_checkpoint} && -f ${source_checkpoint_root}/target_value_expert.safetensors ]]; then
  target_value_checkpoint=${source_checkpoint_root}/target_value_expert.safetensors
fi
if [[ -z ${target_value_state} && -f ${source_checkpoint_root}/value_ema_state.json ]]; then
  target_value_state=${source_checkpoint_root}/value_ema_state.json
fi

for required in "${source_checkpoint}" "${source_config}" "${model_python}" \
  "${robotwin_python}" "${initial_dataset_root}/robonana_norm_stats.json"; do
  if [[ ! -f ${required} ]]; then
    echo "required file does not exist: ${required}" >&2
    exit 2
  fi
done
if [[ ! -d ${replay_root} ]]; then
  echo "cumulative replay root does not exist: ${replay_root}" >&2
  exit 2
fi

rollout_base=$(dirname "${replay_root}")
collection=$(basename "${replay_root}")
if [[ ! ${collection} =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "replay basename must be one safe path component: ${collection}" >&2
  exit 2
fi
mkdir -p "${state_dir}" "${run_root}"

find_trained_checkpoint() {
  local phase_project=$1
  local phase_steps=$2
  find "${phase_project}/models" -path \
    "*/checkpoint_*_step_${phase_steps}/transformer/diffusion_pytorch_model.bin" \
    -type f -print -quit 2>/dev/null || true
}

world_project=${project_dir}/world_policy
critic_project=${project_dir}/critic
world_checkpoint=$(find_trained_checkpoint "${world_project}" "${world_policy_steps}")
if [[ ! -f ${state_dir}/world_policy.done ]]; then
  if [[ -z ${world_checkpoint} ]]; then
    env \
      ROBONANA_PYTHON="${model_python}" \
      ROBONANA_GPU_IDS="${train_gpu_ids}" \
      ROBONANA_BATCH_SIZE="${train_batch_size}" \
      ROBONANA_NUM_WORKERS="${ROBONANA_NUM_WORKERS:-4}" \
      ROBONANA_MAX_STEPS="${world_policy_steps}" \
      ROBONANA_REPLAY_ROOT="${replay_root}" \
      ROBONANA_COLLECTION_ROUND="${round_id}" \
      ROBONANA_PROJECT_DIR="${world_project}" \
      ROBONANA_MAC_PHASE=world_policy \
      ROBONANA_MAC_INITIALIZATION="${initialization}" \
      ROBONANA_MAC_PRETRAIN_CHECKPOINT="${source_checkpoint}" \
      ROBONANA_MAC_PRETRAIN_CONFIG="${source_config}" \
      ROBONANA_RESUME="${ROBONANA_RESUME:-1}" \
      ROBONANA_CHECKPOINT_INTERVAL="${ROBONANA_CHECKPOINT_INTERVAL:-100}" \
      ROBONANA_EARLY_CHECKPOINT_STEPS="${ROBONANA_EARLY_CHECKPOINT_STEPS:-10}" \
      WANDB_MODE="${WANDB_MODE:-online}" \
      WANDB_NAME="${WANDB_NAME:-hanging-mug-mac-round${round_id}-world-policy}" \
      bash "${repo_root}/scripts/run_robotwin_train.sh" \
        --config robonana.configs.robotwin_flux2_4b_mac_from120k.config
    world_checkpoint=$(find_trained_checkpoint "${world_project}" "${world_policy_steps}")
  fi
  if [[ -z ${world_checkpoint} ]]; then
    echo "world/policy training did not produce a step-${world_policy_steps} checkpoint" >&2
    exit 1
  fi
  touch "${state_dir}/world_policy.done"
fi
if [[ -z ${world_checkpoint} ]]; then
  echo "world_policy.done exists but its checkpoint is missing" >&2
  exit 1
fi
world_config=${world_project}/config.json
if [[ ! -f ${world_config} ]]; then
  echo "world/policy config is missing: ${world_config}" >&2
  exit 1
fi

trained_checkpoint=$(find_trained_checkpoint "${critic_project}" "${critic_steps}")
if [[ ! -f ${state_dir}/critic.done ]]; then
  if [[ -z ${trained_checkpoint} ]]; then
    env \
      ROBONANA_PYTHON="${model_python}" \
      ROBONANA_GPU_IDS="${train_gpu_ids}" \
      ROBONANA_BATCH_SIZE="${train_batch_size}" \
      ROBONANA_NUM_WORKERS="${ROBONANA_NUM_WORKERS:-4}" \
      ROBONANA_MAX_STEPS="${critic_steps}" \
      ROBONANA_REPLAY_ROOT="${replay_root}" \
      ROBONANA_COLLECTION_ROUND="${round_id}" \
      ROBONANA_PROJECT_DIR="${critic_project}" \
      ROBONANA_MAC_PHASE=critic \
      ROBONANA_MAC_INITIALIZATION=trained \
      ROBONANA_MAC_PRETRAIN_CHECKPOINT="${world_checkpoint}" \
      ROBONANA_MAC_PRETRAIN_CONFIG="${world_config}" \
      ROBONANA_MAC_TARGET_VALUE_CHECKPOINT="${target_value_checkpoint}" \
      ROBONANA_MAC_TARGET_VALUE_STATE="${target_value_state}" \
      ROBONANA_RESUME="${ROBONANA_RESUME:-1}" \
      ROBONANA_CHECKPOINT_INTERVAL="${ROBONANA_CHECKPOINT_INTERVAL:-100}" \
      ROBONANA_EARLY_CHECKPOINT_STEPS="${ROBONANA_EARLY_CHECKPOINT_STEPS:-10}" \
      WANDB_MODE="${WANDB_MODE:-online}" \
      WANDB_NAME="${WANDB_NAME:-hanging-mug-mac-round${round_id}-critic}" \
      bash "${repo_root}/scripts/run_robotwin_train.sh" \
        --config robonana.configs.robotwin_flux2_4b_mac_from120k.config
    trained_checkpoint=$(find_trained_checkpoint "${critic_project}" "${critic_steps}")
  fi
  if [[ -z ${trained_checkpoint} ]]; then
    echo "critic training did not produce a step-${critic_steps} checkpoint" >&2
    exit 1
  fi
  touch "${state_dir}/critic.done"
fi
if [[ -z ${trained_checkpoint} ]]; then
  echo "critic.done exists but its checkpoint is missing" >&2
  exit 1
fi
trained_config=${critic_project}/config.json
if [[ ! -f ${trained_config} ]]; then
  echo "critic config is missing: ${trained_config}" >&2
  exit 1
fi

run_ranked_eval() {
  local candidate_count=$1
  local output_dir=$2
  env \
    ROBONANA_TRAINED_CHECKPOINT="${trained_checkpoint}" \
    ROBONANA_MODEL_CONFIG="${trained_config}" \
    ROBONANA_MODEL_PYTHON="${model_python}" \
    ROBOTWIN_PATH="${robotwin_path}" \
    ROBOTWIN_CONDA_ENV="${robotwin_env}" \
    ROBONANA_ROBOTWIN_PYTHON="${robotwin_python}" \
    ROBONANA_DATASET_ROOT="${initial_dataset_root}" \
    ROBONANA_EVAL_TASKS=hanging_mug \
    ROBONANA_EVAL_SERVER_GPUS="${server_gpu_id}" \
    ROBONANA_EVAL_SIM_GPUS="${sim_gpu_id}" \
    ROBONANA_EVAL_JOBS_PER_GPU=1 \
    ROBONANA_EPISODE_CPU_FALLBACK=0 \
    ROBONANA_ROBOTWIN_STATIC_CAMERAS=head_camera \
    ROBONANA_EVAL_RUN_DIR="${output_dir}" \
    ROBONANA_EVAL_SEED_GROUP="${seed_group}" \
    ROBONANA_INFERENCE_MODE=action_q_rejection \
    ROBONANA_REJECTION_CANDIDATE_COUNT="${candidate_count}" \
    ROBONANA_Q_RETURN_SCALE="${ROBONANA_MAC_RETURN_SCALE:-1000}" \
    ROBONANA_PORT_BASE="${ROBONANA_PORT_BASE:-18700}" \
    EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-1}" \
    bash "${repo_root}/scripts/eval_robotwin_all_tasks_parallel.sh" \
      demo_clean "${test_num}"
}

m1_eval_dir=${run_root}/m1_eval
if [[ ! -f ${state_dir}/m1_eval.done ]]; then
  run_ranked_eval 1 "${m1_eval_dir}"
  touch "${state_dir}/m1_eval.done"
fi

m32_eval_dir=${run_root}/m32_selected_collection
if [[ ! -f ${state_dir}/m32_collection.done ]]; then
  env \
    ROBONANA_MODEL_PYTHON="${model_python}" \
    ROBONANA_MODEL_CONFIG="${trained_config}" \
    ROBOTWIN_PATH="${robotwin_path}" \
    ROBOTWIN_CONDA_ENV="${robotwin_env}" \
    ROBONANA_ROBOTWIN_PYTHON="${robotwin_python}" \
    ROBONANA_ROLLOUT_BASE="${rollout_base}" \
    ROBONANA_INITIAL_DATASET_ROOT="${initial_dataset_root}" \
    ROBONANA_STATS_SOURCE="${initial_dataset_root}/robonana_norm_stats.json" \
    ROBONANA_TRAINED_CHECKPOINT="${trained_checkpoint}" \
    ROBONANA_COLLECTION_ROUND="${next_round}" \
    ROBONANA_POLICY_VERSION="mac_round${round_id}_m32" \
    ROBONANA_SERVER_GPU_ID="${server_gpu_id}" \
    ROBONANA_SIM_GPU_ID="${sim_gpu_id}" \
    ROBONANA_PREPARE_GPU_ID="${prepare_gpu_id}" \
    ROBONANA_ROBOTWIN_STATIC_CAMERAS=head_camera \
    ROBONANA_EVAL_RUN_DIR="${m32_eval_dir}" \
    ROBONANA_EVAL_SEED_GROUP="${seed_group}" \
    ROBONANA_INFERENCE_MODE=action_q_rejection \
    ROBONANA_REJECTION_CANDIDATE_COUNT="${ROBONANA_MAC_EVAL_CANDIDATES:-32}" \
    ROBONANA_Q_RETURN_SCALE="${ROBONANA_MAC_RETURN_SCALE:-1000}" \
    EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-1}" \
    TEST_NUM="${test_num}" \
    PORT="${ROBONANA_COLLECTION_PORT:-18720}" \
    bash "${repo_root}/scripts/collect_prepare_robotwin_rollouts.sh" \
      hanging_mug demo_clean "${collection}" "${seed_group}"
  touch "${state_dir}/m32_collection.done"
fi

"${model_python}" - "${m1_eval_dir}" "${m32_eval_dir}" \
  "${run_root}/comparison.json" <<'PY'
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


m1 = read_rate(sys.argv[1])
m32 = read_rate(sys.argv[2])
payload = {
    "m1": m1,
    "m32_q_rejection": m32,
    "success_rate_delta": m32["success_rate"] - m1["success_rate"],
}
Path(sys.argv[3]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY

echo "Round ${round_id} complete. Selected M=32 trajectories were appended as round ${next_round}."
echo "Next round source checkpoint: ${trained_checkpoint}"
echo "Next round source config: ${trained_config}"
