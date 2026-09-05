#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
task_name=${1:?"usage: $0 TASK TASK_CONFIG COLLECTION [SEED]"}
task_config=${2:?"usage: $0 TASK TASK_CONFIG COLLECTION [SEED]"}
collection=${3:?"usage: $0 TASK TASK_CONFIG COLLECTION [SEED]"}
seed=${4:-0}
if [[ ! "${collection}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "COLLECTION must be one safe path component: ${collection}" >&2
  exit 2
fi

rollout_base=${ROBONANA_ROLLOUT_BASE:-/workspace/hongjia/robonana_rollouts}
dataset_root="${rollout_base}/${collection}"
initial_dataset_root=${ROBONANA_INITIAL_DATASET_ROOT:-/workspace/datasets/fact-robotwin-v2/RoboTwin}
flux_checkpoint=${ROBONANA_FLUX_CHECKPOINT_DIR:-${repo_root}/checkpoints/FLUX.2-klein-base-4B}
trained_checkpoint=${ROBONANA_TRAINED_CHECKPOINT:?"set ROBONANA_TRAINED_CHECKPOINT to a complete diffusion_pytorch_model.bin"}
stats_source=${ROBONANA_STATS_SOURCE:-${initial_dataset_root}/robonana_norm_stats.json}
model_python=${ROBONANA_MODEL_PYTHON:-${repo_root}/.venv/bin/python}
robotwin_path=${ROBOTWIN_PATH:-/workspace/hongjia/RoboTwin}
robotwin_env=${ROBOTWIN_CONDA_ENV:-/workspace/.conda/envs/robotwin2}
robotwin_python=${ROBONANA_ROBOTWIN_PYTHON:-${robotwin_env}/bin/python}
fact_conda_env=${FACT_CONDA_ENV:-$(dirname "$(dirname "${model_python}")")}
deploy_policy=${ROBONANA_DEPLOY_POLICY_PATH:-${repo_root}/src/robonana/configs/robotwin_eval_train_seen.yml}
isolated_eval=${repo_root}/scripts/eval_robotwin_all_tasks_parallel.sh
server_gpu_id=${ROBONANA_SERVER_GPU_ID:-6}
sim_gpu_id=${ROBONANA_SIM_GPU_ID:-7}
prepare_gpu_id=${ROBONANA_PREPARE_GPU_ID:-7}
static_camera_csv=${ROBONANA_ROBOTWIN_STATIC_CAMERAS:-head_camera}
port=${PORT:-8094}
test_num=${TEST_NUM:-1}
eval_run_dir=${ROBONANA_EVAL_RUN_DIR:-${dataset_root}/logs/isolated_collection}
video_log=${EVAL_VIDEO_LOG:-0}

count_dataset_episodes() {
  local data_dir="${dataset_root}/${task_name}/robonana_rollout/data"
  if [[ ! -d ${data_dir} ]]; then
    printf '0\n'
    return
  fi
  find "${data_dir}" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l
}

count_completed_eval_episodes() {
  local ledger rows
  local total=0
  if [[ ! -d ${eval_run_dir} ]]; then
    printf '0\n'
    return
  fi
  while IFS= read -r -d '' ledger; do
    rows=$(grep -cve '^[[:space:]]*$' "${ledger}" || true)
    total=$((total + rows))
  done < <(
    find "${eval_run_dir}" -type f \
      -path "*/task_runs/${task_name}/episodes.jsonl" -print0
  )
  printf '%d\n' "${total}"
}

for required_path in "${trained_checkpoint}" "${stats_source}" "${model_python}" \
  "${robotwin_python}" "${deploy_policy}" "${isolated_eval}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required file does not exist: ${required_path}" >&2
    exit 2
  fi
done
if [[ ! -d "${flux_checkpoint}" ]]; then
  echo "FLUX checkpoint directory does not exist: ${flux_checkpoint}" >&2
  exit 2
fi
if [[ -n "${ROBONANA_SERVER_GPU_IDS:-}" || -n "${ROBONANA_CLIENT_GPU_ID:-}" ]]; then
  echo "ROBONANA_SERVER_GPU_IDS/ROBONANA_CLIENT_GPU_ID are obsolete." >&2
  echo "Set disjoint ROBONANA_SERVER_GPU_ID and ROBONANA_SIM_GPU_ID instead." >&2
  exit 2
fi
if ! [[ ${server_gpu_id} =~ ^[0-9]+$ && ${sim_gpu_id} =~ ^[0-9]+$ \
  && ${prepare_gpu_id} =~ ^[0-9]+$ ]]; then
  echo "server, simulator, and preparation GPU IDs must be non-negative integers" >&2
  exit 2
fi
if [[ ${server_gpu_id} == "${sim_gpu_id}" ]]; then
  echo "policy server and SAPIEN simulator GPUs must be disjoint" >&2
  exit 2
fi

mkdir -p "${dataset_root}/logs"
episode_count_before=$(count_dataset_episodes)
completed_eval_before=$(count_completed_eval_episodes)
if (( completed_eval_before > test_num )); then
  echo "evaluation ledger has ${completed_eval_before} rows, target is ${test_num}" >&2
  exit 2
fi
expected_new_episodes=$((test_num - completed_eval_before))
env \
  ROBONANA_TRAINED_CHECKPOINT="${trained_checkpoint}" \
  ROBONANA_STATS_PATH="${stats_source}" \
  ROBONANA_MODEL_PYTHON="${model_python}" \
  ROBONANA_MODEL_CONFIG="${ROBONANA_MODEL_CONFIG:-}" \
  ROBONANA_FLUX_CHECKPOINT_DIR="${flux_checkpoint}" \
  FACT_CONDA_ENV="${fact_conda_env}" \
  ROBOTWIN_PATH="${robotwin_path}" \
  ROBOTWIN_CONDA_ENV="${robotwin_env}" \
  ROBONANA_ROBOTWIN_PYTHON="${robotwin_python}" \
  DEPLOY_POLICY_PATH="${deploy_policy}" \
  ROBONANA_DATASET_ROOT="${initial_dataset_root}" \
  ROBONANA_EVAL_TASKS="${task_name}" \
  ROBONANA_EVAL_SERVER_GPUS="${server_gpu_id}" \
  ROBONANA_EVAL_SIM_GPUS="${sim_gpu_id}" \
  ROBONANA_EVAL_JOBS_PER_GPU=1 \
  ROBONANA_EPISODE_CPU_FALLBACK=0 \
  ROBONANA_EVAL_SEED_GROUP="${seed}" \
  ROBONANA_EVAL_RUN_DIR="${eval_run_dir}" \
  ROBONANA_PORT_BASE="${port}" \
  EVAL_VIDEO_LOG="${video_log}" \
  ROBONANA_ROBOTWIN_STATIC_CAMERAS="${static_camera_csv}" \
  ROBONANA_ROLLOUT_DATASET_ROOT="${dataset_root}" \
  ROBONANA_INITIAL_DATASET_ROOT="${initial_dataset_root}" \
  ROBONANA_ROLLOUT_CHECKPOINT="${trained_checkpoint}" \
  bash "${isolated_eval}" "${task_config}" "${test_num}"

episode_count=$(count_dataset_episodes)
expected_episode_count=$((episode_count_before + expected_new_episodes))
if [[ ${episode_count} -ne ${expected_episode_count} ]]; then
  echo "isolated collection has ${episode_count} episodes; expected " \
    "${episode_count_before} existing + ${expected_new_episodes} resumed/new" >&2
  exit 1
fi

env \
  CUDA_VISIBLE_DEVICES="${prepare_gpu_id}" \
  PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2/src:${repo_root}/third_party/flux2_official/src" \
  "${model_python}" "${repo_root}/scripts/prepare_robotwin_rollouts.py" \
    --dataset-root "${dataset_root}" \
    --initial-dataset-root "${initial_dataset_root}" \
    --checkpoint "${flux_checkpoint}" \
    --stats-source "${stats_source}" \
    --device cuda:0 \
    --batch-size "${ROBONANA_PREPARE_BATCH_SIZE:-16}"

echo "Prepared separate rollout dataset: ${dataset_root}"
