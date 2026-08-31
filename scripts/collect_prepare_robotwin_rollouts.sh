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
client_python_wrapper=${repo_root}/scripts/robotwin_eval_python.sh
deploy_policy=${ROBONANA_DEPLOY_POLICY_PATH:-${repo_root}/src/robonana/configs/robotwin_eval_train_seen.yml}
server_gpu_id=${ROBONANA_SERVER_GPU_ID:-6}
sim_gpu_id=${ROBONANA_SIM_GPU_ID:-7}
prepare_gpu_id=${ROBONANA_PREPARE_GPU_ID:-7}
static_camera_csv=${ROBONANA_ROBOTWIN_STATIC_CAMERAS:-head_camera}
port=${PORT:-8094}
test_num=${TEST_NUM:-1}

for required_path in "${trained_checkpoint}" "${stats_source}" "${model_python}" \
  "${robotwin_python}" "${client_python_wrapper}" "${deploy_policy}"; do
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
server_log="${dataset_root}/logs/inference_server.log"
runtime_dir="/tmp/robonana_robotwin_${collection}_${port}"
mkdir -p "${runtime_dir}"

cleanup_server() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" 2>/dev/null || true
  fi
}
cleanup() {
  cleanup_server
  rmdir "${runtime_dir}" 2>/dev/null || true
}
trap cleanup EXIT

server_args=(
  "${model_python}" "${repo_root}/scripts/inference_server_robotwin.py"
  --checkpoint "${trained_checkpoint}"
  --flux-checkpoint-dir "${flux_checkpoint}"
  --stats-path "${stats_source}"
  --model-device cuda:0
  --vae-device cuda:0
  --text-encoder-device cuda:0
  --dtype bf16
  --action-chunk 48
  --horizon 24
  --num-inference-steps 20
  --port "${port}"
)
if [[ -n "${ROBONANA_MODEL_CONFIG:-}" ]]; then
  server_args+=(--model-config "${ROBONANA_MODEL_CONFIG}")
fi

env \
  CUDA_VISIBLE_DEVICES="${server_gpu_id}" \
  PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2/src:${repo_root}/third_party/flux2_official/src" \
  "${server_args[@]}" \
    >"${server_log}" 2>&1 &
server_pid=$!

env \
  CUDA_VISIBLE_DEVICES="${sim_gpu_id}" \
  OIDN_DEFAULT_DEVICE=cuda \
  XDG_RUNTIME_DIR="${runtime_dir}" \
  PYTHONPATH="${repo_root}/src" \
  FACT_CONDA_ENV="${fact_conda_env}" \
  ROBOTWIN_PATH="${robotwin_path}" \
  ROBOTWIN_CONDA_ENV="${robotwin_env}" \
  ROBONANA_ROBOTWIN_PYTHON="${robotwin_python}" \
  CLIENT_PYTHON="${client_python_wrapper}" \
  DEPLOY_POLICY_PATH="${deploy_policy}" \
  POLICY_NAME=robonana_robotwin.adapter \
  PORT="${port}" \
  TEST_NUM="${test_num}" \
  EXECUTE_ACTIONS_PER_PLAN="${EXECUTE_ACTIONS_PER_PLAN:-48}" \
  SERVER_TIMEOUT_MS="${SERVER_TIMEOUT_MS:-600000}" \
  SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}" \
  EVAL_VIDEO_LOG=0 \
  PYTHONUNBUFFERED=1 \
  LOW_FREQUENCY_RGB=0 \
  SKIP_ACTION_RENDER_SYNC=0 \
  ROBONANA_ROBOTWIN_STATIC_CAMERAS="${static_camera_csv}" \
  ROBONANA_SAPIEN_RENDER_DEVICE=cuda:0 \
  ROBONANA_ROLLOUT_DATASET_ROOT="${dataset_root}" \
  ROBONANA_INITIAL_DATASET_ROOT="${initial_dataset_root}" \
  ROBONANA_ROLLOUT_CHECKPOINT="${trained_checkpoint}" \
  bash "${repo_root}/third_party/FACT/evaluation/robotwin/launch_client.sh" \
    "${task_name}" "${task_config}" "${collection}" "${seed}"

cleanup_server
server_pid=""
rmdir "${runtime_dir}" 2>/dev/null || true
trap - EXIT

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
