#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
task_config=${1:-${TASK_CONFIG:-demo_clean}}
test_num=${2:-${TEST_NUM:-50}}
robotwin_path=${ROBOTWIN_PATH:-/workspace/hongjia/RoboTwin}
dataset_root=${ROBONANA_DATASET_ROOT:-/workspace/datasets/fact-robotwin-v2/RoboTwin}
checkpoint=${ROBONANA_TRAINED_CHECKPOINT:?set ROBONANA_TRAINED_CHECKPOINT to diffusion_pytorch_model.bin}
flux_checkpoint=${ROBONANA_FLUX_CHECKPOINT_DIR:-${repo_root}/checkpoints/FLUX.2-klein-base-4B}
stats_path=${ROBONANA_STATS_PATH:-${dataset_root}/robonana_norm_stats.json}
model_python=${ROBONANA_MODEL_PYTHON:-/data3/hongjia/conda/envs/robonana/bin/python}
robotwin_env=${ROBOTWIN_CONDA_ENV:-/data3/hongjia/conda/envs/robotwin2}
gpu_csv=${ROBONANA_EVAL_GPUS:-0,1,2,3,4,5,6,7}
port_base=${ROBONANA_PORT_BASE:-18000}
run_dir=${ROBONANA_EVAL_RUN_DIR:-${repo_root}/outputs/robotwin_full_eval_$(date +%Y%m%d_%H%M%S)}
deploy_policy=${ROBONANA_DEPLOY_POLICY_PATH:-${repo_root}/src/robonana/configs/robotwin_eval_train_seen.yml}

IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
if [[ ${#gpu_ids[@]} -eq 0 ]]; then
  echo "ROBONANA_EVAL_GPUS resolved to an empty GPU list" >&2
  exit 2
fi
for required in "${checkpoint}" "${stats_path}" "${deploy_policy}" "${model_python}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done
if [[ ! -d "${flux_checkpoint}" || ! -d "${robotwin_env}" ]]; then
  echo "Missing FLUX checkpoint or RoboTwin environment" >&2
  exit 2
fi

step_limits="${robotwin_path}/task_config/_eval_step_limit.yml"
mapfile -t tasks < <(grep -oE '^[a-z0-9_]+:' "${step_limits}" | tr -d ':')
if [[ ${#tasks[@]} -ne 50 ]]; then
  echo "Expected exactly 50 RoboTwin eval tasks, found ${#tasks[@]}" >&2
  exit 2
fi

mkdir -p "${run_dir}/workers" "${run_dir}/value_traces"
touch "${run_dir}/.started"
"${model_python}" "${repo_root}/scripts/audit_robotwin_instructions.py" \
  --dataset-root "${dataset_root}" \
  --robotwin-root "${robotwin_path}" \
  --output "${run_dir}/instruction_audit.json" \
  --require-seen \
  > "${run_dir}/instruction_audit.log"

run_worker() {
  local rank=$1
  local gpu=$2
  local port=$((port_base + rank))
  local worker_dir="${run_dir}/workers/rank_${rank}_gpu_${gpu}"
  local runtime_dir="/tmp/robonana_eval_${USER}_${port}"
  local shard=()
  local index
  for index in "${!tasks[@]}"; do
    if (( index % ${#gpu_ids[@]} == rank )); then
      shard+=("${tasks[index]}")
    fi
  done
  mkdir -p "${worker_dir}" "${runtime_dir}"

  local server_args=(
    "${model_python}" "${repo_root}/scripts/inference_server_robotwin.py"
    --checkpoint "${checkpoint}"
    --flux-checkpoint-dir "${flux_checkpoint}"
    --stats-path "${stats_path}"
    --model-device cuda:0
    --vae-device cuda:0
    --text-encoder-device cuda:0
    --dtype bf16
    --action-chunk 48
    --horizon 24
    --num-inference-steps 20
    --return-chunk-value
    --port "${port}"
  )
  if [[ -n "${ROBONANA_MODEL_CONFIG:-}" ]]; then
    server_args+=(--model-config "${ROBONANA_MODEL_CONFIG}")
  fi

  local server_pid=""
  cleanup_worker() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      kill "${server_pid}"
      wait "${server_pid}" 2>/dev/null || true
    fi
    rmdir "${runtime_dir}" 2>/dev/null || true
  }
  trap cleanup_worker EXIT
  trap 'exit 130' INT TERM

  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2_official/src:${repo_root}/third_party/flux2/src" \
    "${server_args[@]}" \
    > "${worker_dir}/server.log" 2>&1 &
  server_pid=$!

  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    XDG_RUNTIME_DIR="${runtime_dir}" \
    PYTHONPATH="${repo_root}/src" \
    ROBOTWIN_PATH="${robotwin_path}" \
    ROBOTWIN_CONDA_ENV="${robotwin_env}" \
    DEPLOY_POLICY_PATH="${deploy_policy}" \
    POLICY_NAME=robonana_robotwin.adapter \
    PORT="${port}" \
    TEST_NUM="${test_num}" \
    EXECUTE_ACTIONS_PER_PLAN=48 \
    SERVER_TIMEOUT_MS=600000 \
    SERVER_WAIT_SECONDS=600 \
    EVAL_VIDEO_LOG=1 \
    TRACE_ROOT="${run_dir}/value_traces" \
    ENABLE_VALUE_VIS=1 \
    TRACE_VALUE_ONLY=1 \
    LOW_FREQUENCY_RGB=0 \
    SKIP_ACTION_RENDER_SYNC=0 \
    ROBONANA_OVERLAY_CHUNK_VALUE=1 \
    TASK_LIST="${shard[*]}" \
    SWEEP_OUT="${worker_dir}/sweep" \
    bash "${repo_root}/third_party/FACT/evaluation/robotwin/eval_all_tasks.sh" \
      "${task_config}" "${test_num}"
}

declare -a worker_pids=()
cleanup_all_workers() {
  local pid
  for pid in "${worker_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${worker_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup_all_workers EXIT INT TERM
for rank in "${!gpu_ids[@]}"; do
  run_worker "${rank}" "${gpu_ids[rank]}" \
    > "${run_dir}/workers/rank_${rank}.log" 2>&1 &
  worker_pids+=("$!")
done

worker_status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    worker_status=1
  fi
done
worker_pids=()
trap - EXIT INT TERM

results_csv="${run_dir}/results.csv"
results_body="${run_dir}/results.body"
: > "${results_body}"
for rank in "${!gpu_ids[@]}"; do
  worker_csv="${run_dir}/workers/rank_${rank}_gpu_${gpu_ids[rank]}/sweep/results.csv"
  if [[ -f "${worker_csv}" ]]; then
    tail -n +2 "${worker_csv}" >> "${results_body}"
  fi
done
echo "task,success,total,success_rate" > "${results_csv}"
sort -t, -k1,1 "${results_body}" >> "${results_csv}"
rm -f "${results_body}"

awk -F, '
  NR>1 && $4!="ERROR" {success+=$2; total+=$3; macro+=$4; tasks++}
  NR>1 && $4=="ERROR" {errors++}
  END {
    printf "tasks=%d errors=%d\n", tasks, errors
    printf "micro_success_rate=%.6f (%d/%d)\n", total ? success/total : 0, success, total
    printf "macro_success_rate=%.6f\n", tasks ? macro/tasks : 0
  }
' "${results_csv}" | tee "${run_dir}/summary.txt"

find "${robotwin_path}/eval_result" -type f -name '*.mp4' -newer "${run_dir}/.started" -print \
  | sort > "${run_dir}/mp4_manifest.txt"
find "${run_dir}/value_traces" -type f -name '*.npz' -print \
  | sort > "${run_dir}/value_trace_manifest.txt"

result_tasks=$(awk 'END {print NR-1}' "${results_csv}")
mp4_count=$(wc -l < "${run_dir}/mp4_manifest.txt")
value_trace_count=$(wc -l < "${run_dir}/value_trace_manifest.txt")
expected_episodes=$((50 * test_num))
{
  echo "result_tasks=${result_tasks}/50"
  echo "annotated_mp4=${mp4_count}/${expected_episodes}"
  echo "value_traces=${value_trace_count}/${expected_episodes}"
} | tee -a "${run_dir}/summary.txt"

if [[ ${worker_status} -ne 0 ]] \
  || grep -q ',ERROR$' "${results_csv}" \
  || [[ ${result_tasks} -ne 50 ]] \
  || [[ ${mp4_count} -ne ${expected_episodes} ]] \
  || [[ ${value_trace_count} -ne ${expected_episodes} ]]; then
  echo "One or more eval workers/tasks failed; inspect ${run_dir}/workers" >&2
  exit 1
fi
echo "RoboTwin full-task eval complete: ${run_dir}"
