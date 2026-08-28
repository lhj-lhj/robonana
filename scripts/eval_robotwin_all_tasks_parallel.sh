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
fact_conda_env=${FACT_CONDA_ENV:-$(dirname "$(dirname "${model_python}")")}
gpu_csv=${ROBONANA_EVAL_GPUS:-0,1,2,3,4,5,6,7}
sapien_denoiser=${ROBONANA_SAPIEN_DENOISER:-optix}
port_base=${ROBONANA_PORT_BASE:-18000}
run_dir=${ROBONANA_EVAL_RUN_DIR:-${repo_root}/outputs/robotwin_full_eval_$(date +%Y%m%d_%H%M%S)}
deploy_policy=${ROBONANA_DEPLOY_POLICY_PATH:-${repo_root}/src/robonana/configs/robotwin_eval_train_seen.yml}
task_timeout_seconds=${ROBONANA_TASK_TIMEOUT_SECONDS:-21600}
task_max_attempts=${ROBONANA_TASK_MAX_ATTEMPTS:-3}
jobs_per_gpu=${ROBONANA_EVAL_JOBS_PER_GPU:-1}
batch_wait_ms=${ROBONANA_EVAL_BATCH_WAIT_MS:-6}
aux_outputs=${ROBONANA_EVAL_AUX_OUTPUTS:-1}
client_python_wrapper=${repo_root}/scripts/robotwin_eval_python.sh

IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
if [[ ${#gpu_ids[@]} -eq 0 ]]; then
  echo "ROBONANA_EVAL_GPUS resolved to an empty GPU list" >&2
  exit 2
fi
if [[ "${sapien_denoiser}" != "oidn" && "${sapien_denoiser}" != "optix" \
  && "${sapien_denoiser}" != "none" ]]; then
  echo "ROBONANA_SAPIEN_DENOISER must be one of: oidn, optix, none" >&2
  exit 2
fi
for required in "${checkpoint}" "${stats_path}" "${deploy_policy}" "${model_python}" \
  "${robotwin_env}/bin/python" "${client_python_wrapper}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done
if [[ ! -d "${flux_checkpoint}" || ! -d "${robotwin_env}" ]]; then
  echo "Missing FLUX checkpoint or RoboTwin environment" >&2
  exit 2
fi
if ! [[ ${task_timeout_seconds} =~ ^[1-9][0-9]*$ && ${task_max_attempts} =~ ^[1-9][0-9]*$ \
  && ${jobs_per_gpu} =~ ^[1-9][0-9]*$ ]]; then
  echo "Timeout, max attempts, and ROBONANA_EVAL_JOBS_PER_GPU must be positive integers" >&2
  exit 2
fi
if [[ "${aux_outputs}" != "0" && "${aux_outputs}" != "1" ]]; then
  echo "ROBONANA_EVAL_AUX_OUTPUTS must be 0 or 1" >&2
  exit 2
fi
if (( jobs_per_gpu > 1 )) && [[ "${aux_outputs}" != "0" ]]; then
  echo "Dynamic batching is Stage-1-only; set ROBONANA_EVAL_AUX_OUTPUTS=0" >&2
  exit 2
fi

step_limits="${robotwin_path}/task_config/_eval_step_limit.yml"
mapfile -t tasks < <(grep -oE '^[a-z0-9_]+:' "${step_limits}" | tr -d ':')
if [[ ${#tasks[@]} -ne 50 ]]; then
  echo "Expected exactly 50 RoboTwin eval tasks, found ${#tasks[@]}" >&2
  exit 2
fi

mkdir -p "${run_dir}/workers" "${run_dir}/value_traces" "${run_dir}/stage2_images"
[[ -e "${run_dir}/.started" ]] || touch "${run_dir}/.started"
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
  local sweep_dir="${worker_dir}/sweep"
  local results_csv="${sweep_dir}/results.csv"
  mkdir -p "${worker_dir}" "${runtime_dir}" "${sweep_dir}/logs" "${worker_dir}/attempts"
  [[ -f "${results_csv}" ]] || echo "task,success,total,success_rate" > "${results_csv}"

  local server_script="${repo_root}/scripts/inference_server_robotwin.py"
  if (( jobs_per_gpu > 1 )); then
    server_script="${repo_root}/scripts/inference_server_robotwin_batched.py"
  fi
  local server_args=(
    "${model_python}" "${server_script}"
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
    --port "${port}"
  )
  if (( jobs_per_gpu > 1 )); then
    server_args+=(
      --max-batch-size "${jobs_per_gpu}"
      --max-batch-wait-ms "${batch_wait_ms}"
      --max-clients "$((jobs_per_gpu * 2))"
    )
  elif [[ "${aux_outputs}" == "1" ]]; then
    server_args+=(--return-chunk-q --return-stage2-image)
  fi
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

  local client_env=(
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "XDG_RUNTIME_DIR=${runtime_dir}"
    "PYTHONPATH=${repo_root}/src"
    "ROBOTWIN_PATH=${robotwin_path}"
    "ROBOTWIN_CONDA_ENV=${robotwin_env}"
    "ROBONANA_ROBOTWIN_PYTHON=${robotwin_env}/bin/python"
    "CLIENT_PYTHON=${client_python_wrapper}"
    "FACT_CONDA_ENV=${fact_conda_env}"
    "DEPLOY_POLICY_PATH=${deploy_policy}"
    "POLICY_NAME=robonana_robotwin.adapter"
    "PORT=${port}"
    "TEST_NUM=${test_num}"
    "EXECUTE_ACTIONS_PER_PLAN=48"
    "SERVER_TIMEOUT_MS=600000"
    "SERVER_WAIT_SECONDS=600"
    "EVAL_VIDEO_LOG=1"
    "TRACE_ROOT=${run_dir}/value_traces"
    "ENABLE_VALUE_VIS=${aux_outputs}"
    "TRACE_VALUE_ONLY=1"
    "LOW_FREQUENCY_RGB=0"
    "SKIP_ACTION_RENDER_SYNC=0"
    "ROBONANA_OVERLAY_CHUNK_RETURN=${aux_outputs}"
    "ROBONANA_STAGE2_IMAGE_ROOT=${run_dir}/stage2_images"
    # cuda:0 is this rank's sole logical device after CUDA_VISIBLE_DEVICES isolation.
    "ROBONANA_SAPIEN_RENDER_DEVICE=cuda:0"
    "ROBONANA_SAPIEN_DENOISER=${sapien_denoiser}"
  )

  upsert_result() {
    local task_name=$1
    local row=$2
    (
      flock -x 9
      local temporary="${results_csv}.tmp.${BASHPID}"
      awk -F, -v task_name="${task_name}" 'NR == 1 || $1 != task_name' \
        "${results_csv}" > "${temporary}"
      printf '%s\n' "${row}" >> "${temporary}"
      mv "${temporary}" "${results_csv}"
    ) 9>"${results_csv}.lock"
  }

  run_task() {
    local task_name=$1
    local existing
    existing=$(awk -F, -v task_name="${task_name}" \
      '$1 == task_name && $4 != "ERROR" && $3 != "" {print; exit}' "${results_csv}")
    if [[ -n "${existing}" ]]; then
      echo "[resume-skip] ${task_name}: ${existing}"
      return 0
    fi

    local attempt attempt_dir attempt_csv attempt_log row attempt_rc
    for ((attempt = 1; attempt <= task_max_attempts; attempt++)); do
      attempt_dir="${worker_dir}/attempts/${task_name}/attempt_${attempt}_$(date +%Y%m%d_%H%M%S)"
      mkdir -p "${attempt_dir}/runtime"
      echo "[attempt ${attempt}/${task_max_attempts}] ${task_name} timeout=${task_timeout_seconds}s"
      attempt_rc=0
      timeout --signal=TERM --kill-after=60 "${task_timeout_seconds}" \
        env "${client_env[@]}" \
          XDG_RUNTIME_DIR="${attempt_dir}/runtime" \
          TASK_LIST="${task_name}" \
          SWEEP_OUT="${attempt_dir}" \
          bash "${repo_root}/third_party/FACT/evaluation/robotwin/eval_all_tasks.sh" \
            "${task_config}" "${test_num}" || attempt_rc=$?
      attempt_csv="${attempt_dir}/results.csv"
      attempt_log="${attempt_dir}/logs/${task_name}.log"
      row=""
      if [[ -f "${attempt_csv}" ]]; then
        row=$(awk -F, -v task_name="${task_name}" \
          '$1 == task_name && $4 != "ERROR" && $3 != "" {print; exit}' "${attempt_csv}")
      fi
      if [[ -n "${row}" && -f "${attempt_log}" ]] \
        && grep -Eq 'OIDN Error:|ErrorDeviceLost|DeviceLost|Render Error' "${attempt_log}"; then
        echo "[retry] ${task_name}: renderer error found in successful client log"
        row=""
      fi
      if [[ -n "${row}" ]]; then
        upsert_result "${task_name}" "${row}"
        if [[ -f "${attempt_dir}/logs/${task_name}.log" ]]; then
          cp "${attempt_dir}/logs/${task_name}.log" \
            "${sweep_dir}/logs/${task_name}.attempt_${attempt}.log"
        fi
        echo "[recovered] ${row}"
        return 0
      fi
      echo "[retry] ${task_name}: attempt ${attempt} rc=${attempt_rc}"
    done
    upsert_result "${task_name}" "${task_name},,,ERROR"
    echo "[failed] ${task_name}: exhausted ${task_max_attempts} attempts" >&2
    return 1
  }

  run_task_slot() {
    local slot=$1
    local slot_status=0
    local shard_index task_name
    for shard_index in "${!shard[@]}"; do
      if (( shard_index % jobs_per_gpu == slot )); then
        task_name=${shard[shard_index]}
        run_task "${task_name}" || slot_status=1
      fi
    done
    return "${slot_status}"
  }

  local client_status=0
  local -a task_slot_pids=()
  local slot slot_pid
  for ((slot = 0; slot < jobs_per_gpu; slot++)); do
    run_task_slot "${slot}" &
    task_slot_pids+=("$!")
  done
  for slot_pid in "${task_slot_pids[@]}"; do
    if ! wait "${slot_pid}"; then
      client_status=1
    fi
  done
  cleanup_worker
  trap - EXIT INT TERM
  return "${client_status}"
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
find "${run_dir}/stage2_images" -type f -name '*.png' -print \
  | sort > "${run_dir}/stage2_image_manifest.txt"

result_tasks=$(awk 'END {print NR-1}' "${results_csv}")
mp4_count=$(wc -l < "${run_dir}/mp4_manifest.txt")
value_trace_count=$(wc -l < "${run_dir}/value_trace_manifest.txt")
stage2_image_count=$(wc -l < "${run_dir}/stage2_image_manifest.txt")
expected_episodes=$((50 * test_num))
{
  echo "jobs_per_gpu=${jobs_per_gpu} dynamic_batch=$((jobs_per_gpu > 1)) batch_wait_ms=${batch_wait_ms}"
  echo "result_tasks=${result_tasks}/50"
  echo "mp4=${mp4_count}/${expected_episodes}"
  if [[ "${aux_outputs}" == "1" ]]; then
    echo "value_traces=${value_trace_count}/${expected_episodes}"
    echo "stage2_images=${stage2_image_count} (at least ${expected_episodes})"
  else
    echo "aux_outputs=disabled (Stage-1 success-rate fast path)"
  fi
} | tee -a "${run_dir}/summary.txt"

artifact_status=0
if [[ "${aux_outputs}" == "1" ]] \
  && { [[ ${value_trace_count} -ne ${expected_episodes} ]] \
    || [[ ${stage2_image_count} -lt ${expected_episodes} ]]; }; then
  artifact_status=1
fi
if [[ ${worker_status} -ne 0 ]] \
  || grep -q ',ERROR$' "${results_csv}" \
  || [[ ${result_tasks} -ne 50 ]] \
  || [[ ${mp4_count} -lt ${expected_episodes} ]] \
  || [[ ${artifact_status} -ne 0 ]]; then
  echo "One or more eval workers/tasks failed; inspect ${run_dir}/workers" >&2
  exit 1
fi
echo "RoboTwin full-task eval complete: ${run_dir}"
