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
robotwin_python=${ROBONANA_ROBOTWIN_PYTHON:-${robotwin_env}/bin/python}
fact_conda_env=${FACT_CONDA_ENV:-$(dirname "$(dirname "${model_python}")")}
server_gpu_csv=${ROBONANA_EVAL_SERVER_GPUS:-0,1,2,3}
sim_gpu_csv=${ROBONANA_EVAL_SIM_GPUS:-4,5,6,7}
port_base=${ROBONANA_PORT_BASE:-18000}
run_dir=${ROBONANA_EVAL_RUN_DIR:-${repo_root}/outputs/robotwin_full_eval_$(date +%Y%m%d_%H%M%S)}
deploy_policy=${ROBONANA_DEPLOY_POLICY_PATH:-${repo_root}/src/robonana/configs/robotwin_eval_train_seen.yml}
task_timeout_seconds=${ROBONANA_TASK_TIMEOUT_SECONDS:-86400}
task_max_attempts=${ROBONANA_TASK_MAX_ATTEMPTS:-3}
episode_timeout_seconds=${ROBONANA_EPISODE_TIMEOUT_SECONDS:-3600}
episode_gpu_attempts=${ROBONANA_EPISODE_GPU_ATTEMPTS:-2}
episode_cpu_fallback=${ROBONANA_EPISODE_CPU_FALLBACK:-0}
jobs_per_gpu=${ROBONANA_EVAL_JOBS_PER_GPU:-1}
batch_wait_ms=${ROBONANA_EVAL_BATCH_WAIT_MS:-100}
static_camera_csv=${ROBONANA_ROBOTWIN_STATIC_CAMERAS:-head_camera}
video_log=${EVAL_VIDEO_LOG:-1}
seed_group=${ROBONANA_EVAL_SEED_GROUP:-0}
client_python_wrapper=${repo_root}/scripts/robotwin_eval_python.sh
isolated_task_runner=${repo_root}/scripts/eval_robotwin_task_isolated.py

terminate_process_tree() {
  local root_pid=$1
  local -a child_pids=()
  local child_pid
  if ! kill -0 "${root_pid}" 2>/dev/null; then
    return 0
  fi
  # Freeze the parent before enumerating descendants so it cannot create a
  # new nested timeout/process group between discovery and termination.
  kill -STOP "${root_pid}" 2>/dev/null || true
  mapfile -t child_pids < <(pgrep -P "${root_pid}" 2>/dev/null || true)
  for child_pid in "${child_pids[@]}"; do
    terminate_process_tree "${child_pid}"
  done
  kill -CONT "${root_pid}" 2>/dev/null || true
  kill -TERM "${root_pid}" 2>/dev/null || true
}

if [[ -n "${ROBONANA_EVAL_GPUS:-}" ]]; then
  echo "ROBONANA_EVAL_GPUS is no longer supported because it colocates policy and SAPIEN." >&2
  echo "Set disjoint ROBONANA_EVAL_SERVER_GPUS and ROBONANA_EVAL_SIM_GPUS instead." >&2
  exit 2
fi
IFS=',' read -r -a server_gpu_ids <<< "${server_gpu_csv}"
IFS=',' read -r -a sim_gpu_ids <<< "${sim_gpu_csv}"
if [[ ${#server_gpu_ids[@]} -eq 0 || ${#sim_gpu_ids[@]} -eq 0 ]]; then
  echo "ROBONANA_EVAL_SERVER_GPUS and ROBONANA_EVAL_SIM_GPUS must not be empty" >&2
  exit 2
fi
if [[ ${#server_gpu_ids[@]} -ne ${#sim_gpu_ids[@]} ]]; then
  echo "Policy-server and simulator GPU lists must have the same length" >&2
  exit 2
fi
declare -A server_gpu_set=()
declare -A sim_gpu_set=()
for gpu in "${server_gpu_ids[@]}"; do
  if [[ ! ${gpu} =~ ^[0-9]+$ || -n "${server_gpu_set[${gpu}]:-}" ]]; then
    echo "Invalid or duplicate policy-server GPU: ${gpu}" >&2
    exit 2
  fi
  server_gpu_set["${gpu}"]=1
done
for gpu in "${sim_gpu_ids[@]}"; do
  if [[ ! ${gpu} =~ ^[0-9]+$ || -n "${sim_gpu_set[${gpu}]:-}" ]]; then
    echo "Invalid or duplicate simulator GPU: ${gpu}" >&2
    exit 2
  fi
  if [[ -n "${server_gpu_set[${gpu}]:-}" ]]; then
    echo "GPU ${gpu} is assigned to both policy serving and SAPIEN; the pools must be disjoint" >&2
    exit 2
  fi
  sim_gpu_set["${gpu}"]=1
done
for required in "${checkpoint}" "${stats_path}" "${deploy_policy}" "${model_python}" \
  "${robotwin_python}" "${client_python_wrapper}" "${isolated_task_runner}"; do
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
  && ${episode_timeout_seconds} =~ ^[1-9][0-9]*$ \
  && ${episode_gpu_attempts} =~ ^[1-9][0-9]*$ \
  && ${jobs_per_gpu} =~ ^[1-9][0-9]*$ ]]; then
  echo "Task/episode timeouts, retry counts, and jobs per GPU must be positive integers" >&2
  exit 2
fi
if [[ ${episode_cpu_fallback} != 0 && ${episode_cpu_fallback} != 1 ]]; then
  echo "ROBONANA_EPISODE_CPU_FALLBACK must be 0 or 1" >&2
  exit 2
fi
if [[ ${video_log} != 0 && ${video_log} != 1 ]]; then
  echo "EVAL_VIDEO_LOG must be 0 or 1" >&2
  exit 2
fi
if ! [[ ${seed_group} =~ ^[0-9]+$ ]]; then
  echo "ROBONANA_EVAL_SEED_GROUP must be a non-negative integer" >&2
  exit 2
fi
if ! "${robotwin_python}" - <<'PY'
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


spec = importlib.util.find_spec("sapien")
if spec is None or spec.origin is None:
    raise SystemExit("SAPIEN is not installed in the RoboTwin environment")
sapien_root = Path(spec.origin).resolve().parent
library_root = (sapien_root / "../sapien.libs").resolve()
manifest_path = library_root / "robonana_oidn_gpu_serial.json"
if not manifest_path.is_file():
    raise SystemExit(
        f"serialized GPU OIDN manifest is missing: {manifest_path}; "
        "run scripts/install_sapien_oidn_blackwell.sh"
    )
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "oidn_version": "2.4.1",
    "sapien_version": "3.0.0.dev20260601+6a50b78b",
    "schema_version": 1,
    "svulkan_commit": "74d6529a6a213bfb84dee75035600b79eb7c3c44",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(
            f"serialized GPU OIDN manifest has {key}={manifest.get(key)!r}, expected {value!r}"
        )
svulkan = library_root / "libsvulkan2.so"
if not svulkan.is_file() or sha256(svulkan) != manifest.get("libsvulkan2_sha256"):
    raise SystemExit("libsvulkan2.so does not match the serialized GPU OIDN manifest")
oidn_root = Path(manifest.get("oidn_library_dir", ""))
required = [
    oidn_root / "libOpenImageDenoise.so.2.4.1",
    oidn_root / "libOpenImageDenoise_core.so.2.4.1",
    oidn_root / "libOpenImageDenoise_device_cuda.so.2.4.1",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("missing serialized GPU OIDN libraries: " + ", ".join(missing))
print(
    "SAPIEN preflight: serialized Vulkan/CUDA OIDN 2.4.1, "
    f"svulkan={manifest['svulkan_commit'][:8]}"
)
PY
then
  echo "OIDN preflight failed for ${robotwin_env}" >&2
  exit 2
fi

step_limits="${robotwin_path}/task_config/_eval_step_limit.yml"
mapfile -t all_tasks < <(grep -oE '^[a-z0-9_]+:' "${step_limits}" | tr -d ':')
if [[ ${#all_tasks[@]} -ne 50 ]]; then
  echo "Expected exactly 50 RoboTwin eval tasks, found ${#all_tasks[@]}" >&2
  exit 2
fi
tasks=("${all_tasks[@]}")
if [[ -n "${ROBONANA_EVAL_TASKS:-}" ]]; then
  declare -A known_tasks=()
  declare -A selected_tasks=()
  for task_name in "${all_tasks[@]}"; do
    known_tasks["${task_name}"]=1
  done
  IFS=',' read -r -a requested_tasks <<< "${ROBONANA_EVAL_TASKS}"
  tasks=()
  for task_name in "${requested_tasks[@]}"; do
    task_name=${task_name//[[:space:]]/}
    if [[ -z "${task_name}" ]]; then
      continue
    fi
    if [[ -z "${known_tasks[${task_name}]:-}" ]]; then
      echo "Unknown RoboTwin task in ROBONANA_EVAL_TASKS: ${task_name}" >&2
      exit 2
    fi
    if [[ -n "${selected_tasks[${task_name}]:-}" ]]; then
      echo "Duplicate RoboTwin task in ROBONANA_EVAL_TASKS: ${task_name}" >&2
      exit 2
    fi
    selected_tasks["${task_name}"]=1
    tasks+=("${task_name}")
  done
  if [[ ${#tasks[@]} -eq 0 ]]; then
    echo "ROBONANA_EVAL_TASKS did not contain any task names" >&2
    exit 2
  fi
fi
expected_task_count=${#tasks[@]}

mkdir -p "${run_dir}/workers"
[[ -e "${run_dir}/.started" ]] || touch "${run_dir}/.started"
"${model_python}" "${repo_root}/scripts/audit_robotwin_instructions.py" \
  --dataset-root "${dataset_root}" \
  --robotwin-root "${robotwin_path}" \
  --output "${run_dir}/instruction_audit.json" \
  --require-seen \
  > "${run_dir}/instruction_audit.log"

run_worker() {
  local rank=$1
  local server_gpu=$2
  local sim_gpu=$3
  local port=$((port_base + rank))
  local worker_dir="${run_dir}/workers/rank_${rank}_server_${server_gpu}_sim_${sim_gpu}"
  local runtime_dir="/tmp/robonana_eval_${USER}_${port}"
  local shard=()
  local index
  for index in "${!tasks[@]}"; do
    if (( index % ${#sim_gpu_ids[@]} == rank )); then
      shard+=("${tasks[index]}")
    fi
  done
  local sweep_dir="${worker_dir}/sweep"
  local results_csv="${sweep_dir}/results.csv"
  mkdir -p "${worker_dir}" "${runtime_dir}" "${sweep_dir}/logs" "${worker_dir}/attempts"
  [[ -f "${results_csv}" ]] || echo "task,success,total,success_rate" > "${results_csv}"

  local server_args=(
    "${model_python}" "${repo_root}/scripts/inference_server_robotwin_batched.py"
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
    --inference-mode "${ROBONANA_INFERENCE_MODE:-action}"
    --rejection-candidate-count "${ROBONANA_REJECTION_CANDIDATE_COUNT:-32}"
    --q-return-scale "${ROBONANA_Q_RETURN_SCALE:-1000}"
    --port "${port}"
    --max-batch-size "${jobs_per_gpu}"
    --max-batch-wait-ms "${batch_wait_ms}"
    --max-clients "$((jobs_per_gpu * 2))"
  )
  if [[ -n "${ROBONANA_MODEL_CONFIG:-}" ]]; then
    server_args+=(--model-config "${ROBONANA_MODEL_CONFIG}")
  fi

  local server_pid=""
  local -a task_slot_pids=()
  cleanup_worker() {
    local task_slot_pid
    for task_slot_pid in "${task_slot_pids[@]}"; do
      terminate_process_tree "${task_slot_pid}"
    done
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      terminate_process_tree "${server_pid}"
      wait "${server_pid}" 2>/dev/null || true
    fi
    rmdir "${runtime_dir}" 2>/dev/null || true
  }
  trap cleanup_worker EXIT
  trap 'exit 130' INT TERM

  env \
    CUDA_VISIBLE_DEVICES="${server_gpu}" \
    PYTHONPATH="${repo_root}/src:${repo_root}/third_party/FACT:${repo_root}/third_party/flux2_official/src:${repo_root}/third_party/flux2/src" \
    "${server_args[@]}" \
    > "${worker_dir}/server.log" 2>&1 &
  server_pid=$!

  local client_env=(
    "CUDA_VISIBLE_DEVICES=${sim_gpu}"
    "OIDN_DEFAULT_DEVICE=cuda"
    "XDG_RUNTIME_DIR=${runtime_dir}"
    "PYTHONPATH=${repo_root}/src"
    "ROBOTWIN_PATH=${robotwin_path}"
    "ROBOTWIN_CONDA_ENV=${robotwin_env}"
    "ROBONANA_ROBOTWIN_PYTHON=${robotwin_python}"
    "CLIENT_PYTHON=${client_python_wrapper}"
    "FACT_CONDA_ENV=${fact_conda_env}"
    "DEPLOY_POLICY_PATH=${deploy_policy}"
    "POLICY_NAME=robonana_robotwin.adapter"
    "PORT=${port}"
    "TEST_NUM=${test_num}"
    "EXECUTE_ACTIONS_PER_PLAN=48"
    "SERVER_TIMEOUT_MS=600000"
    "SERVER_WAIT_SECONDS=600"
    "EVAL_VIDEO_LOG=${video_log}"
    "PYTHONUNBUFFERED=1"
    "LOW_FREQUENCY_RGB=0"
    "SKIP_ACTION_RENDER_SYNC=0"
    "ROBONANA_ROBOTWIN_STATIC_CAMERAS=${static_camera_csv}"
    # cuda:0 is this rank's sole logical device after CUDA_VISIBLE_DEVICES isolation.
    "ROBONANA_SAPIEN_RENDER_DEVICE=cuda:0"
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
    local task_state_dir="${worker_dir}/task_runs/${task_name}"
    local fallback_flag=--no-cpu-fallback
    if [[ ${episode_cpu_fallback} == 1 ]]; then
      fallback_flag=--cpu-fallback
    fi
    local existing
    existing=$(awk -F, -v task_name="${task_name}" \
      '$1 == task_name && $4 != "ERROR" && $3 != "" {print; exit}' "${results_csv}")
    if [[ -n "${existing}" ]]; then
      echo "[resume-skip] ${task_name}: ${existing}"
      return 0
    fi

    mkdir -p "${task_state_dir}"
    local attempt attempt_dir attempt_csv attempt_log row attempt_rc
    for ((attempt = 1; attempt <= task_max_attempts; attempt++)); do
      attempt_dir="${worker_dir}/attempts/${task_name}/attempt_${attempt}_$(date +%Y%m%d_%H%M%S)"
      mkdir -p "${attempt_dir}"
      attempt_log="${attempt_dir}/task.log"
      echo "[attempt ${attempt}/${task_max_attempts}] ${task_name} " \
        "task_timeout=${task_timeout_seconds}s episode_timeout=${episode_timeout_seconds}s"
      attempt_rc=0
      timeout --signal=TERM --kill-after=60 "${task_timeout_seconds}" \
        env "${client_env[@]}" \
          "${model_python}" "${isolated_task_runner}" \
            --task-name "${task_name}" \
            --task-config "${task_config}" \
            --test-num "${test_num}" \
            --output-dir "${task_state_dir}" \
            --launch-client "${repo_root}/third_party/FACT/evaluation/robotwin/launch_client.sh" \
            --episode-timeout-seconds "${episode_timeout_seconds}" \
            --gpu-attempts "${episode_gpu_attempts}" \
            --seed-group "${seed_group}" \
            "${fallback_flag}" \
            > "${attempt_log}" 2>&1 || attempt_rc=$?
      attempt_csv="${task_state_dir}/results.csv"
      row=""
      if [[ -f "${attempt_csv}" ]]; then
        row=$(awk -F, -v task_name="${task_name}" \
          '$1 == task_name && $4 != "ERROR" && $3 != "" {print; exit}' "${attempt_csv}")
      fi
      if [[ -n "${row}" ]]; then
        upsert_result "${task_name}" "${row}"
        if [[ -f "${attempt_log}" ]]; then
          cp "${attempt_log}" "${sweep_dir}/logs/${task_name}.attempt_${attempt}.log"
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
    terminate_process_tree "${pid}"
  done
  for pid in "${worker_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup_all_workers EXIT INT TERM
for rank in "${!sim_gpu_ids[@]}"; do
  run_worker "${rank}" "${server_gpu_ids[rank]}" "${sim_gpu_ids[rank]}" \
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
for rank in "${!sim_gpu_ids[@]}"; do
  worker_csv="${run_dir}/workers/rank_${rank}_server_${server_gpu_ids[rank]}_sim_${sim_gpu_ids[rank]}/sweep/results.csv"
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

find "${run_dir}/workers" -type f -path '*/task_runs/*/mp4_manifest.txt' -exec cat {} + \
  | sort -u > "${run_dir}/mp4_manifest.txt"

result_tasks=$(awk 'END {print NR-1}' "${results_csv}")
mp4_count=$(wc -l < "${run_dir}/mp4_manifest.txt")
expected_episodes=$((expected_task_count * test_num))
expected_mp4=$((video_log * expected_episodes))
{
  echo "mode=action_only renderer=sapien_oidn episode_isolation=1"
  echo "episode_timeout_seconds=${episode_timeout_seconds} gpu_attempts=${episode_gpu_attempts} cpu_fallback=${episode_cpu_fallback}"
  echo "jobs_per_gpu=${jobs_per_gpu} dynamic_batch=$((jobs_per_gpu > 1)) batch_wait_ms=${batch_wait_ms}"
  echo "result_tasks=${result_tasks}/${expected_task_count}"
  echo "mp4=${mp4_count}/${expected_mp4}"
} | tee -a "${run_dir}/summary.txt"

if [[ ${worker_status} -ne 0 ]] \
  || grep -q ',ERROR$' "${results_csv}" \
  || [[ ${result_tasks} -ne ${expected_task_count} ]] \
  || [[ ${mp4_count} -lt ${expected_mp4} ]]; then
  echo "One or more eval workers/tasks failed; inspect ${run_dir}/workers" >&2
  exit 1
fi
echo "RoboTwin full-task eval complete: ${run_dir}"
