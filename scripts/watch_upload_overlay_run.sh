#!/usr/bin/env bash
# Wait for a local-only overlay pipeline to finish, verify it, then upload it.
set -euo pipefail

PROJECT_ROOT="${ROBONANA_PROJECT_ROOT:-/data3/hongjia/robonana}"
RUN_NAME="${ROBONANA_RUN_NAME:-hanging_mug_return_overlay150_step1000_20260902}"
OUTPUT_ROOT="${ROBONANA_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/${RUN_NAME}}"
HF="${ROBONANA_HF:-/data3/hongjia/conda/envs/robonana/bin/hf}"
HF_REPO="${ROBONANA_HF_REPO:-AvaX1/robonana-eval-videos}"
EXPECTED_COUNT="${ROBONANA_EXPECTED_OVERLAYS:-150}"
PIPELINE_PID_FILE="${OUTPUT_ROOT}/pipeline.pid"
PIPELINE_LOG="${OUTPUT_ROOT}/pipeline_status.txt"
UPLOAD_LOG="${OUTPUT_ROOT}/upload_watcher.log"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${UPLOAD_LOG}") 2>&1

if [[ ! -f "${PIPELINE_PID_FILE}" ]]; then
  echo "status=upload_failed reason=missing_pipeline_pid_file"
  exit 1
fi

pipeline_pid="$(<"${PIPELINE_PID_FILE}")"
echo "status=waiting_for_local_annotation pipeline_pid=${pipeline_pid} started_at=$(date --iso-8601=seconds)"
while kill -0 "${pipeline_pid}" 2>/dev/null; do
  sleep 60
done

last_status="$(grep '^status=' "${PIPELINE_LOG}" | tail -n 1 || true)"
if [[ "${last_status}" != status=complete_local_only* ]]; then
  echo "status=upload_failed reason=annotation_not_complete last_status=${last_status}"
  exit 1
fi

video_count="$(find "${OUTPUT_ROOT}/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
telemetry_count="$(find "${OUTPUT_ROOT}/telemetry" -maxdepth 1 -type f -name '*.json' | wc -l)"
manifest_count="$(wc -l < "${OUTPUT_ROOT}/manifest.jsonl")"
if [[ "${video_count}" -ne "${EXPECTED_COUNT}" || "${telemetry_count}" -ne "${EXPECTED_COUNT}" || "${manifest_count}" -ne "${EXPECTED_COUNT}" ]]; then
  echo "status=upload_failed reason=count_mismatch videos=${video_count} telemetry=${telemetry_count} manifest=${manifest_count}"
  exit 1
fi

(
  cd "${OUTPUT_ROOT}"
  find . -type f \
    ! -name SHA256SUMS \
    ! -name pipeline_status.txt \
    ! -name hf_upload_status.json \
    ! -name upload_watcher.log \
    -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

archive="${OUTPUT_ROOT}.tar.zst"
archive_sha="${archive}.sha256"
tar --zstd -cf "${archive}" -C "$(dirname "${OUTPUT_ROOT}")" "$(basename "${OUTPUT_ROOT}")"
sha256sum "${archive}" > "${archive_sha}"

echo "status=uploading_hf repo=${HF_REPO} started_at=$(date --iso-8601=seconds)"
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
echo "status=upload_complete videos=${video_count} telemetry=${telemetry_count} hf=${HF_REPO}/${RUN_NAME} completed_at=$(date --iso-8601=seconds)"
