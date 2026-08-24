#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 OPTIX_RUN OIDN_RUN EVAL_RESULT_ROOT OUTPUT_ROOT SNAPSHOT_NAME" >&2
  exit 2
fi

optix_run=$(realpath "$1")
oidn_run=$(realpath "$2")
eval_result_root=$(realpath "$3")
output_root=$(realpath "$4")
snapshot_name=$5
if [[ "${snapshot_name}" == */* || -z "${snapshot_name}" ]]; then
  echo "SNAPSHOT_NAME must be a non-empty basename" >&2
  exit 2
fi
for path in "${optix_run}" "${oidn_run}" "${eval_result_root}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Required directory does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${optix_run}/.started" ]]; then
  echo "OptiX run has no .started cutoff: ${optix_run}" >&2
  exit 2
fi
for command in rsync tar zstd sha256sum; do
  command -v "${command}" >/dev/null || {
    echo "Required command is unavailable: ${command}" >&2
    exit 2
  }
done

snapshot_root="${output_root}/${snapshot_name}"
archive="${output_root}/${snapshot_name}.tar.zst"
if [[ -e "${snapshot_root}" || -e "${archive}" || -e "${archive}.sha256" ]]; then
  echo "Snapshot output already exists for ${snapshot_name}" >&2
  exit 2
fi

mkdir -p \
  "${snapshot_root}/runs/optix_parallel" \
  "${snapshot_root}/runs/oidn233_serial" \
  "${snapshot_root}/mp4" \
  "${snapshot_root}/metadata"

# Copy the live OIDN run before collecting videos so its small mutable logs are
# frozen quickly. The stopped OptiX run is immutable at this point.
rsync -a "${oidn_run}/" "${snapshot_root}/runs/oidn233_serial/"
rsync -a "${optix_run}/" "${snapshot_root}/runs/optix_parallel/"

# Exclude an MP4 that ffmpeg may still be writing for the live OIDN episode.
mp4_cutoff="${snapshot_root}/metadata/mp4_cutoff"
touch -d '2 minutes ago' "${mp4_cutoff}"
mp4_manifest0="${snapshot_root}/metadata/mp4_manifest.null"
find "${eval_result_root}" -type f -name '*.mp4' \
  -newer "${optix_run}/.started" ! -newer "${mp4_cutoff}" \
  -printf '%P\0' | sort -z > "${mp4_manifest0}"
tr '\0' '\n' < "${mp4_manifest0}" \
  > "${snapshot_root}/metadata/mp4_manifest.txt"
rsync -aR --from0 --files-from="${mp4_manifest0}" \
  "${eval_result_root}/" "${snapshot_root}/mp4/"

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "optix_run=${optix_run}"
  echo "oidn_run=${oidn_run}"
  echo "eval_result_root=${eval_result_root}"
  echo "mp4_cutoff=$(date -u -r "${mp4_cutoff}" +%Y-%m-%dT%H:%M:%SZ)"
  echo "optix_stage2=$(find "${snapshot_root}/runs/optix_parallel/stage2_images" -type f -name '*.png' 2>/dev/null | wc -l)"
  echo "optix_values=$(find "${snapshot_root}/runs/optix_parallel/value_traces" -type f -name '*.npz' 2>/dev/null | wc -l)"
  echo "oidn_stage2=$(find "${snapshot_root}/runs/oidn233_serial/stage2_images" -type f -name '*.png' 2>/dev/null | wc -l)"
  echo "oidn_values=$(find "${snapshot_root}/runs/oidn233_serial/value_traces" -type f -name '*.npz' 2>/dev/null | wc -l)"
  echo "mp4=$(tr -cd '\0' < "${mp4_manifest0}" | wc -c)"
} > "${snapshot_root}/metadata/snapshot_summary.txt"

(
  cd "${snapshot_root}"
  find . -type f ! -path './metadata/FILES.sha256' -print0 \
    | sort -z | xargs -0 sha256sum > metadata/FILES.sha256
)
tar -I 'zstd -T4 -1' -cf "${archive}" -C "${output_root}" "${snapshot_name}"
sha256sum "${archive}" > "${archive}.sha256"

echo "snapshot_root=${snapshot_root}"
echo "archive=${archive}"
cat "${archive}.sha256"
