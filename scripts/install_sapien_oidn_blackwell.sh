#!/usr/bin/env bash
set -euo pipefail

oidn_version=2.3.3
oidn_sha256=3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f
oidn_archive_name="oidn-${oidn_version}.x86_64.linux.tar.gz"
oidn_url="https://github.com/RenderKit/oidn/releases/download/v${oidn_version}/${oidn_archive_name}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ENV_PREFIX [OIDN_ARCHIVE]" >&2
  exit 2
fi

env_prefix=$(realpath "$1")
python_bin="${env_prefix}/bin/python"
archive=${2:-}
if [[ ! -x "${python_bin}" ]]; then
  echo "Target environment has no executable Python: ${python_bin}" >&2
  exit 2
fi

sapien_dir=$("${python_bin}" - <<'PY'
from pathlib import Path
import sapien

print(Path(sapien.__file__).resolve().parent)
PY
)
case "${sapien_dir}" in
  "${env_prefix}"/*) ;;
  *)
    echo "SAPIEN resolved outside target environment: ${sapien_dir}" >&2
    exit 2
    ;;
esac

loader="${sapien_dir}/_oidn_tricks.py"
library_dir="${sapien_dir}/oidn_library"
if [[ ! -f "${loader}" || ! -d "${library_dir}" ]]; then
  echo "SAPIEN OIDN loader/library directory is missing under ${sapien_dir}" >&2
  exit 2
fi

work_dir=$(mktemp -d)
cleanup() {
  rm -rf "${work_dir}"
}
trap cleanup EXIT

if [[ -n "${archive}" ]]; then
  archive=$(realpath "${archive}")
  if [[ ! -f "${archive}" ]]; then
    echo "OIDN archive does not exist: ${archive}" >&2
    exit 2
  fi
else
  archive="${work_dir}/${oidn_archive_name}"
  curl -fL --retry 3 --output "${archive}" "${oidn_url}"
fi

echo "${oidn_sha256}  ${archive}" | sha256sum --check --status || {
  echo "OIDN archive SHA256 mismatch: ${archive}" >&2
  exit 2
}
tar -xzf "${archive}" -C "${work_dir}"
source_dir="${work_dir}/oidn-${oidn_version}.x86_64.linux/lib"

libraries=(
  "libOpenImageDenoise.so.${oidn_version}"
  "libOpenImageDenoise_core.so.${oidn_version}"
  "libOpenImageDenoise_device_cuda.so.${oidn_version}"
)
for library in "${libraries[@]}"; do
  if [[ ! -f "${source_dir}/${library}" ]]; then
    echo "Expected library is missing from OIDN archive: ${library}" >&2
    exit 2
  fi
  install -m 0755 "${source_dir}/${library}" "${library_dir}/${library}"
done

old_count=$(grep -o '2\.0\.1' "${loader}" | wc -l || true)
new_count=$(grep -o "${oidn_version}" "${loader}" | wc -l || true)
if [[ ${old_count} -eq 2 && ${new_count} -eq 0 ]]; then
  sed -i "s/2\.0\.1/${oidn_version}/g" "${loader}"
elif [[ ${old_count} -eq 0 && ${new_count} -eq 2 ]]; then
  echo "SAPIEN OIDN loader already targets ${oidn_version}"
else
  echo "Unexpected OIDN versions in ${loader}: old=${old_count} new=${new_count}" >&2
  exit 2
fi

"${python_bin}" - <<PY
import ctypes
from pathlib import Path
import sapien

root = Path(sapien.__file__).resolve().parent / "oidn_library"
ctypes.CDLL(str(root / "libOpenImageDenoise_core.so.${oidn_version}"), ctypes.RTLD_LOCAL)
ctypes.CDLL(str(root / "libOpenImageDenoise.so.${oidn_version}"), ctypes.RTLD_LOCAL)
print(f"SAPIEN={sapien.__version__} OIDN=${oidn_version} library_dir={root}")
PY
