#!/usr/bin/env bash
# 中文：环境维护：安装 Blackwell/B200 所需的 SAPIEN/OIDN 依赖修复。
# English: Environment maintenance: install SAPIEN/OIDN fixes for Blackwell/B200.
# 调用 / Invocation: 仅显式维护时运行；会修改 Python 环境和依赖。 / Explicit maintenance only; modifies the Python environment and dependencies.
# 导航 / Guide: scripts/README.md (env)
set -euo pipefail

sapien_version=3.0.0.dev20260601+6a50b78b
sapien_wheel_url='https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20260601%2B6a50b78b-cp310-cp310-manylinux_2_28_x86_64.whl'
svulkan_repo=https://github.com/haosulab/sapien-vulkan-2.git
svulkan_commit=74d6529a6a213bfb84dee75035600b79eb7c3c44
oidn_version=2.4.1

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ENV_PREFIX [PERSISTENT_BUILD_ROOT]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
env_prefix=$(realpath "$1")
build_root=${2:-${env_prefix}/.robonana-sapien-oidn}
mkdir -p "${build_root}"
build_root=$(realpath "${build_root}")
python_bin=${env_prefix}/bin/python
cmake_bin=${CMAKE_BIN:-cmake}
cuda_path=${CUDA_PATH:-/usr/local/cuda}
cuda_arch=${CMAKE_CUDA_ARCHITECTURES:-100}
source_dir=${SVULKAN_SOURCE_DIR:-${build_root}/sapien-vulkan-2-${svulkan_commit:0:8}}
build_dir=${SVULKAN_BUILD_DIR:-${build_root}/build-${svulkan_commit:0:8}}
patch_file=${repo_root}/patches/sapien/0001-serialize-oidn-vulkan-cuda.patch
ray_patch_file=${repo_root}/patches/sapien/0002-terminate-zero-throughput-rays.patch

apply_ray_guard() {
  # 中文：修复纯黑目标的零散射光线；不改任务颜色/几何，也不切换渲染器。
  # English: Guard zero-scattering rays without changing task colors/geometry or renderer.
  # Upstream: https://github.com/haosulab/sapien-vulkan-2/blob/74d6529a6a213bfb84dee75035600b79eb7c3c44/shader/rt/camera.rchit
  # Upstream: https://github.com/haosulab/sapien-vulkan-2/blob/74d6529a6a213bfb84dee75035600b79eb7c3c44/shader/rt/camera.rgen
  local package_dir
  package_dir=$("${python_bin}" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("sapien").origin).resolve().parent)')
  case "${package_dir}" in
    "${env_prefix}"/*) ;;
    *) echo "SAPIEN resolved outside target environment: ${package_dir}" >&2; exit 2 ;;
  esac
  if git -C "${package_dir}" apply --reverse --check "${ray_patch_file}" 2>/dev/null; then
    echo "SAPIEN zero-throughput ray guard is already applied"
    return
  fi
  git -C "${package_dir}" apply --check "${ray_patch_file}"
  local shader
  for shader in camera.rchit camera.rgen; do
    if [[ ! -f "${package_dir}/vulkan_shader/rt/${shader}.robonana-original" ]]; then
      cp -p "${package_dir}/vulkan_shader/rt/${shader}" "${package_dir}/vulkan_shader/rt/${shader}.robonana-original"
    fi
  done
  git -C "${package_dir}" apply "${ray_patch_file}"
  echo "Installed SAPIEN zero-throughput ray guard: ${package_dir}/vulkan_shader/rt"
}

# 中文：已有正确OIDN运行时仅更新shader，无需重编译库；新进程才会加载。
# English: An existing OIDN runtime can update shaders without rebuilding; affects new processes.
# Invocation: ROBONANA_SAPIEN_SHADER_ONLY=1 bash scripts/env/install_sapien_oidn_blackwell.sh ENV_PREFIX
if [[ "${ROBONANA_SAPIEN_SHADER_ONLY:-0}" == "1" ]]; then
  apply_ray_guard
  exit 0
fi

for executable in "${python_bin}" "${cmake_bin}" ninja git "${cuda_path}/bin/nvcc"; do
  if ! command -v "${executable}" >/dev/null 2>&1 && [[ ! -x "${executable}" ]]; then
    echo "Required executable is missing: ${executable}" >&2
    exit 2
  fi
done
if [[ ! -f "${patch_file}" ]]; then
  echo "Missing serialized OIDN patch: ${patch_file}" >&2
  exit 2
fi
if [[ ! -f /usr/include/GL/gl.h ]]; then
  echo "Missing GL/gl.h; install libgl-dev before building svulkan2" >&2
  exit 2
fi

installed_version=$("${python_bin}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("sapien"))
except PackageNotFoundError:
    print("")
PY
)
if [[ "${installed_version}" != "${sapien_version}" ]]; then
  "${python_bin}" -m pip install --no-deps --force-reinstall "${sapien_wheel_url}"
fi
apply_ray_guard

if [[ ! -d "${source_dir}/.git" ]]; then
  git clone --no-checkout "${svulkan_repo}" "${source_dir}"
fi
git -C "${source_dir}" fetch --no-tags origin "${svulkan_commit}"
git -C "${source_dir}" checkout --detach "${svulkan_commit}"
if git -C "${source_dir}" apply --reverse --check "${patch_file}"; then
  echo "Serialized OIDN patch is already applied"
elif git -C "${source_dir}" apply --check "${patch_file}"; then
  git -C "${source_dir}" apply "${patch_file}"
else
  echo "Serialized OIDN patch does not apply cleanly to ${svulkan_commit}" >&2
  exit 2
fi

env CUDA_PATH="${cuda_path}" CUDACXX="${cuda_path}/bin/nvcc" \
  "${cmake_bin}" \
    -S "${source_dir}" \
    -B "${build_dir}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSVULKAN2_CUDA_INTEROP=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_SKIP_INSTALL_RULES=ON
env CUDA_PATH="${cuda_path}" \
  "${cmake_bin}" --build "${build_dir}" --target svulkan2 -j "${BUILD_JOBS:-32}"
env CUDA_PATH="${cuda_path}" CUDACXX="${cuda_path}/bin/nvcc" \
  "${cmake_bin}" --build "${build_dir}" --target OpenImageDenoise_device_cuda \
    -j "${BUILD_JOBS:-32}"

svulkan_library=${build_dir}/libsvulkan2.so
oidn_library_dir=${build_dir}/_deps/oidn-build
for required in "${svulkan_library}" \
  "${oidn_library_dir}/libOpenImageDenoise.so.${oidn_version}" \
  "${oidn_library_dir}/libOpenImageDenoise_core.so.${oidn_version}" \
  "${oidn_library_dir}/libOpenImageDenoise_device_cuda.so.${oidn_version}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Expected build artifact is missing: ${required}" >&2
    exit 2
  fi
done

sapien_dir=$("${python_bin}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("sapien")
if spec is None or spec.origin is None:
    raise SystemExit("SAPIEN package is not discoverable after installation")
print(Path(spec.origin).resolve().parent)
PY
)
case "${sapien_dir}" in
  "${env_prefix}"/*) ;;
  *)
    echo "SAPIEN resolved outside target environment: ${sapien_dir}" >&2
    exit 2
    ;;
esac

sapien_lib_dir=${sapien_dir}/../sapien.libs
target_library=${sapien_lib_dir}/libsvulkan2.so
upstream_backup=${sapien_lib_dir}/libsvulkan2.so.upstream-nightly
manifest=${sapien_lib_dir}/robonana_oidn_gpu_serial.json
if [[ ! -f "${upstream_backup}" ]]; then
  cp -a "${target_library}" "${upstream_backup}"
fi
install -m 0755 "${svulkan_library}" "${target_library}"

SAPIEN_VERSION="${sapien_version}" \
SVULKAN_COMMIT="${svulkan_commit}" \
OIDN_VERSION="${oidn_version}" \
PATCH_FILE="${patch_file}" \
SVULKAN_LIBRARY="${target_library}" \
OIDN_LIBRARY_DIR="${oidn_library_dir}" \
BUILD_DIR="${build_dir}" \
MANIFEST="${manifest}" \
"${python_bin}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


patch = Path(os.environ["PATCH_FILE"]).resolve()
library = Path(os.environ["SVULKAN_LIBRARY"]).resolve()
manifest = Path(os.environ["MANIFEST"]).resolve()
payload = {
    "build_dir": str(Path(os.environ["BUILD_DIR"]).resolve()),
    "libsvulkan2_sha256": sha256(library),
    "oidn_library_dir": str(Path(os.environ["OIDN_LIBRARY_DIR"]).resolve()),
    "oidn_version": os.environ["OIDN_VERSION"],
    "patch_sha256": sha256(patch),
    "sapien_version": os.environ["SAPIEN_VERSION"],
    "schema_version": 1,
    "svulkan_commit": os.environ["SVULKAN_COMMIT"],
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Installed serialized GPU OIDN runtime: {manifest}")
PY

env -u LD_LIBRARY_PATH "${python_bin}" - <<PY
import sapien

assert sapien.__version__ == "${sapien_version}"
print("SAPIEN", sapien.__version__, "OIDN", "${oidn_version}", "svulkan", "${svulkan_commit}")
PY
