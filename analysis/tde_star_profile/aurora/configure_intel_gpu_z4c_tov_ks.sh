#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build/aurora-intel-gpu-z4c_tov_ks}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${PROJECT_ROOT}/install/z4c_tov_ks_intel_gpu}"
BUILD_PARALLELISM="${BUILD_PARALLELISM:-8}"

load_module_if_available() {
  local module_name="$1"
  if type module >/dev/null 2>&1; then
    module load "${module_name}" >/dev/null 2>&1 || true
  fi
}

if type module >/dev/null 2>&1; then
  module reset >/dev/null 2>&1 || true
  load_module_if_available oneapi
  load_module_if_available oneapi/release
  load_module_if_available cmake
fi

export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-COMPOSITE}"

if command -v mpic++ >/dev/null 2>&1; then
  CXX_COMPILER="${CXX_COMPILER:-mpic++}"
elif command -v icpx >/dev/null 2>&1; then
  CXX_COMPILER="${CXX_COMPILER:-icpx}"
else
  echo "ERROR: neither mpic++ nor icpx was found in PATH." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${INSTALL_PREFIX}" "${PROJECT_ROOT}/build_logs"

echo "Repository: ${REPO_DIR}"
echo "Build dir:  ${BUILD_DIR}"
echo "Install:    ${INSTALL_PREFIX}"
echo "Compiler:   ${CXX_COMPILER}"
echo "Problem:    z4c_tov_ks"
echo "GPU env:    SYCL_DEVICE_FILTER=${SYCL_DEVICE_FILTER}, ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR}"

cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DPROBLEM=z4c_tov_ks \
  -DAthena_ENABLE_MPI=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_SYCL=ON \
  -DKokkos_ARCH_INTEL_PVC=ON \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:--fp-model=precise}"

cmake --build "${BUILD_DIR}" --target athena --parallel "${BUILD_PARALLELISM}"

echo "Built executable: ${BUILD_DIR}/src/athena"
