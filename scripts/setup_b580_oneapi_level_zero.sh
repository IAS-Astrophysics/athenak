#!/usr/bin/env bash
set -euo pipefail

# Install the system pieces needed for oneAPI SYCL to see an Intel Arc B580
# through the Level Zero backend.  Run this script with sudo from the repo root:
#   sudo bash scripts/setup_b580_oneapi_level_zero.sh

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: run with sudo/root" >&2
  exit 1
fi

USER_TO_FIX="${SUDO_USER:-${USER}}"
if [[ -z "${USER_TO_FIX}" || "${USER_TO_FIX}" == "root" ]]; then
  USER_TO_FIX="${ATHENAK_USER:-}"
fi

WORKDIR="${WORKDIR:-/tmp/athenak-b580-oneapi}"
ONEAPI_ROOT="${ONEAPI_ROOT:-/home/${USER_TO_FIX}/intel/oneapi}"
LEVEL_ZERO_VER="${LEVEL_ZERO_VER:-1.28.6}"
COMPUTE_RUNTIME_VER="${COMPUTE_RUNTIME_VER:-26.18.38308.1}"
IGC_VER="${IGC_VER:-2.34.4}"
IGC_BUILD="${IGC_BUILD:-21428}"
GMM_VER="${GMM_VER:-22.10.0}"

B580_PCI_ID="8086:e20b"

detect_b580() {
  if command -v lspci >/dev/null 2>&1; then
    lspci -Dnnd 8086: | grep -qi "${B580_PCI_ID}"
    return $?
  fi

  for dev in /sys/bus/pci/devices/*; do
    [[ -r "${dev}/vendor" && -r "${dev}/device" ]] || continue
    [[ "$(tr '[:upper:]' '[:lower:]' < "${dev}/vendor")" == "0x8086" ]] || continue
    [[ "$(tr '[:upper:]' '[:lower:]' < "${dev}/device")" == "0xe20b" ]] || continue
    return 0
  done
  return 1
}

if ! detect_b580; then
  echo "error: Intel Arc B580 (${B580_PCI_ID}) was not detected." >&2
  echo "Detected Intel PCI devices:" >&2
  lspci -Dnnd 8086: >&2 || true
  exit 1
fi

if ! lsmod | grep -q '^xe '; then
  echo "warning: kernel module 'xe' is not loaded. The B580 should normally use xe." >&2
fi

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

download() {
  local url="$1"
  local file="${url##*/}"
  file="${file//%2B/+}"
  if [[ ! -f "${file}" ]]; then
    wget -O "${file}" "${url}"
  fi
}

download "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VER}/libze1_${LEVEL_ZERO_VER}%2Bu24.04_amd64.deb"
download "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VER}/intel-igc-core-2_${IGC_VER}+${IGC_BUILD}_amd64.deb"
download "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VER}/intel-igc-opencl-2_${IGC_VER}+${IGC_BUILD}_amd64.deb"
download "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VER}/intel-ocloc_${COMPUTE_RUNTIME_VER}-0_amd64.deb"
download "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VER}/libigdgmm12_${GMM_VER}_amd64.deb"
download "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VER}/libze-intel-gpu1_${COMPUTE_RUNTIME_VER}-0_amd64.deb"

apt-get update
apt-get install -y wget ca-certificates pciutils intel-gpu-tools
dpkg -i ./*.deb || apt-get -f install -y

if [[ -n "${USER_TO_FIX}" && "${USER_TO_FIX}" != "root" ]]; then
  usermod -aG render,video "${USER_TO_FIX}" || true
fi

if [[ -d "${ONEAPI_ROOT}/compiler" ]]; then
  cat >/etc/ld.so.conf.d/oneapi-local.conf <<EOF
${ONEAPI_ROOT}/compiler/latest/lib
${ONEAPI_ROOT}/compiler/latest/opt/compiler/lib
EOF
  cat >/etc/profile.d/oneapi-local.sh <<EOF
if [ -r "${ONEAPI_ROOT}/setvars.sh" ]; then
  . "${ONEAPI_ROOT}/setvars.sh" >/dev/null 2>&1
fi
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
EOF
else
  echo "warning: ONEAPI_ROOT=${ONEAPI_ROOT} not found; not adding oneAPI profile hooks." >&2
fi

ldconfig

echo "B580 PCI device:"
lspci -Dnnd 8086: | grep -i "${B580_PCI_ID}" || true

echo "SYCL Level Zero devices:"
if [[ -r "${ONEAPI_ROOT}/setvars.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ONEAPI_ROOT}/setvars.sh" >/dev/null 2>&1
fi
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls --verbose

echo "Done. Log out and back in if group membership was updated."
