#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <case-name> <input-file> <out-dir> [athena overrides...]" >&2
  exit 2
fi

case_name=$1
input_file=$2
out_dir=$3
shift 3

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
athena="${ATHENA_BIN:-${repo_root}/build-sycl-grtorus/src/athena}"
oneapi_root="${ONEAPI_ROOT:-/home/hzhu/intel/oneapi}"
neo_root="${NEO_ROOT:-${repo_root}/.codex/neo-26.18-root}"

mkdir -p "${out_dir}"

export LD_LIBRARY_PATH="${neo_root}/usr/local/lib:${neo_root}/usr/lib/x86_64-linux-gnu:${oneapi_root}/compiler/2025.3/lib:${oneapi_root}/compiler/2025.3/opt/compiler/lib:${oneapi_root}/umf/1.0/lib:${oneapi_root}/2025.3/lib:${LD_LIBRARY_PATH:-}"
export PATH="${neo_root}/usr/bin:${oneapi_root}/compiler/2025.3/bin:${PATH}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER="${ZE_ENABLE_PCI_ID_DEVICE_ORDER:-1}"
export KOKKOS_PROFILE_LIBRARY="${KOKKOS_PROFILE_LIBRARY:-${repo_root}/benchmarks/gr_torus_b580/libkokkos_mem_tool.so}"

log="${out_dir}/${case_name}.log"
rss_log="${out_dir}/${case_name}.rss_kb"
summary="${out_dir}/${case_name}.summary"

{
  echo "case=${case_name}"
  echo "input=${input_file}"
  echo "athena=${athena}"
  echo "device_selector=${ONEAPI_DEVICE_SELECTOR}"
  echo "started=$(date -Is)"
  sycl-ls --verbose | sed -n '/\\[level_zero:gpu\\]/,/^$/p'
} >"${summary}"

"${athena}" -i "${input_file}" "$@" >"${log}" 2>&1 &
pid=$!
peak_rss=0
while kill -0 "${pid}" 2>/dev/null; do
  rss=$(awk '/VmRSS:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || echo 0)
  hwm=$(awk '/VmHWM:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || echo 0)
  [[ -n "${rss}" ]] || rss=0
  [[ -n "${hwm}" ]] || hwm=0
  (( hwm > peak_rss )) && peak_rss=${hwm}
  printf '%(%s)T %s %s\n' -1 "${rss}" "${hwm}" >>"${rss_log}"
  sleep 0.2
done
set +e
wait "${pid}"
status=$?
set -e

{
  echo "finished=$(date -Is)"
  echo "exit_status=${status}"
  echo "host_vm_hwm_kb=${peak_rss}"
  grep -E 'zone-cycles/cpu_second|cpu time used|MeshBlock-cycles|Terminating on cycle limit|KOKKOS_MEMORY_PEAK|SYCLDevice|HostSpace|SharedSpace|AnonymousSpace' "${log}" || true
} >>"${summary}"

exit "${status}"
