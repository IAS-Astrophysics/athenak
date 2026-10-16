#!/bin/bash
set -euo pipefail
rank=${PALS_RANKID:-${PMI_RANK:-0}}
local_rank=${PALS_LOCAL_RANKID:-$rank}
if [ "$CZ_NRANKS" = 12 ]; then
 export ZE_AFFINITY_MASK="$((local_rank / 2)).$((local_rank % 2))"
else
 export ZE_AFFINITY_MASK="$local_rank"
fi
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export KOKKOS_MEMORY_BUDGET_BYTES=$((${CZ_KOKKOS_TOTAL_BYTES:-16000000000} / CZ_NRANKS))
export KOKKOS_MEMORY_REPORT="$CZ_OUT/memory-$rank.json"
exec > "$CZ_OUT/rank-$rank.log" 2>&1
printf 'rank=%s host=%s affinity=%s\n' "$rank" "$(hostname)" "$ZE_AFFINITY_MASK"
exec /usr/bin/python3.11 source/scripts/tests/run_bounded.py --seconds "${CZ_SECONDS:-420}" --memory-gb "$CZ_RSS_GB" --gpu-allocations --report "$CZ_OUT/resources-$rank.json" -- "$@"
