#!/bin/bash
# Source this file. Match the validated SANE continuation build/runtime split.
module load gcc/14.3.0 oneapi/release/2026.1.0
if [[ ${1:-runtime} == build ]]; then
  module load mpich/prd/5.0.0.aurora_test.87e2045
else
  module load mpich/prd/5.0.0.aurora_test.51a9474
fi
module load libfabric/2.3.1 cray-pals/1.8.0 cray-libpals/1.8.0 cmake/3.31.11
export MPICH_GPU_SUPPORT_ENABLED=1
export MPIR_CVAR_CH4_IPC_ZE_SHAREABLE_HANDLE=pidfd
export MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=1024
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export OMP_NUM_THREADS=8 OMP_PLACES=threads OMP_PROC_BIND=spread
export FI_MR_CACHE_MONITOR=disabled FI_LOG_LEVEL=ERROR
