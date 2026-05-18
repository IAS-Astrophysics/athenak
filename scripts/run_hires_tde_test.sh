#!/bin/bash
set -o pipefail

RUN_DIR=$1
INPUT=/home/hz0693/athenak/inputs/dyngr/z4c_tov_ks_sunlike_pwp_tde_hydro_4rt_hires.athinput
ATHENA=/home/hz0693/athenak/build_cuda_mpi_z4c_tov_ks/src/athena

cd /home/hz0693/athenak || exit 1
echo "RUN_DIR=${RUN_DIR}"
date
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

srun --nodes=1 \
  --ntasks=4 \
  --cpus-per-task=12 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${ATHENA}" \
  -i "${INPUT}" \
  -d "${RUN_DIR}" \
  -t 00:55:00 2>&1 | tee -a "${RUN_DIR}/run.log"
