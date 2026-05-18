#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hz0693/athenak
run_root=${1:-"${repo_root}/runs/tov_minkowski_stability_gpu_${SLURM_JOB_ID:-manual}"}
athena=${ATHENA_EXE:-"${repo_root}/build_cuda_nompi_z4c_tov_ks/src/athena"}
cases=${TOV_CASES:-unboosted_L3_n32,unboosted_L3_n48,unboosted_L3_n64,boosted_L3_n32,boosted_L3_n48,boosted_L3_n64,unboosted_L5_n80,boosted_L5_n80}
tlim=${TOV_TLIM:-5.0}
wall_time=${TOV_CASE_WALL_TIME:-00:20:00}
boost=${TOV_MODERATE_BOOST:-0.2}
z4c_diss=${TOV_Z4C_DISS:-0.0}
z4c_kappa1=${TOV_Z4C_KAPPA1:-0.0}
z4c_kappa2=${TOV_Z4C_KAPPA2:-0.0}

cd "${repo_root}"

python3 analysis/run_tov_stability.py \
  --athena "${athena}" \
  --runner-prefix "srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-12} --gpus-per-task=1" \
  --run-root "${run_root}" \
  --cases "${cases}" \
  --tlim "${tlim}" \
  --wall-time "${wall_time}" \
  --moderate-boost "${boost}" \
  --z4c-diss "${z4c_diss}" \
  --z4c-kappa1 "${z4c_kappa1}" \
  --z4c-kappa2 "${z4c_kappa2}"
