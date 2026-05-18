#!/bin/bash
#SBATCH --job-name=tov_mink_damp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/hz0693/athenak/runs/tov_minkowski_damping_sweep_%j.out
#SBATCH --error=/home/hz0693/athenak/runs/tov_minkowski_damping_sweep_%j.err

set -euo pipefail

repo=/home/hz0693/athenak
root="${repo}/runs/tov_minkowski_damping_sweep_${SLURM_JOB_ID}"
cases=unboosted_L3_n48,boosted_L3_n48,unboosted_L5_n80,boosted_L5_n80

mkdir -p "${root}"
cd "${repo}"
echo "${root}" > "${repo}/runs/tov_minkowski_stability_latest.txt"

date
echo "RUN_ROOT=${root}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

run_config() {
  local label=$1
  local diss=$2
  local kappa1=$3
  local kappa2=$4
  local out="${root}/${label}"
  mkdir -p "${out}"
  echo "CONFIG ${label}: diss=${diss}, kappa1=${kappa1}, kappa2=${kappa2}"
  TOV_TLIM=20.0 \
  TOV_CASE_WALL_TIME=00:20:00 \
  TOV_CASES="${cases}" \
  TOV_Z4C_DISS="${diss}" \
  TOV_Z4C_KAPPA1="${kappa1}" \
  TOV_Z4C_KAPPA2="${kappa2}" \
    "${repo}/scripts/run_tov_stability_gpu.sh" "${out}"
}

run_config no_diss_no_damp 0.0 0.0 0.0
run_config diss_0p1_no_damp 0.1 0.0 0.0
run_config diss_0p5_no_damp 0.5 0.0 0.0
run_config diss_0p5_k1_0p02_k2_0 0.5 0.02 0.0
run_config diss_0p5_k1_0p02_k2_0p02 0.5 0.02 0.02

date
