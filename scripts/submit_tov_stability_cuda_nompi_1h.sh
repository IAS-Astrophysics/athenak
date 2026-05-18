#!/bin/bash
#SBATCH --job-name=tov_mink_nompi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/hz0693/athenak/runs/tov_minkowski_stability_nompi_%j.out
#SBATCH --error=/home/hz0693/athenak/runs/tov_minkowski_stability_nompi_%j.err

set -euo pipefail

repo=/home/hz0693/athenak
run_root="${repo}/runs/tov_minkowski_stability_nompi_${SLURM_JOB_ID}"

mkdir -p "${run_root}"
cd "${repo}"
echo "${run_root}" > "${repo}/runs/tov_minkowski_stability_latest.txt"

date
echo "RUN_ROOT=${run_root}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

TOV_TLIM=5.0 \
TOV_CASE_WALL_TIME=00:12:00 \
TOV_CASES=unboosted_L3_n32,unboosted_L3_n48,unboosted_L3_n64,boosted_L3_n32,boosted_L3_n48,boosted_L3_n64,unboosted_L5_n80,boosted_L5_n80 \
  "${repo}/scripts/run_tov_stability_gpu.sh" "${run_root}"

date
