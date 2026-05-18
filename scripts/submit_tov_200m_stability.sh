#!/bin/bash
#SBATCH --job-name=tov200_stab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/hz0693/athenak/runs/tov_200m_stability_%j.out
#SBATCH --error=/home/hz0693/athenak/runs/tov_200m_stability_%j.err

set -euo pipefail

repo=/home/hz0693/athenak
run_root="${repo}/runs/tov_200m_stability_${SLURM_JOB_ID}"

mkdir -p "${run_root}"
cd "${repo}"
echo "${run_root}" > "${repo}/runs/tov_200m_stability_latest.txt"

date
echo "RUN_ROOT=${run_root}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python3 analysis/run_tov_200m_stability.py \
  --athena "${repo}/build_cuda_nompi_z4c_tov_ks/src/athena" \
  --runner-prefix "srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-12} --gpus-per-task=1" \
  --run-root "${run_root}" \
  --tlim 200.0 \
  --hst-dt 1.0 \
  --wall-time 02:55:00 \
  --boost 0.2 \
  --cases unboosted_L5_n80,boosted_L5_n80,unboosted_L10_n80_amr1,boosted_L10_n80_amr1

date
