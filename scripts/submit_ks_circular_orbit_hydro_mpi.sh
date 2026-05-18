#!/bin/bash
#SBATCH --job-name=ks_orbit_hydro
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --constraint=nomig&gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/hz0693/athenak/runs/ks_circular_orbit_hydro_%j.out
#SBATCH --error=/home/hz0693/athenak/runs/ks_circular_orbit_hydro_%j.err

set -euo pipefail

repo=/home/hz0693/athenak
input="${repo}/inputs/dyngr/z4c_tov_ks_circular_orbit_hydro.athinput"
athena="${repo}/build_cuda_mpi_z4c_tov_ks/src/athena"
run_dir="${repo}/runs/ks_circular_orbit_hydro_${SLURM_JOB_ID}"

mkdir -p "${run_dir}"
cd "${repo}"
cp "${input}" "${run_dir}/used_input.athinput"
echo "${run_dir}" > "${repo}/runs/ks_circular_orbit_hydro_latest.txt"

date
echo "RUN_DIR=${run_dir}"
echo "ATHENA=${athena}"
echo "INPUT=${input}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

srun --nodes=1 \
  --ntasks=4 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${athena}" \
  -i "${input}" \
  -d "${run_dir}" \
  -t 00:55:00 2>&1 | tee -a "${run_dir}/run.log"

date
