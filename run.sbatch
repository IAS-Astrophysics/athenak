#!/usr/bin/bash
## Run athenak on rusty
#SBATCH --partition gpu
#SBATCH --constraint a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=0-1

source envs.sh #load modules needed to build athenak
export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

srun athena -i inputs/tests/bondi.athinput

# --cpus-per-task=$SLURM_CPUS_PER_TASK \
#    --cpu-bind=cores --gpu-bind=single:2 \
# bash -c "unset CUDA_VISIBLE_DEVICES; \