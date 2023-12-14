#!/bin/bash
### Example script to run jobs on cluster with SLURM

#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jmstone@ias.edu

module purge
module load rh/devtoolset/7
module load openmpi/gcc/3.0.3/64

srun /home/jmstone/athenak/build/src/athena -i linear_wave.athinput time/nlim=100
