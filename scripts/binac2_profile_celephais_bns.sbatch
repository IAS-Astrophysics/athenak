#!/bin/bash
#SBATCH --job-name=celephais-bns-profile
#SBATCH --account=bw25h005
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/10/project/bw25h005/athenak-celephais-batch/logs/profile-%j.out
#SBATCH --error=/pfs/10/project/bw25h005/athenak-celephais-batch/logs/profile-%j.err

set -euo pipefail

readonly experiment=/pfs/10/project/bw25h005/athenak-celephais-batch
readonly celephais_runtime="${experiment}/runtime/Celephais"
readonly athena_source="${experiment}/build/profile-athenak/src/athena"
readonly input_source="${HOME}/ptmp/athenak/inputs/dyn_grmhd/celephais_bns.athinput"
readonly output_comparator="${HOME}/ptmp/athenak/scripts/compare_celephais_outputs.py"
readonly id_source=/pfs/10/project/bw25h005/res19_imbal/converged_BNS_NOSYM_QUASI_EQUIL.dd2.35.0.3.z30.0.z0.2.8.q1.0.0.0.15
readonly table_source=/pfs/10/project/bw25h005/DD2_T0.1_beta_m931.athtab
readonly run_root="${experiment}/bench/bns/profile_${SLURM_JOB_ID}"
readonly node_runtime="/tmp/celephais-bns-profile-${SLURM_JOB_ID}"
readonly reference_run="${experiment}/bench/bns/skip_velocity_4542312/control_1"

source "${celephais_runtime}/cmake/env/binac2.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "${run_root}" "${node_runtime}/Celephais/data" "${node_runtime}/id"
rsync -a "${celephais_runtime}/data/" "${node_runtime}/Celephais/data/"
rsync -a "${athena_source}" "${node_runtime}/athena"
rsync -a "${input_source}" "${node_runtime}/celephais_bns.athinput"
rsync -a "${id_source}.toml" "${id_source}.dat" "${node_runtime}/id/"
rsync -a "${table_source}" "${node_runtime}/"
export HOME_CELEPHAIS="${node_runtime}/Celephais"
export GMON_OUT_PREFIX="${run_root}/gmon"

readonly athena="${node_runtime}/athena"
readonly input="${node_runtime}/celephais_bns.athinput"
readonly initial_data="${node_runtime}/id/$(basename "${id_source}").toml"
readonly eos_table="${node_runtime}/$(basename "${table_source}")"

cd "${run_root}"
/usr/bin/time -p srun --mpi=pmix "${athena}" -i "${input}" \
  problem/batch_fields=true \
  problem/skip_vacuum_velocity=true \
  problem/initial_data_file="${initial_data}" problem/table="${eos_table}" \
  mhd/table="${eos_table}" time/nlim=1 2>&1 | tee run.log

python3 "${output_comparator}" "${reference_run}" "${run_root}"

gprof -b "${athena}" "${run_root}"/gmon.* > gprof.txt
echo "[celephais-profile] output matches Stage-I batch and gprof.txt is ready"
