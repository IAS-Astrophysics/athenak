#!/bin/bash
#SBATCH --job-name=celephais-bns-skip-ab
#SBATCH --account=bw25h005
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/10/project/bw25h005/athenak-celephais-batch/logs/skip-velocity-%j.out
#SBATCH --error=/pfs/10/project/bw25h005/athenak-celephais-batch/logs/skip-velocity-%j.err

set -euo pipefail

readonly experiment=/pfs/10/project/bw25h005/athenak-celephais-batch
readonly celephais_runtime="${experiment}/runtime/Celephais"
readonly athena="${experiment}/build/athenak/src/athena"
readonly input_source="${HOME}/ptmp/athenak/inputs/dyn_grmhd/celephais_bns.athinput"
readonly output_comparator="${HOME}/ptmp/athenak/scripts/compare_celephais_outputs.py"
readonly id_source=/pfs/10/project/bw25h005/res19_imbal/converged_BNS_NOSYM_QUASI_EQUIL.dd2.35.0.3.z30.0.z0.2.8.q1.0.0.0.15
readonly table_source=/pfs/10/project/bw25h005/DD2_T0.1_beta_m931.athtab
readonly run_root="${experiment}/bench/bns/skip_velocity_${SLURM_JOB_ID}"
readonly run_runtime="${run_root}/runtime"

source "${celephais_runtime}/cmake/env/binac2.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "${run_root}" "${run_runtime}/Celephais/data" "${run_runtime}/id"
rsync -a "${celephais_runtime}/data/" "${run_runtime}/Celephais/data/"
rsync -a "${input_source}" "${run_runtime}/celephais_bns.athinput"
rsync -a "${id_source}.toml" "${id_source}.dat" "${run_runtime}/id/"
rsync -a "${table_source}" "${run_runtime}/"
export HOME_CELEPHAIS="${run_runtime}/Celephais"

readonly input="${run_runtime}/celephais_bns.athinput"
readonly initial_data="${run_runtime}/id/$(basename "${id_source}").toml"
readonly eos_table="${run_runtime}/$(basename "${table_source}")"

sha256sum "${athena}" "${input}"

run_case() {
  local label="$1"
  local skip_vacuum_velocity="$2"
  local run_dir="${run_root}/${label}"
  mkdir -p "${run_dir}"
  cd "${run_dir}"
  echo "[celephais-case] label=${label} batch_fields=true skip_vacuum_velocity=${skip_vacuum_velocity}"
  /usr/bin/time -p srun --mpi=pmix --exact --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NTASKS}" --ntasks-per-node=64 --cpus-per-task=1 \
    --ntasks-per-core=1 --hint=nomultithread \
    --cpu-bind=cores "${athena}" -i "${input}" \
    problem/batch_fields=true \
    problem/skip_vacuum_velocity="${skip_vacuum_velocity}" \
    problem/initial_data_file="${initial_data}" problem/table="${eos_table}" \
    mhd/table="${eos_table}" time/nlim=1 2>&1 | tee run.log
}

# Counterbalanced pairs expose node drift while retaining the full production
# output path in every arm.
run_case control_1 false
run_case optimized_1 true
run_case optimized_2 true
run_case control_2 false

python3 "${output_comparator}" "${run_root}/control_1" \
  "${run_root}/optimized_1" "${run_root}/optimized_2" "${run_root}/control_2"
