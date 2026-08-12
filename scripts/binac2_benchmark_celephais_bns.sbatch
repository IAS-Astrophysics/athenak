#!/bin/bash
#SBATCH --job-name=celephais-bns-ab
#SBATCH --account=bw25h005
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/pfs/10/project/bw25h005/athenak-celephais-batch/logs/benchmark-%j.out
#SBATCH --error=/pfs/10/project/bw25h005/athenak-celephais-batch/logs/benchmark-%j.err

set -euo pipefail

readonly experiment=/pfs/10/project/bw25h005/athenak-celephais-batch
readonly celephais_runtime="${experiment}/runtime/Celephais"
readonly athena="${experiment}/build/athenak/src/athena"
readonly input_source="${experiment}/runtime/athenak/inputs/dyn_grmhd/celephais_bns.athinput"
readonly id_source=/pfs/10/project/bw25h005/res19_imbal/converged_BNS_NOSYM_QUASI_EQUIL.dd2.35.0.3.z30.0.z0.2.8.q1.0.0.0.15
readonly table_source=/pfs/10/project/bw25h005/DD2_T0.1_beta_m931.athtab
readonly run_root="${experiment}/bench/bns/run_${SLURM_JOB_ID}"
readonly node_runtime="/tmp/celephais-bns-${SLURM_JOB_ID}"

source "${celephais_runtime}/cmake/env/binac2.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "${run_root}" "${node_runtime}/Celephais/data" "${node_runtime}/id"
rsync -a "${celephais_runtime}/data/" "${node_runtime}/Celephais/data/"
rsync -a "${input_source}" "${node_runtime}/celephais_bns.athinput"
rsync -a "${id_source}.toml" "${id_source}.dat" "${node_runtime}/id/"
rsync -a "${table_source}" "${node_runtime}/"
export HOME_CELEPHAIS="${node_runtime}/Celephais"

readonly input="${node_runtime}/celephais_bns.athinput"
readonly initial_data="${node_runtime}/id/$(basename "${id_source}").toml"
readonly eos_table="${node_runtime}/$(basename "${table_source}")"

run_case() {
  local label="$1"
  local batch_fields="$2"
  local run_dir="${run_root}/${label}"
  mkdir -p "${run_dir}"
  cd "${run_dir}"
  echo "[celephais-case] label=${label} batch_fields=${batch_fields}"
  /usr/bin/time -p srun --mpi=pmix "${athena}" -i "${input}" \
    problem/batch_fields="${batch_fields}" \
    problem/initial_data_file="${initial_data}" problem/table="${eos_table}" \
    mhd/table="${eos_table}" time/nlim=1 2>&1 | tee run.log
}

# Counterbalanced order reduces first-run and node-drift bias.
run_case baseline_1 false
run_case batch_1 true
run_case batch_2 true
run_case baseline_2 false

for run_dir in "${run_root}"/*; do
  find "${run_dir}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.hst' \) -print0 \
    | sort -z | xargs -0 -r sha256sum \
    | sed "s#  ${run_dir}/#  #" > "${run_dir}/output.sha256"
  test -s "${run_dir}/output.sha256"
done

for label in batch_1 batch_2 baseline_2; do
  diff -u "${run_root}/baseline_1/output.sha256" \
          "${run_root}/${label}/output.sha256"
done
echo "[celephais-check] all baseline and batch outputs are identical"
