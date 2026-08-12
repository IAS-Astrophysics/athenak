#!/bin/bash
#SBATCH --job-name=celephais-bns-build
#SBATCH --account=bw25h005
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/10/project/bw25h005/athenak-celephais-batch/logs/build-%j.out
#SBATCH --error=/pfs/10/project/bw25h005/athenak-celephais-batch/logs/build-%j.err

set -euo pipefail

readonly experiment=/pfs/10/project/bw25h005/athenak-celephais-batch
readonly athenak_source="${HOME}/ptmp/athenak"
readonly celephais_source="${HOME}/ptmp/Celephais-batch"
readonly celephais_build="${experiment}/build/celephais"
readonly celephais_prefix="${experiment}/install/celephais"
readonly athenak_build="${experiment}/build/athenak"

source "${celephais_source}/cmake/env/binac2.sh"

cmake -S "${celephais_source}" -B "${celephais_build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${celephais_prefix}" \
  -DCELEPHAIS_LIB_OUTPUT_DIR="${celephais_build}/lib" \
  -DBUILD_APPS=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_TOOLCHAIN_FILE="${celephais_source}/cmake/toolchains/binac2.cmake"
cmake --build "${celephais_build}" --target celephais --parallel "${SLURM_CPUS_PER_TASK}"
cmake --install "${celephais_build}"

cmake -S "${athenak_source}" -B "${athenak_build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_SERIAL=ON \
  -DPROBLEM=dyn_grmhd/celephais/celephais_bns \
  -DCELEPHAIS_SOURCE_DIR="${celephais_source}" \
  -DCELEPHAIS_BINARY_DIR="${celephais_build}" \
  -DCMAKE_PREFIX_PATH="${celephais_prefix}"
cmake --build "${athenak_build}" --target athena --parallel "${SLURM_CPUS_PER_TASK}"
