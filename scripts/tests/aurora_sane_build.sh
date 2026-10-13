#!/bin/bash -l
# Run against an isolated source copy with the SANE Kokkos 4.7.02 compatibility
# patch and compiler launcher copied from the validated production provenance.
set -euo pipefail
root=$(cd "$1" && pwd)
build=$2
source "$root/scripts/tests/aurora_sane_environment.sh" build
mkdir -p "$build"
module list > "$build/modules.txt" 2>&1
icpx --version > "$build/compiler.txt"
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
 -DCMAKE_CXX_COMPILER_LAUNCHER="$root/scripts/tests/sane_sycl_compile.py" \
 -DMPI_CXX_COMPILER=mpicxx -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_OPENMP=ON \
 -DOpenMP_CXX_FLAGS=-fiopenmp -DOpenMP_CXX_LIB_NAMES='iomp5;pthread' \
 -DPROBLEM=dynbbh -DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON \
 -DKokkos_ARCH_INTEL_GEN=OFF -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=OFF \
 -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=ON \
 -DKokkos_ENABLE_DEBUG=OFF -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
 -DKokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK=ON \
 -DCMAKE_CXX_FLAGS='-O3 -fsycl -fiopenmp' -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG' \
 -DCMAKE_C_FLAGS='-O3 -ffp-model=precise' \
 -DCMAKE_EXE_LINKER_FLAGS='-fsycl-max-parallel-link-jobs=8'
python3.11 "$root/scripts/tests/run_bounded.py" --seconds 1800 --memory-gb 24 \
 --report "$build/resources.json" -- cmake --build "$build" -j4
sha256sum "$build/src/athena" > "$build/executable.sha256"
