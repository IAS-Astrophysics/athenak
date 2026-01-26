#!/bin/bash
# PROB='blast'  # test using standard blast problem
PROB='disk-magnetosphere'  # problem to build
ARCHDEFAULT="H200"

# Check for architecture argument
if [ -z "$1" ]; then
    ARCH=$ARCHDEFAULT
else
    ARCH=$1
fi

echo "Building for architecture: $ARCH"

# Format build directory name
BUILD_DIR="build_${ARCH}"
mkdir $BUILD_DIR

# H200 architecture
if [[ "$ARCH" == "H200" ]]; then
    source athenak_h200
    cmake3 -DPROBLEM=$PROB \
           -DKokkos_ENABLE_CUDA=On \
           -DKokkos_ARCH_HOPPER90=On \
           -DCMAKE_CXX_COMPILER=/home/cfairbairn/Projects/Magnetosphere/athenak/kokkos/bin/nvcc_wrapper \
           -DAthena_ENABLE_MPI=On \
           -B $BUILD_DIR
fi

# A100 architecture
if [[ "$ARCH" == "A100" ]]; then
    source athenak_a100
    cmake3 -DPROBLEM=$PROB \
          -DKokkos_ENABLE_CUDA=On \
          -DKokkos_ARCH_AMPERE80=On \
          -DCMAKE_CXX_COMPILER=/home/cfairbairn/Projects/Magnetosphere/athenak/kokkos/bin/nvcc_wrapper \
          -DAthena_ENABLE_MPI=On \
          -B $BUILD_DIR
fi

# Typhon architecture (cascade lake nodes)
if [[ "$ARCH" == "typhon" ]]; then
    source athenak_typhon
    cmake3 -DPROBLEM=$PROB \
    -D Kokkos_ARCH_SKX=On \
    -D Athena_ENABLE_MPI=On \
    -B $BUILD_DIR
fi

# My Macbook architecture
if [[ "$ARCH" == "mac" ]]; then
    cmake -DPROBLEM=$PROB -DKokkos_ARCH_ARMV81=On -DAthena_ENABLE_MPI=On ../
fi

cd $BUILD_DIR

