athenak=/mnt/home/msiwek/software/athenak/
build=$athenak/build

source $athenak/envs.sh #load modules needed to build athenak
export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

#for 'make clean' (doesn't exist in cmake), probably have to delete everything in build directory \

cmake \
  -D CMAKE_CXX_COMPILER=$athenak/kokkos/bin/nvcc_wrapper \
  -D Kokkos_ENABLE_CUDA=On \
  -D Kokkos_ARCH_AMPERE80=On \
  -D Athena_ENABLE_MPI=On \
  -B $build \
  -S $athenak

cd $build
make -j 4