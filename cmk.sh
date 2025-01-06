module purge
module load modules/2.1.1 slurm cuda/11.8.0 openmpi/cuda-4.0.7
export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

athenak=/mnt/home/msiwek/software/athenak/
build=$athenak/build

cmake \
  -D CMAKE_CXX_COMPILER=$athenak/kokkos/bin/nvcc_wrapper \
  -D Kokkos_ENABLE_CUDA=On \
  -D Kokkos_ARCH_AMPERE80=On \
  -D Athena_ENABLE_MPI=On \
  -B $build \
  -S $athenak

cd $build
make -j 4