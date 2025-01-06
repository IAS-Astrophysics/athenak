source envs.sh #load modules needed to build athenak
export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

athenak=/mnt/home/msiwek/software/athenak/
build=$athenak/build

#for 'make clean', try adding '--target clean' after -B $build \

cmake \
  -D CMAKE_CXX_COMPILER=$athenak/kokkos/bin/nvcc_wrapper \
  -D Kokkos_ENABLE_CUDA=On \
  -D Kokkos_ARCH_AMPERE80=On \
  -D Athena_ENABLE_MPI=On \
  -B $build \
  # --target clean
  -S $athenak

cd $build
make -j 4