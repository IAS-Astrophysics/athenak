# AthenaK

Experimental version of [Athena++](https://github.com/PrincetonUniversity/athena) with [Kokkos](https://github.com/kokkos/kokkos).

## How to clone

It is important to ensure that Kokkos, which is included as a Git submodule within the AthenaK source code, is cloned alongside the main project repository:
```shell
git clone --recurse-submodules -j2 git@github.com:IAS-Astrophysics/athenak.git
```
For older verisons of Git, replace the options with `--recursive`. If you clone the code without Kokkos (or download the repository as a `.zip` or release tarball), you must get Kokkos manually in the root directory containing the code:
```shell
git submodule init
git submodule update
```
We also recommend applying `scratch_fix.patch` to Kokkos. This can be done as follows:
```shell
cd kokkos
git apply ../scratch_fix.patch
```
This patch fixes a small bug in Kokkos 4.1 with uninitialized CUDA scratch locks which can result in crashes when using level 1 scratch memory with NVIDIA GPUs.

## How to build

The code uses CMake to manage builds.  In-source builds are not allowed; you must create a new build directory:
```shell
mkdir build
cd build
```

Then run `cmake` (version 3.0 or later) for the specific target architecture in the build subdirectory as follows:

### Default build for CPU
```shell
cmake3 ../
```

### Build for CPU with custom problem generator (located in `/src/pgen/name.cpp`)
```shell
cmake3 -D PROBLEM=name ../
```

### To build in debug mode, add
```shell
cmake3 -D CMAKE_BUILD_TYPE=Debug ../
```

### Default build for CPU with MPI
```shell
cmake3 -D Athena_ENABLE_MPI=ON ../
```

### Default build for Intel Broadwell CPU with the Classic Intel C++ compiler and GCC C compiler 
```shell
cmake3 -DCMAKE_CXX_COMPILER=icpc -D Kokkos_ARCH_BDW=On -D Kokkos_ENABLE_OPENMP=ON \
   -D CMAKE_CXX_FLAGS="-O3 -inline-forceinline -qopenmp-simd -qopt-prefetch=4 -diag-disable 3180 " \
   -D CMAKE_C_FLAGS="-O3 -finline-functions" ../
```

### Debug build for NVIDIA A100 GPU (requires GCC and CUDA Toolkit)
E.g. `apollo` at IAS:
```shell
cmake3 -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On -DCMAKE_CXX_COMPILER=${path_to_code}/athenak/kokkos/bin/nvcc_wrapper -D CMAKE_BUILD_TYPE=DEBUG ../
```

### Default build for NVIDIA V100 GPU (requires GCC and CUDA Toolkit)
E.g. `cuda` at IAS:
```shell
cmake3 -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_VOLTA70=On -DCMAKE_CXX_COMPILER=${path_to_code}/athenak/kokkos/bin/nvcc_wrapper ../
```

### Default build for Intel Dat

### Build One Puncture problem for CPU

```shell
cmake ../ -DPROBLEM=z4c_one_puncture 
```

### Build Two Punctures problem for CPU
For this you need first to install two external libraries, the GNU Scientific Library (GSL) and TwoPunctures inital data solver. 

#### GSL:
```shell
cd $HOME && mkdir -p usr/gsl && mkdir codes && cd codes
# grab source
wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz

# extract and configure for local install
tar -zxvf gsl-2.5.tar.gz

cd gsl-2.5

./configure --prefix=--prefix=/installation/path/usr/gsl

make -j8
# make check
make install

# link gsl into athenak
ln -s /installation/path/usr/gsl ${path_to_code}
```
#### `twopuncturesc`:
```shell
cd $HOME && cd usr
git clone git@github.com:computationalrelativity/TwoPuncturesC.git
cd twopuncturesc
make -j8

# link twopuncturesc into athenak
ln -s /installation/path/usr/twopunctures ${path_to_code}
```
Now create build directory and configure with CMake

```
mkdir build_z4c_twopunc && cd build_z4c_twopunc
cmake ../ -DPROBLEM=z4c_two_puncture
```
