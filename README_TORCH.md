# Building AthenaK with LibTorch (Rhea flavor mixing)

This describes the `Athena_ENABLE_TORCH` CMake option, added to support the Rhea
ML flavor-mixing model in `radiation_m1` (`flavor_mix = rhea`). Full design context is in
[`rhea_athenak_port_design.md`](../rhea_athenak_port_design.md) (repo root, one level up from
this file) — this file only covers how to configure and build.

## Version pin

LibTorch **2.6.0** (matches Rhea's own Python/training pin). Do not use a newer or older
version without checking the design doc §7/§11 first — a version bump is deferred.

## Getting LibTorch

Download a prebuilt distribution matching your target backend from
[pytorch.org](https://pytorch.org/get-started/locally/) (select "LibTorch", C++/Java, and
the CUDA/ROCm/CPU variant you need), or point at an existing install (e.g. the LibTorch that
ships inside a PyTorch Python environment — see below).

## CMake flags

- `-DAthena_ENABLE_TORCH=ON` (default `OFF`) enables the option.
- CMake needs to find LibTorch's own CMake package. Provide one of:
  - `-DCMAKE_PREFIX_PATH=<path-to-libtorch>` (standard LibTorch convention), or
  - `-DTorch_DIR=<path-to-libtorch>/share/cmake/Torch`.
- If pointing at the LibTorch bundled inside a PyTorch Python install, use:
  ```
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')"
  ```
  Some conda/conda-forge PyTorch builds additionally require the environment's root prefix
  itself on `CMAKE_PREFIX_PATH` (semicolon-separated) so CMake can resolve LibTorch's own
  `find_package(protobuf)` dependency, e.g.:
  ```
  -DCMAKE_PREFIX_PATH="<torch-cmake-dir>;<conda-env-root>"
  ```
  This was hit when building against a conda-forge LibTorch locally; not needed for the
  official pytorch.org prebuilt zips, which vendor their own dependencies.

## Per-backend configure lines

- **CUDA (e.g. Perlmutter/A100, `sm_80`)**:
  ```
  cmake -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On \
        -DCMAKE_CXX_COMPILER=$(pwd)/kokkos/bin/nvcc_wrapper \
        -DAthena_ENABLE_MPI=On -DAthena_ENABLE_TORCH=On \
        -DCMAKE_PREFIX_PATH=<path-to-libtorch-2.6.0-cu*> ..
  ```
- **HIP** (no AMD architecture flag defaulted — set `Kokkos_ARCH_<target>` for your GPU):
  ```
  cmake -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_<target>=On \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DAthena_ENABLE_TORCH=On -DCMAKE_PREFIX_PATH=<path-to-libtorch-2.6.0-rocm*> ..
  ```
- **SYCL (Aurora, Intel Data Center GPU Max / PVC)**: requires the vendored `kokkos`
  submodule at **>= 4.7** (already the case on this branch as of the Package 0 commit).
  ```
  cmake -DKokkos_ENABLE_SYCL=On -DKokkos_ARCH_INTEL_PVC=On \
        -DCMAKE_CXX_COMPILER=icpx \
        -DAthena_ENABLE_TORCH=On -DCMAKE_PREFIX_PATH=<path-to-libtorch-2.6.0-xpu> ..
  ```
- **CPU (Serial or OpenMP)** — fastest path for local iteration/unit tests:
  ```
  cmake -DKokkos_ENABLE_SERIAL=On \
        -DAthena_ENABLE_TORCH=On -DCMAKE_PREFIX_PATH=<path-to-libtorch-2.6.0-cpu> ..
  ```

Then build as usual: `cmake --build . -j <N>`.

## Known friction: ABI flags

`find_package(Torch)` propagates its own `CMAKE_CXX_FLAGS` and `_GLIBCXX_USE_CXX11_ABI`
setting through the imported `torch` target. This **must** agree with whatever Kokkos and
the device compiler (`nvcc_wrapper`/`hipcc`/`icpx`) were built with, or you get either a hard
link error (undefined `std::__cxx11::...` symbols) or a silent ODR violation that only
misbehaves at runtime. AthenaK's own CI only exercises CUDA+`nvcc_wrapper`; there is no
in-repo precedent yet for `find_package(Torch)` combined with `hipcc`/`icpx` — budget time
for this on the first HIP/SYCL build attempt. Check the actual `TORCH_CXX_FLAGS` your
LibTorch distribution sets (it was empty for the CPU-only conda-forge build used to validate
this CMake wiring — do not assume it is always empty; official pytorch.org distributions do
set it explicitly per ABI variant).

## Verifying the build

With `Athena_ENABLE_TORCH=OFF` (default), the binary has zero LibTorch dependency
(`ldd athena | grep -i torch` returns nothing) and behavior is unaffected. With it `ON`,
`ldd athena` should show `libtorch.so`/`libtorch_cpu.so`/`libc10.so` (plus CUDA/HIP/SYCL
backend libraries as applicable) resolved from your `CMAKE_PREFIX_PATH`. Running e.g.
`inputs/hydro/sod.athinput` should produce bit-identical output between `ON` and `OFF`
builds, since no code path is wired up to actually use LibTorch yet beyond the placeholder
in `radiation_m1.hpp`/`radiation_m1.cpp` — see the design doc's work-breakdown (§10) for
what each subsequent package adds.
