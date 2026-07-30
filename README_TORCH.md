# Building AthenaK with LibTorch (Rhea flavor mixing)

This describes the `Athena_ENABLE_TORCH` CMake option, added to support the Rhea
ML flavor-mixing model in `radiation_m1` (`flavor_mix = rhea`). This file covers how to
configure and build against LibTorch; see `radiation_m1_rhea.hpp`/`radiation_m1_rhea.cpp`
for the interop layer itself.

## Version pin

LibTorch **2.6.0** (matches Rhea's own Python/training pin). A version bump is deferred;
treat a newer or older release as untested until it has been validated against Rhea's own
Python/training pin.

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
  submodule at **>= 4.7** (already the case on this branch). **Builds, links, and runs
  correctly on a PVC** as of 2026-07-29 — see "Validated on Aurora" below.
  ```
  cmake -DKokkos_ENABLE_SYCL=On -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=On \
        -DKokkos_ARCH_INTEL_PVC=On -DCMAKE_CXX_COMPILER=icpx \
        -DAthena_ENABLE_NURATES=On -DAthena_ENABLE_TORCH=On \
        -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
  ```
  On Aurora, `module load frameworks` is where the XPU-enabled LibTorch lives — but do **not**
  load it, at build time or run time; point `CMAKE_PREFIX_PATH` at its
  `.../torch/share/cmake` directly and keep the ordinary production module stack. Scripted as
  `batchtools/templates/athenak/aurora/{environment,configure}-rhea.sh`; see "LibTorch without
  the frameworks module" below for why this matters and the evidence that it works.
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
misbehaves at runtime. AthenaK's own CI only exercises CUDA+`nvcc_wrapper`.

**Resolved for icpx/SYCL:** this did *not* bite on Aurora (2026-07-29, oneAPI 2025.3.1 +
frameworks-module torch `2.10.0a0`). Configure, compile, and link all succeeded with no
`_GLIBCXX_USE_CXX11_ABI` override needed and no undefined `std::__cxx11::...` symbols. The
one real portability defect the icpx build exposed was unrelated to ABI: `ResolveDevice()` in
`radiation_m1_rhea.cpp` called `Kokkos::SYCL::device_id(exec)`, but `device_id(exec)` is a
member of `Kokkos::Tools::Experimental::DeviceTypeTraits<Space>`, not of the space classes —
the CUDA and HIP branches had the same mistake and are now fixed too, though still uncompiled.
Every problem found after that point was in the *launch environment*, not the code (see
"Validated on Aurora" above).
`hipcc` remains without in-repo precedent. Check the actual `TORCH_CXX_FLAGS` your
LibTorch distribution sets (it was empty for the CPU-only conda-forge build used to validate
this CMake wiring — do not assume it is always empty; official pytorch.org distributions do
set it explicitly per ABI variant).

## Validated on Aurora (2026-07-29)

The SYCL/XPU interop path in `radiation_m1_rhea.cpp` was designed against documentation and
headers only; it has now been **executed on real PVC hardware** (oneAPI 2025.3.1,
frameworks-module torch `2.10.0a0+git449b176`, one node of the `debug` queue, a single tile).
The Rhea single-zone test with the trained checkpoint reproduces the model's asymptotic
prediction and agrees with an x86 CPU `Kokkos::Serial` build of the same commit to
**2.9e-7 relative** — the float32 floor of the network itself — with per-sector `N` conserved
to 2.2e-16. `predict_all` alone, run from Python on the same node, agrees CPU-vs-XPU to
4.5e-6.

Two results worth recording explicitly:

- **The USM-context concern did not materialise.** `Predict()` hands a raw Kokkos device
  pointer to `torch::from_blob(..., TensorOptions().device(kXPU, i))`, and SYCL USM
  allocations are context-bound: Kokkos constructs its queue as `sycl::queue{device, ...}`
  (`kokkos/core/src/SYCL/Kokkos_SYCL_Instance.cpp`) while Torch constructs its own, so nothing
  in either API guarantees they share a `sycl::context`. In practice Intel's DPC++ hands both
  the platform default context and the borrow is valid. This is an *observation on one
  configuration*, not a guarantee — if a future oneAPI or torch build faults at the first
  `predict_all`, this is the first thing to check
  (`DevExeSpace().sycl_queue().get_context()` vs
  `c10::xpu::getCurrentXPUStream().queue().get_context()`).
- `c10::xpu::getStreamFromExternal` and `XPUCachingAllocator::setMemoryFraction` behave as
  expected against torch `2.10.0a0`, not just the 2.12.1 headers they were written against.

MPI is validated as of 2026-07-29 too (`-DAthena_ENABLE_MPI=ON`): two ranks, one MeshBlock each,
each constructing its own `RheaModel` and loading the checkpoint onto its own device, give
**bit-identical** results to the 1-rank run over all 61 cycles — both when the ranks share one
GPU's two tiles (`ZE_AFFINITY_MASK=0.0/0.1`) and when they occupy two distinct GPUs (`0/1`). The
per-rank device binding that `ResolveDevice()` relies on therefore holds under both layouts, which
is what matters: `gpu_tile_compact.sh`/`gpu_dev_compact.sh` leave each rank exactly one visible
device, so Kokkos's and Torch's indices are both 0 and cannot diverge.

Use `file_type = bin`, not `tab`, for any multi-rank run: the tab writer has every rank `fopen`
the same file in append mode with buffered `fprintf`s serialised only by `MPI_Barrier`/`fflush`
(`outputs/formatted_table.cpp:105-180`), whereas the binary writer does MPI-IO collective writes
at per-rank offsets (`outputs/binary.cpp:240-284`). Note bin is float32 (`binary.cpp:224` casts
regardless of `Real`), so it resolves ~1e-7 rather than tab's `%24.17e`.

### LibTorch without the frameworks module

`module load frameworks` is the obvious way to get an XPU LibTorch on Aurora, and it is the wrong
thing to have in a job. It is also unnecessary — at **either** stage. Verified 2026-07-29:

- **Runtime**: `ldd src/athena` reports **zero** unresolved libraries with only the production
  module stack (`boost`/`fftw`/`cmake`, i.e. `environment.sh`). The linked binary's `RUNPATH`
  already covers `.../torch/lib`, and `libsycl`/`libmkl` come from `oneapi/release/2025.3.1`,
  which Aurora loads *by default*.
- **Configure**: `find_package(Torch)` succeeds with
  `-DCMAKE_PREFIX_PATH=<torch-prefix>/share/cmake` and no module loaded. `TorchConfig.cmake` and
  `Caffe2Config.cmake` search only inside `TORCH_INSTALL_PREFIX`, `Caffe2Targets.cmake` names no
  path outside it, SYCL is found from the default oneAPI, and `TORCH_CXX_FLAGS` comes back
  **empty** — independently re-confirming there is no ABI flag to reconcile here.
- `frameworks` does **not** swap `mpich` or `oneapi`: both stacks carry
  `mpich/opt/5.0.0.aurora_test.3c70a61` and `oneapi/release/2025.3.1`. So this choice is
  orthogonal to enabling MPI.

Why it matters is the next section: the module mutates two `ZE_*`/`ONEAPI_*` variables in ways
that break the standard Aurora launch recipe and undermine an assumption this port relies on.
Keeping it out of the environment makes both problems disappear rather than requiring a
workaround, because the *site defaults* are already the values we want (`COMPOSITE`,
`level_zero:gpu`).

### Aurora launch environment: two traps

These are what you hit if you do load the module. Both cost a job apiece and neither is visible
from a login node.

1. **`module load frameworks` sets `ZE_FLAT_DEVICE_HIERARCHY=FLAT`**, under which PVC tiles are
   root devices and `ZE_AFFINITY_MASK` is a flat list (`0`, `1`, ... `11`).
   `/soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh` — which
   `batchtools/templates/athenak/aurora/batch.sub` uses — emits COMPOSITE `gpu.tile` masks
   (`0.0`, `0.1`, ...). Under FLAT those match **nothing**, and the failure is quiet: `sycl-ls`
   reports "No platforms found", `torch.xpu.device_count()` returns 0, and
   `Kokkos::initialize()` aborts with "no GPU available for execution". **`module load
   frameworks` and `gpu_tile_compact.sh` cannot simply be combined** — so don't load it, and the
   wrapper works as written. Under the site-default `COMPOSITE` hierarchy, `gpu_tile_compact.sh`
   gives each rank exactly one visible tile, which is precisely the condition that makes Kokkos's
   and Torch's device indices agree trivially (both 0). That is the right MPI story for this port.
2. **The frameworks module sets `ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"`** (for
   Triton-XPU/vLLM/Ray/dpctl; it prints this on stderr and sanctions reverting). Exposing two
   backends makes SYCL enumerate every physical GPU twice, and this port's correctness rests on
   Kokkos's device index and Torch's XPU index naming the same physical device from two
   independently built enumerations. The site default is already `level_zero:gpu` alone, so once
   again the fix is to not load the module; set `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` explicitly
   if you must have it loaded.

Also note `set -u` in a job script makes `module load` abort immediately — Lmod's bash init
reads `ZSH_EVAL_CONTEXT` unconditionally.

The staged bring-up script that established all of the above (device probing, Kokkos-without-
Torch, Torch-without-AthenaK, then the test itself, each stage's exit status recorded rather
than aborting the job) is `scripts/aurora_rhea_singlezone.sub` in the NeuOscillations
superproject.

## Verifying the build

With `Athena_ENABLE_TORCH=OFF` (default), the binary has zero LibTorch dependency
(`ldd athena | grep -i torch` returns nothing) and behavior is unaffected. With it `ON`,
`ldd athena` should show `libtorch.so`/`libtorch_cpu.so`/`libc10.so` (plus CUDA/HIP/SYCL
backend libraries as applicable) resolved from your `CMAKE_PREFIX_PATH`. Running e.g.
`inputs/hydro/sod.athinput` should produce bit-identical output between `ON` and `OFF`
builds, since problems that don't set `flavor_mix = rhea` never touch LibTorch; see
`inputs/tests/rad_m1_rhea_singlezone.athinput` and `inputs/tests/rad_m1_tov_rhea.athinput`
for problems that exercise the Rhea mixing path itself:
```
python3 scripts/make_toy_rhea_model.py toy_flavor_swap.pt --gamma-code 10.0      # single-zone
python3 scripts/make_toy_rhea_model.py toy_flavor_swap_tov.pt --gamma-code 1.0   # both TOV inputs
```
