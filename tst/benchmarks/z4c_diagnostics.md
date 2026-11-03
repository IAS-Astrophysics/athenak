# Z4c diagnostic-loop benchmark

PR #790 baseline: `22baa243970fa1880b2bbc48e88a590069d55e47`.
Measured 2026-10-06 on an x86-64 environment reporting AMD EPYC 9V74,
GCC 13.3.0, Release (`-O3 -DNDEBUG`), Kokkos Serial at pinned submodule
`6739bc623081648af9e752b616d9671527922cbf`, aggressive vectorization enabled.
The two executables differ only in `src/outputs/z4c_diagnostics.hpp`.

The super-Poynting contraction changes from 243 loop iterations to 27 + 27 + 27,
with divisions reduced from 243 to 3 (source-level counts before compiler
optimization). Levi-Civita construction divides after the inner sum, reducing
source-level divisions from 81 to 27. The signs, metric and normalization are
unchanged. Two explicitly initialized nonsymmetric 3x3 temporaries hold the
raised magnetic tensor and Q; the correctness checks exercise non-diagonal
metrics as well as boosted punctures.

## Full-kernel results

Median time per active cell, excluding initialization, evolution and output I/O.
Each row includes five alternating baseline/candidate process pairs, three
bounded evolution steps, and 20 timed kernel invocations per binary after
omitting the first invocation in each process. The 32^3 sweep was pinned to CPU 0.
All cases use four allocated ghosts; derivative order is independent.

| Grid | Data | Spatial order | Baseline ns/cell | Optimized ns/cell | Baseline/optimized | Max output abs difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16³ | Rotated wave | 2 | 1899.1 | 1672.0 | 1.136 | 1.084e-19 |
| 16³ | Boosted punctures | 2 | 1866.6 | 1646.5 | 1.134 | 5.551e-17 |
| 16³ | Rotated wave | 4 | 2830.3 | 2707.8 | 1.045 | 2.168e-19 |
| 16³ | Boosted punctures | 4 | 2926.1 | 2777.7 | 1.053 | 2.776e-17 |
| 16³ | Rotated wave | 6 | 4432.4 | 4263.3 | 1.040 | 1.626e-19 |
| 16³ | Boosted punctures | 6 | 4513.6 | 4439.1 | 1.017 | 5.551e-17 |
| 32³ | Rotated wave | 2 | 1983.6 | 1669.9 | 1.188 | 2.168e-19 |
| 32³ | Boosted punctures | 2 | 1922.3 | 1708.5 | 1.125 | 4.441e-16 |
| 32³ | Rotated wave | 4 | 2949.6 | 2664.5 | 1.107 | 2.168e-19 |
| 32³ | Boosted punctures | 4 | 3001.6 | 2711.2 | 1.107 | 1.332e-15 |
| 32³ | Rotated wave | 6 | 4561.2 | 4387.2 | 1.040 | 1.084e-19 |
| 32³ | Boosted punctures | 6 | 4531.4 | 4361.9 | 1.039 | 4.441e-16 |

All 17 double-precision diagnostic fields on the output slice match the baseline
within roundoff (comparison tolerance `rtol=2e-11, atol=2e-13`). The timed kernel
computes the entire 3D grid; the high-precision table contains a 1D slice.
The largest absolute difference across both grids is 1.332e-15.

These measurements are specific to this CPU/backend/compiler and are not a
universal speedup claim. No CUDA compiler or GPU is present in this environment;
new GPU register usage, spills, occupancy and timings remain unmeasured. Use a
CUDA device profiler and full-kernel timing on the deployment hardware before
claiming GPU performance. The included Kokkos Tools timer is CPU/Serial-only.

## Reproduce

Build both revisions with the same flags and Kokkos submodule, saving the original
executable before updating the diagnostic header. From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build -j 4
c++ -O2 -std=c++17 -shared -fPIC tst/benchmarks/z4c_diagnostics_timer.cpp -o timer.so
python tst/benchmarks/z4c_diagnostics.py --baseline /absolute/path/baseline/athena \
  --candidate /absolute/path/candidate/athena --profiler ./timer.so --size 16
# Repeat with --size 32 (optionally taskset -c 0 on Linux).
```

## Targeted validation

47 CPU tests pass (7.02 s):

```sh
ATHENA_OVERHAUL_EXE=/absolute/path/candidate/athena python -m pytest -q \
  tst/test_suite/z4c/test_z4c_overhaul_cpu.py \
  tst/test_suite/z4c/test_z4c_restart_cpu.py \
  tst/test_suite/z4c/test_z4c_conversion_cpu.py \
  tst/test_suite/z4c/test_z4c_fastflow_cpu.py
```

The expanded independent NumPy super-Poynting/metric-norm tests also pass on the
baseline executable (six cases). The rotated-wave tests advance one step before
reading ADM output because that generator initially populates Z4c storage.
`git diff --check` passes; C++ lint has only the diagnostic header's two existing
`build/include_subdir` findings, with no new findings. The timer passes lint.
No MPI or GPU regressions were run for this revision.
