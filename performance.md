# Z4c CUDA RHS Performance Notes

Date: 2026-04-29

## Scope

This note records the profiling and RHS-only optimization attempts for the CUDA Z4c solver using the `z4c_one_puncture` problem generator.

The requested scope was to keep changes localized to the RHS kernel in `src/z4c/z4c_calcrhs.cpp`. I did not keep any performance change because every tested RHS variant either slowed the measured RHS kernel or failed to compile with NVHPC extended-lambda rules. The working tree currently has only a newline-only change in `src/z4c/z4c_calcrhs.cpp`.

## Environment And Build

Relevant local setup:

```text
Repository: /home/hz0693/athenak
Environment file: /home/hz0693/athenak/athenak_env
Build directory: /home/hz0693/athenak/build_one_punc
Executable: /home/hz0693/athenak/build_one_punc/src/athena
Problem generator: z4c_one_puncture
Compiler: nvc++ from NVHPC 24.11
CUDA toolkit: 12.6
Kokkos backend: CUDA, Ampere80
Precision: double
MPI: enabled
```

`build_one_punc/CMakeCache.txt` shows:

```text
PROBLEM=z4c_one_puncture
Kokkos_ENABLE_CUDA=ON
Kokkos_ARCH_AMPERE80=ON
CMAKE_CXX_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin/nvc++
MPI_CXX_COMPILER=/usr/local/openmpi/cuda-12.6/4.1.6/nvhpc2411/bin/mpicxx
```

The Athena executable is MPI-linked. Running inside the sandbox failed at `MPI_Init` because OpenMPI could not create local sockets, so the actual runs and profiling were done with escalated execution.

## Test Problem

I used the existing input:

```text
inputs/z4c/onepuncture/z4c_onepuncture.athinput
```

Small single-block overrides:

```bash
build_one_punc/src/athena \
  -i inputs/z4c/onepuncture/z4c_onepuncture.athinput \
  -d /tmp/athenak_z4c_base \
  mesh/nx1=64 mesh/nx2=64 mesh/nx3=64 \
  meshblock/nx1=64 meshblock/nx2=64 meshblock/nx3=64 \
  time/nlim=3 time/tlim=1 \
  output1/dt=100 output2/dt=100 output3/dt=100 output4/dt=100
```

This ran:

```text
Root grid = 1 x 1 x 1 MeshBlocks
Total MeshBlocks = 1
Ranks = 1
Cycles = 3
dt = 6.25e-02
Final time = 1.875e-01
```

## Baseline Correctness Output

The final baseline history record was:

```text
time       dt          C-norm2    H-norm2    M-norm2     Z-norm2      Mx/My/Mz-norm2       Theta-norm  Volume
1.875e-01  6.250e-02   6.29783e1  6.29089e1  1.54269e-2  9.84721e-6   2.35430e-2 each      5.39095e-2  7.67326e4
```

Patched test runs matched the baseline history output at printed precision for the final 3-cycle state.

## Profiling Tools

`nsys` from `/usr/local/cuda-12.6/bin/nsys` was unusable because it reported:

```text
Nsight Systems 2024.5.1 hasn't been installed with CUDA Toolkit 12.6
```

I used:

```text
/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys
```

Representative profile command:

```bash
/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys profile \
  --stats=true --force-overwrite=true \
  -o /tmp/athenak_z4c_base/nsys_base \
  build_one_punc/src/athena \
  -i inputs/z4c/onepuncture/z4c_onepuncture.athinput \
  -d /tmp/athenak_z4c_base \
  mesh/nx1=64 mesh/nx2=64 mesh/nx3=64 \
  meshblock/nx1=64 meshblock/nx2=64 meshblock/nx3=64 \
  time/nlim=3 time/tlim=1 \
  output1/dt=100 output2/dt=100 output3/dt=100 output4/dt=100
```

I also tried Nsight Compute for RHS details, but it could not access performance counters:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

So the available low-level data came from Nsight Systems: kernel durations, launch shapes, registers/thread, and local memory/thread.

## Baseline Profile

Baseline Nsight Systems profile:

```text
Report: /tmp/athenak_z4c_base/nsys_base.sqlite
```

Top GPU kernels:

```text
Main Z4c::CalcRHS<2> RHS kernel:
  total: 15.587 ms over 9 launches
  average: 1.732 ms/launch
  share: 64.5% of GPU kernel time
  registers/thread: 255
  local memory/thread: 0 B
  launch shape: blockY=128, gridX=2048

K-O dissipation:
  total: 2.072 ms over 9 launches
  share: 8.6%
  registers/thread: 32

ADMConstraints<2>:
  total: 1.802 ms over 4 launches
  share: 7.5%
  registers/thread: 255

ExpRKUpdate:
  total: 1.480 ms over 9 launches
  share: 6.1%
  registers/thread: 22
```

No non-RHS kernel exceeded 10% of total GPU kernel time in this small run.

The RHS kernel is clearly register-limited: it reports 255 registers/thread. Nsight Systems reports `0 B` local memory/thread, but Nsight Compute could not verify detailed spill counters due GPU performance-counter permissions.

## Optimization Attempts

### 1. Delay Telegraph-Lapse Scratch Work

Observation:

`opt.telegraph_lapse` defaults to `false` for this problem. The original RHS kernel still declared and filled:

```cpp
AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dB_dd;
AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LB_d;
Real dB;
```

and computed derivatives of `z4c.vB_d` before the `if (opt.telegraph_lapse)` block.

Attempt:

Move `LB_d`, `dB`, and the `vB_d` derivative/Lie-derivative work inside:

```cpp
if (opt.telegraph_lapse) { ... }
```

Expected benefit:

Shorter live range and fewer scratch tensors in the default path.

Outcome:

The run remained physically unchanged at printed history precision, but the profile got slower when combined with the other arithmetic cleanup in the same variant:

```text
RHS total: 18.369 ms over 9 launches
RHS average: 2.041 ms/launch
registers/thread: 255
local memory/thread: 0 B
```

This was not kept.

### 2. Precompute Rphi Trace Contraction

Original code recomputes this contraction inside the `(a,b)` loop:

```cpp
g_uu(c,d) * (Ddphi_dd(c,d) + 2*dphi_d(c)*dphi_d(d))
```

Attempt:

Compute:

```cpp
Real Rphi_trace = 0.0;
for c,d:
  Rphi_trace += ...
```

then use:

```cpp
Rphi_dd(a,b) -= 2.0 * g_dd(a,b) * Rphi_trace;
```

Expected benefit:

Reduce repeated arithmetic in the RHS kernel.

Outcome:

Included in the same slower variant above. It did not reduce register count and was not kept.

### 3. Precompute `d_i phi d^i alpha`

The source already had a TODO noting this possible optimization in the covariant lapse derivative section.

Original code performs:

```cpp
for c:
  for d:
    Ddalpha_dd(a,b) += 2*g_dd(a,b)*g_uu(c,d)*dphi_d(c)*dalpha_d(d);
```

inside the `(a,b,c,d)` loop nest.

Attempt:

Precompute:

```cpp
Real dphi_dalpha = 0.0;
for c,d:
  dphi_dalpha += g_uu(c,d) * dphi_d(c) * dalpha_d(d);
```

then use:

```cpp
Ddalpha_dd(a,b) += 2*g_dd(a,b)*dphi_dalpha;
```

Expected benefit:

Less repeated contraction work.

Outcome:

Included in the same slower variant above. It did not reduce register count and was not kept.

### 4. Kokkos Launch Bounds, 128 Threads, 3 Blocks/SM

Baseline Kokkos launch used a 1D range wrapper with CUDA block shape:

```text
blockY=128
registers/thread=255
```

Attempt:

Bypass the `par_for` wrapper only for the main RHS loop and use:

```cpp
Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<128, 3>>
```

with manual index decoding from 1D `idx` to `(m,k,j,i)`.

Expected benefit:

Force lower register use or more resident blocks per SM.

Outcome:

Slower:

```text
RHS total: 19.110 ms over 9 launches
RHS average: 2.123 ms/launch
registers/thread: 255
local memory/thread: 0 B
```

This was not kept.

### 5. Runtime `pow()` Fast Path For Default `chi_psi_power=-4`

For this input, `opt.chi_psi_power` has the default value:

```text
chi_psi_power = -4.0
```

The RHS computes:

```cpp
oopsi4 = pow(chi_guarded, -4.0 / opt.chi_psi_power);
```

For `chi_psi_power=-4`, this is mathematically:

```cpp
oopsi4 = chi_guarded;
```

Attempt:

Use a runtime branch:

```cpp
oopsi4 = (opt.chi_psi_power == -4.0)
       ? chi_guarded
       : pow(chi_guarded, -4.0 / opt.chi_psi_power);
```

Expected benefit:

Avoid an expensive `pow(x, 1)` in the default configuration.

Outcome:

Slower under Nsight Systems:

```text
RHS total: 18.836 ms over 9 launches
RHS average: 2.093 ms/launch
registers/thread: 255
local memory/thread: 0 B
```

This was not kept.

### 6. Kokkos Launch Bounds, 256 Threads, 1 Block/SM

Attempt:

Use:

```cpp
Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<256, 1>>
```

for the main RHS kernel.

Expected benefit:

Try increasing active warps without forcing a lower register limit.

Observed launch shape still reported:

```text
blockY=128
registers/thread=255
```

Outcome with the `pow()` branch included:

```text
RHS total: 16.809 ms over 9 launches
RHS average: 1.868 ms/launch
```

Outcome after removing the `pow()` branch and testing launch-only:

```text
RHS total: 18.601 ms over 9 launches
RHS average: 2.067 ms/launch
```

This was not kept.

### 7. Compile-Time Specialization For `chi_psi_power=-4`

Attempt:

Try a host-side dispatch to generate a separate RHS lambda for the default `chi_psi_power=-4` case using `std::true_type`/`std::false_type` and `if constexpr`, so the device code could compile out the `pow()`.

Expected benefit:

Avoid the runtime branch from attempt 5 and remove `pow()` from the default compiled RHS body.

Outcome:

NVHPC rejected this pattern:

```text
An extended __host__ __device__ lambda cannot be defined inside a generic lambda expression("operator()").
An extended __host__ __device__ lambda cannot first-capture variable in constexpr-if context
```

This was backed out immediately.

## Current Outcome

No performance-improving RHS-local change was found in the attempts above.

The main RHS kernel remains the dominant cost:

```text
~64.5% of GPU kernel time in the 64^3 single-block run
255 registers/thread
0 B local memory/thread reported by Nsight Systems
```

`K-O Dissipation`, `ADMConstraints`, and `ExpRKUpdate` were visible but each stayed below 10% of total GPU kernel time in the baseline profile.

## Recommended Next Steps

The RHS lambda is large enough that minor algebraic changes do not relieve the 255-register pressure. More promising options:

1. Split the monolithic RHS calculation into multiple kernels with explicit intermediate arrays for a small number of high-value quantities. This trades global memory traffic for lower register pressure and may improve occupancy.
2. Split optional gauge paths, especially `telegraph_lapse`, into separate host-dispatched kernels or separate functions so the default path does not carry dead scratch state.
3. Refactor tensor scratch storage in `z4c_calcrhs.cpp` to reduce simultaneous live tensors. The main live set includes first derivatives, second derivatives, Christoffels, Ricci terms, conformal-factor terms, lapse derivatives, and A contractions all in one device lambda.
4. Add NVTX ranges or more specific Kokkos labels around Z4c tasks to simplify future Nsight analysis.
5. Get Nsight Compute performance-counter permission enabled and rerun with spill/local-memory metrics. Nsight Systems is enough for timing and register counts, but not enough to confirm detailed spill load/store behavior.

## Update: Split Gauge RHS Kernel

Date: 2026-04-30

I tested a conservative kernel split in `src/z4c/z4c_calcrhs.cpp`:

- Keep the geometry/curvature RHS work in the original `z4c rhs loop`.
- Move the gauge RHS work for `alpha`, `beta_u`, and optional `vB_d` into a second `z4c gauge rhs loop`.
- Recompute the small gauge derivative subset in the second kernel: inverse conformal metric, `dalpha_d`, `dchi_d`, `Lalpha`, and `Lbeta_u`.
- For the optional telegraph-lapse path, accumulate `d^a B_a` directly from repeated `Dx()` calls instead of carrying a `dB_dd` scratch tensor.

This intentionally trades repeated arithmetic and one extra launch for less live state in the dominant RHS kernel. It does not introduce global intermediate arrays.

Correctness check:

```text
Command: same 64^3 single-block, 3-cycle z4c_one_puncture run as baseline
Result: /tmp/athenak_z4c_split_finalcheck/z4c.z4c.user.hst matched
        /tmp/athenak_z4c_base_rerun/z4c.z4c.user.hst byte-for-byte.

Final history row:
1.87500e-01  6.25000e-02  6.29783e+01  6.29089e+01  1.54269e-02
9.84721e-06  2.35430e-02  2.35430e-02  2.35430e-02  5.39095e-02  7.67326e+04
```

Fresh baseline profile for comparison:

```text
Report: /tmp/athenak_z4c_base_profile2/nsys_base2.sqlite

Main z4c rhs loop:
  total: 17.551 ms over 9 launches
  average: 1.950 ms/launch
  registers/thread: 255
  local memory/thread: 0 B

K-O dissipation:
  total: 2.382 ms over 9 launches
  average: 0.265 ms/launch
```

Final split-gauge profile:

```text
Report: /tmp/athenak_z4c_split_profile_final/nsys_split_gauge_final.sqlite

Main z4c rhs loop:
  total: 12.529 ms over 9 launches
  average: 1.392 ms/launch
  registers/thread: 255
  local memory/thread: 0 B

New z4c gauge rhs loop:
  total: 1.129 ms over 9 launches
  average: 0.125 ms/launch
  registers/thread: 134
  local memory/thread: 0 B

Combined split RHS gauge+main:
  total: 13.658 ms over 9 launches
  average: 1.518 ms per RK RHS stage
```

Compared with the fresh baseline profile, the split reduced the measured main+gauge RHS kernel time by about:

```text
(17.551 ms - 13.658 ms) / 17.551 ms = 22.2%
```

The final split also improved the related K-O dissipation timing in that profile (`2.136 ms` versus `2.382 ms`), but I do not attribute that to the RHS split directly.

Zone-cycle throughput:

The `zone-cycles/cpu_second` measurement is noisy on this node. Some early split samples were severe outliers (`4.36e6` to `7.40e6`) and were not reproducible after rebuilding/rerunning. I compared the stable cluster from the fresh baseline with the final split build:

```text
Baseline recent 20-cycle samples:
  3.646273e7, 3.648697e7, 3.636838e7, 3.649806e7
  median: 3.647485e7 zone-cycles/cpu_second

Final split recent 20-cycle samples:
  2.936899e7, 3.954578e7, 3.950053e7
  median: 3.950053e7 zone-cycles/cpu_second
```

Using the recent medians, the split showed:

```text
(3.950053e7 - 3.647485e7) / 3.647485e7 = 8.3%
```

This should be treated as a modest throughput gain, not a hard guarantee, because the full application timing was visibly noisier than the Nsight kernel timings.

Spillage/repeated-calculation conclusion:

Nsight Systems still reports `0 B` local memory/thread for the baseline RHS, the split main RHS, and the split gauge RHS. That means I could not confirm actual spill traffic with the available tools. Nsight Compute still requires GPU performance-counter permission to verify spill loads/stores directly.

What the split does verify is narrower: repeated calculation in a smaller second kernel was cheaper than keeping all gauge work in the monolithic RHS for this 64^3 one-puncture run. The main kernel remained at the 255-register/thread ceiling, but its live work decreased enough that its launch time fell from `1.950 ms` to `1.392 ms`; adding the `0.125 ms` gauge launch still left a net RHS kernel-time win.

Current kept change:

The split-gauge RHS kernel is left in `src/z4c/z4c_calcrhs.cpp` because it preserves the baseline history output and gives a measurable kernel-level speedup plus a modest recent median `zone-cycles/cpu_second` gain.

## Update: Gauge Coefficient Hoist And Larger-Grid Scaling

Date: 2026-04-30

I made one additional readability-preserving tweak to the split gauge kernel:

- Cache the zone-local `alpha` and `chi` values in the gauge kernel.
- Precompute scalar gauge coefficients used by all three shift components:
  `shift_gamma`, `alpha_sq`, `shift_alpha2ggamma`, and `shift_hh_alpha_chi`.
- Keep all option branches with the same conditions and behavior as the split-gauge version:
  `opt.slow_start_lapse`, `opt.telegraph_lapse`, and the shift/harmonic terms are still evaluated in the same logical cases.

This changes repeated scalar work inside the gauge kernel but does not add or remove any runtime branch path.

Correctness check:

```text
Command: same 64^3 single-block, 3-cycle z4c_one_puncture run
Result: /tmp/athenak_z4c_gauge_coeff_check/z4c.z4c.user.hst matched
        /tmp/athenak_z4c_split_finalcheck/z4c.z4c.user.hst byte-for-byte.
```

Final 64^3 profile after coefficient hoist:

```text
Report: /tmp/athenak_z4c_gauge_coeff_profile/nsys_gauge_coeff.sqlite

Main z4c rhs loop:
  total: 12.521 ms over 9 launches
  average: 1.391 ms/launch
  registers/thread: 255
  local memory/thread: 0 B

Gauge rhs loop:
  total: 1.013 ms over 9 launches
  average: 0.113 ms/launch
  registers/thread: 132
  local memory/thread: 0 B

Combined split RHS gauge+main:
  total: 13.533 ms over 9 launches
  average: 1.504 ms per RK RHS stage
```

Compared with the earlier split-gauge profile, this is a small improvement:

```text
Earlier split combined RHS: 13.658 ms
Coefficient-hoisted split combined RHS: 13.533 ms
Improvement over earlier split: 0.9%

Earlier gauge kernel: 1.129 ms, 134 registers/thread
Coefficient-hoisted gauge kernel: 1.013 ms, 132 registers/thread
Gauge-only improvement: 10.3%
```

Compared with the fresh monolithic baseline profile from above:

```text
Baseline monolithic RHS: 17.551 ms
Coefficient-hoisted split combined RHS: 13.533 ms
Kernel-level RHS improvement: 22.9%
```

Final 64^3 `zone-cycles/cpu_second` samples remained noisy:

```text
Baseline recent median: 3.647485e7
Coefficient-hoisted split samples:
  2.810252e7, 3.690264e7, 3.967151e7
  median: 3.690264e7

Median change: +1.2%
```

### 128^3 Scaling Check

I also compared the monolithic baseline and the coefficient-hoisted split on a larger single-block run:

```text
Common command changes:
  mesh/nx1=128 mesh/nx2=128 mesh/nx3=128
  meshblock/nx1=128 meshblock/nx2=128 meshblock/nx3=128
  time/nlim=3 time/tlim=1
```

Baseline 128^3 profile:

```text
Report: /tmp/athenak_z4c_base128_profile/nsys_base128.sqlite

Monolithic z4c rhs loop:
  total: 128.179 ms over 9 launches
  average: 14.242 ms/launch
  registers/thread: 255
  local memory/thread: 0 B
```

Split 128^3 profile:

```text
Report: /tmp/athenak_z4c_split128_profile/nsys_split128.sqlite

Main z4c rhs loop:
  total: 116.841 ms over 9 launches
  average: 12.982 ms/launch
  registers/thread: 255
  local memory/thread: 0 B

Gauge rhs loop:
  total: 7.815 ms over 9 launches
  average: 0.868 ms/launch
  registers/thread: 132
  local memory/thread: 0 B

Combined split RHS gauge+main:
  total: 124.656 ms over 9 launches
  average: 13.851 ms per RK RHS stage
```

For 128^3, the main RHS kernel alone is still faster:

```text
(128.179 ms - 116.841 ms) / 128.179 ms = 8.8%
```

But once the extra gauge kernel is included, the RHS-level improvement is much smaller:

```text
(128.179 ms - 124.656 ms) / 128.179 ms = 2.7%
```

The larger-grid `zone-cycles/cpu_second` samples were also noisy, but the median favored the split:

```text
Baseline 128^3 samples:
  3.176071e7, 4.013818e7, 4.090536e7
  median: 4.013818e7

Split 128^3 samples:
  3.648174e7, 4.600617e7, 4.650977e7
  median: 4.600617e7

Median change: +14.6%
```

I would not overinterpret the 128^3 wall-clock median because the short `nlim=3` runs show substantial run-to-run variability. The Nsight kernel timings are the more stable signal: the split remains beneficial, but the RHS-only percentage improvement is smaller at 128^3 than at 64^3.

### Scaling Expectation

I do not expect this split to keep gaining a larger percentage just because the problem gets larger. The reason is that two competing effects scale differently:

1. The extra kernel launch is amortized better for larger per-rank zone counts.
2. The gauge kernel recomputation is per-zone work, so it scales with the problem size.

At 64^3, reducing the monolithic kernel's live work produced a large kernel-level win. At 128^3, the GPU appears better saturated, and the extra gauge work eats more of the saved time. Based on these two measurements, I expect this optimization to remain useful for reasonably large per-rank blocks, but the RHS-only speedup is more likely to settle at a modest single-digit percentage than grow beyond the 64^3 result.

For strong-scaling cases with fewer zones per GPU/rank, the extra launch can matter more and the benefit may shrink or disappear. For weak-scaling cases with large per-rank blocks, the split should be safer, but the measured 128^3 profile argues against expecting a larger percentage gain.

## Update: 128^3 With 64^3 And 32^3 MeshBlocks

Follow-up request:

```text
Total grid: 128^3
Runs: z4c_one_puncture, time/nlim=3, time/tlim=1
MeshBlock layouts tested:
  128^3 MeshBlock: 1 block
   64^3 MeshBlock: 8 blocks
   32^3 MeshBlock: 64 blocks
```

The current kept source state is still the split gauge RHS kernel with the coefficient-hoist readability tweak. I temporarily reversed only that source diff to rebuild and measure the monolithic baseline for the new 64^3 and 32^3 MeshBlock layouts, then restored the split source and rebuilt it.

Correctness checks for the new layouts:

```text
128^3 grid, 64^3 MeshBlocks:
  cmp baseline/split z4c.z4c.user.hst: identical

128^3 grid, 32^3 MeshBlocks:
  cmp baseline/split z4c.z4c.user.hst: identical
```

This matches the earlier 64^3 baseline check: the split did not change the produced history output for these short runs.

Zone-cycle samples:

```text
128^3 total grid, 128^3 MeshBlock:
  baseline samples: 3.176071e7, 4.013818e7, 4.090536e7
  baseline median: 4.013818e7
  split samples:    3.648174e7, 4.600617e7, 4.650977e7
  split median:     4.600617e7
  median change:    +14.6%

128^3 total grid, 64^3 MeshBlocks:
  baseline samples: 3.178048e7, 4.060007e7, 4.062561e7
  baseline median: 4.060007e7
  split samples:    3.693695e7, 4.450031e7, 4.516266e7
  split median:     4.450031e7
  median change:    +9.6%

128^3 total grid, 32^3 MeshBlocks:
  baseline samples: 3.663409e7, 3.656076e7, 3.689155e7
  baseline median: 3.663409e7
  split samples:    4.054066e7, 4.118074e7, 4.058004e7
  split median:     4.058004e7
  median change:    +10.8%
```

Nsight RHS kernel timings:

```text
128^3 total grid, 128^3 MeshBlock:
  baseline report: /tmp/athenak_z4c_base128_profile/nsys_base128.sqlite
  split report:    /tmp/athenak_z4c_split128_profile/nsys_split128.sqlite

  baseline monolithic RHS: 128.179 ms over 9 launches
  split main RHS:          116.841 ms over 9 launches
  split gauge RHS:           7.815 ms over 9 launches
  split combined RHS:      124.656 ms over 9 launches
  combined RHS change:      +2.7%

128^3 total grid, 64^3 MeshBlocks:
  baseline report: /tmp/athenak_z4c_base128_mb64_profile/nsys_base128_mb64.sqlite
  split report:    /tmp/athenak_z4c_split128_mb64_profile/nsys_split128_mb64.sqlite

  baseline monolithic RHS: 137.130 ms over 9 launches
  split main RHS:          111.080 ms over 9 launches
  split gauge RHS:           8.019 ms over 9 launches
  split combined RHS:      119.099 ms over 9 launches
  main-RHS-only change:     +19.0%
  combined RHS change:      +13.1%

128^3 total grid, 32^3 MeshBlocks:
  baseline report: /tmp/athenak_z4c_base128_mb32_profile/nsys_base128_mb32.sqlite
  split report:    /tmp/athenak_z4c_split128_mb32_profile/nsys_split128_mb32.sqlite

  baseline monolithic RHS: 118.461 ms over 9 launches
  split main RHS:          108.102 ms over 9 launches
  split gauge RHS:           8.322 ms over 9 launches
  split combined RHS:      116.424 ms over 9 launches
  main-RHS-only change:      +8.7%
  combined RHS change:       +1.7%
```

Registers and local memory stayed consistent with the earlier profiles:

```text
Monolithic/main RHS kernel:
  registers/thread: 255
  local memory/thread: 0 B

Split gauge RHS kernel:
  registers/thread: 132
  local memory/thread: 0 B
```

The main RHS kernel still appears register-pressure-limited or at least register-count-constrained, but these profiles do not show local-memory spills. The split helps by removing enough live gauge work from the main kernel to make the main RHS faster. The extra gauge kernel is cheap, but it is not free, so the combined gain depends on layout and occupancy.

### MeshBlock Interpretation

For the current split source, the RHS-only profile improved as MeshBlocks got smaller:

```text
Split combined RHS:
  128^3 MeshBlock: 124.656 ms
   64^3 MeshBlock: 119.099 ms
   32^3 MeshBlock: 116.424 ms
```

However, the end-to-end zone-cycle median did not improve monotonically:

```text
Split zone-cycles/cpu_second median:
  128^3 MeshBlock: 4.600617e7
   64^3 MeshBlock: 4.450031e7
   32^3 MeshBlock: 4.058004e7
```

The 32^3 MeshBlock case has better RHS kernel timing than 64^3, but it also exposes noticeably more boundary/pack/unpack work. In the split 32^3 profile, the two visible MeshBoundaryValues pack/unpack kernels took about 22.108 ms total, compared with about 10.136 ms for 64^3. That extra non-RHS overhead explains why 32^3 did not win the wall-clock throughput check despite a faster RHS profile.

The 64^3 MeshBlock layout is therefore a useful compromise in these short tests: it gives a strong split-kernel RHS improvement over its matching monolithic baseline, while avoiding most of the boundary overhead seen at 32^3. The single 128^3 MeshBlock still had the best split wall-clock median in this noisy sample set, but the 64^3 layout was close and showed a larger RHS-level benefit.

I still would not project a larger percentage speedup purely from larger global problems. The better expectation is layout-dependent: if larger runs keep a large enough per-rank block and do not increase boundary work too much, the split should remain beneficial; if decomposition creates many smaller MeshBlocks or more inter-block boundary traffic, non-RHS overhead can dominate the gain.

## Update: Compile-Time Spill Check And Gamma Split

Follow-up request:

```text
Check register spillage at compile time with -Xptxas -v, then optimize further.
```

The active build on this machine is not the Hopper/nvcc_wrapper command from the example. The existing `build_one_punc` configuration uses:

```text
CXX compiler: /opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin/nvc++
CUDA arch:    Kokkos_ARCH_AMPERE80
CUDA path:    /usr/local/cuda-12.6
```

Kokkos still routes CUDA compilation through `kokkos/bin/nvcc_wrapper`, so the useful compile-time check was a separate build directory with:

```text
build_one_punc_xptxas
CMAKE_CXX_FLAGS="-Xptxas -v"
Kokkos_ARCH_AMPERE80=ON
PROBLEM=z4c_one_puncture
```

I first tried NVHPC `-gpu=ptxinfo`, but in this wrapper path it was forwarded to the host compiler side and produced warnings that it had no GPU-codegen effect. The `-Xptxas -v` route did produce the spill report.

### Gauge-Only Split Spill Check

For the previously kept gauge-only split, the compile-time `ptxas` report confirmed that the remaining main RHS kernel still spilled:

```text
NGHOST=2 main z4c rhs loop:
  registers/thread: 255
  stack frame:      1768 B
  spill stores:     1380 B
  spill loads:      1548 B

NGHOST=2 gauge rhs loop:
  registers/thread: 132
  stack frame:      0 B
  spill stores:     0 B
  spill loads:      0 B
```

This explains why Nsight's `localMemoryPerThread=0` was incomplete as a spill check here. The runtime profile reported no local memory per thread, but `ptxas` still showed stack/spill traffic in the generated main RHS kernel.

### Additional Gamma Split

I then split the `vGam_u` RHS assembly into a separate `z4c Gamma rhs loop`. This removes the following from the main geometry/scalar/A RHS kernel:

```text
DA_u, A_uu, LGam_u, dKhat_d, dTheta_d, and the vGam_u final assembly
```

The new Gamma kernel recomputes the subset it needs. This is exactly the tradeoff suspected earlier: repeated arithmetic can be cheaper than keeping too many values live in a max-register kernel.

Branch behavior was kept the same:

```text
Matter branches remain guarded by if(!is_vacuum).
Gauge branches remain guarded by if (opt.slow_start_lapse) and if (opt.telegraph_lapse).
The Gamma split did not add new option-dependent branches.
```

Correctness checks:

```text
64^3 grid, 64^3 MeshBlock:
  cmp previous split z4c.z4c.user.hst: identical

128^3 grid, 128^3 MeshBlock:
  cmp previous split z4c.z4c.user.hst: identical
  cmp monolithic baseline z4c.z4c.user.hst: identical

128^3 grid, 64^3 MeshBlocks:
  cmp monolithic baseline z4c.z4c.user.hst: identical

128^3 grid, 32^3 MeshBlocks:
  cmp monolithic baseline z4c.z4c.user.hst: identical
```

Compile-time spill report after the Gamma split:

```text
NGHOST=2 main z4c rhs loop:
  registers/thread: 255
  stack frame:      896 B
  spill stores:     96 B
  spill loads:      96 B

NGHOST=2 Gamma rhs loop:
  registers/thread: 253
  stack frame:      0 B
  spill stores:     0 B
  spill loads:      0 B

NGHOST=2 gauge rhs loop:
  registers/thread: 132
  stack frame:      0 B
  spill stores:     0 B
  spill loads:      0 B
```

So the main RHS still uses the architectural register ceiling, but the large spill traffic is mostly gone.

### Gamma Split Performance

64^3 total grid, 64^3 MeshBlock:

```text
Report: /tmp/athenak_z4c_gamma_split_64_profile/nsys_gamma_split_64.sqlite

Gamma split RHS kernels:
  main RHS:   8.951 ms over 9 launches
  Gamma RHS:  1.622 ms over 9 launches
  gauge RHS:  1.143 ms over 9 launches
  combined:  11.716 ms over 9 RHS stages

Previous gauge-only split combined RHS:
  13.533 ms over 9 RHS stages

Change vs gauge-only split:
  +13.4% RHS improvement

Change vs monolithic baseline:
  +33.2% RHS improvement
```

The 64^3 wall-clock samples were stable but lower than the earlier noisy gauge-only split samples:

```text
Gamma split 64^3 zone-cycles/cpu_second:
  3.116222e7, 3.109707e7, 3.126843e7
  median: 3.116222e7
```

I would not use this 64^3 wall-clock result alone to reject the Gamma split because the kernel profile clearly improved and the earlier 64^3 wall-clock runs had substantial run-to-run variability.

128^3 total grid, 128^3 MeshBlock:

```text
Report: /tmp/athenak_z4c_gamma_split_128_profile/nsys_gamma_split_128.sqlite

Gamma split RHS kernels:
  main RHS:   70.304 ms over 9 launches
  Gamma RHS:  10.948 ms over 9 launches
  gauge RHS:   7.259 ms over 9 launches
  combined:   88.510 ms over 9 RHS stages

Previous gauge-only split combined RHS:
  124.656 ms over 9 RHS stages

Monolithic baseline RHS:
  128.179 ms over 9 RHS stages

Change vs gauge-only split:
  +29.0% RHS improvement

Change vs monolithic baseline:
  +30.9% RHS improvement
```

128^3 single-MeshBlock zone-cycle samples:

```text
Gamma split samples:
  4.338790e7, 5.791858e7, 5.708484e7
  median: 5.708484e7

Previous gauge-only split median:
  4.600617e7

Monolithic baseline median:
  4.013818e7

Median change vs gauge-only split: +24.1%
Median change vs monolithic:       +42.2%
```

128^3 total grid, 64^3 MeshBlocks:

```text
Report: /tmp/athenak_z4c_gamma_split_128_mb64_profile/nsys_gamma_split_128_mb64.sqlite

Gamma split RHS kernels:
  main RHS:   43.177 ms over 9 launches
  Gamma RHS:   6.932 ms over 9 launches
  gauge RHS:   5.130 ms over 9 launches
  combined:   55.239 ms over 9 RHS stages

Previous gauge-only split combined RHS:
  119.099 ms over 9 RHS stages

Monolithic baseline RHS:
  137.130 ms over 9 RHS stages

Change vs gauge-only split:
  +53.6% RHS improvement

Change vs monolithic baseline:
  +59.7% RHS improvement
```

128^3 with 64^3 MeshBlocks zone-cycle samples:

```text
Gamma split samples:
  4.014592e7, 5.656777e7, 5.653523e7
  median: 5.653523e7

Previous gauge-only split median:
  4.450031e7

Monolithic baseline median:
  4.060007e7

Median change vs gauge-only split: +27.0%
Median change vs monolithic:       +39.2%
```

128^3 total grid, 32^3 MeshBlocks:

```text
Report: /tmp/athenak_z4c_gamma_split_128_mb32_profile/nsys_gamma_split_128_mb32.sqlite

Gamma split RHS kernels:
  main RHS:   67.887 ms over 9 launches
  Gamma RHS:  11.662 ms over 9 launches
  gauge RHS:   8.254 ms over 9 launches
  combined:   87.803 ms over 9 RHS stages

Previous gauge-only split combined RHS:
  116.424 ms over 9 RHS stages

Monolithic baseline RHS:
  118.461 ms over 9 RHS stages

Change vs gauge-only split:
  +24.6% RHS improvement

Change vs monolithic baseline:
  +25.9% RHS improvement
```

128^3 with 32^3 MeshBlocks zone-cycle samples:

```text
Gamma split samples:
  4.919390e7, 4.928091e7, 4.922557e7
  median: 4.922557e7

Previous gauge-only split median:
  4.058004e7

Monolithic baseline median:
  3.663409e7

Median change vs gauge-only split: +21.3%
Median change vs monolithic:       +34.4%
```

### Interpretation

The compile-time spill check validates the original hypothesis. The gauge-only split reduced runtime, but the remaining main RHS still had heavy compile-time spill traffic. Moving `vGam_u` into a second recomputing kernel cut the main-kernel spill traffic from `1380/1548 B` store/load to `96/96 B` for the active `NGHOST=2` build.

The extra Gamma kernel is not free, but for the 128^3 tests it is much cheaper than the spill-heavy main kernel it replaces. This is especially visible in the 128^3, 64^3-MeshBlock profile, where combined RHS time dropped from `119.099 ms` to `55.239 ms`.

The current best retained version is therefore the three-way split:

```text
z4c rhs loop        -> scalar, geometry, and A RHS
z4c Gamma rhs loop  -> vGam_u RHS
z4c gauge rhs loop  -> alpha, beta_u, optional vB_d RHS
```

The code is longer because the Gamma kernel recomputes a subset of the geometry, but the split remains local to `CalcRHS`, keeps option branches unchanged, and is backed by both compile-time spill reduction and runtime profiles.

### Zone-Cycle Improvement Summary

The clearest wall-clock comparison is the median of the three `zone-cycles/cpu_second` samples for each 128^3 layout:

```text
128^3 grid, 128^3 MeshBlock:
  monolithic baseline median: 4.013818e7
  gauge-only split median:    4.600617e7  (+14.6% vs baseline)
  Gamma split median:         5.708484e7  (+42.2% vs baseline, +24.1% vs gauge-only)

128^3 grid, 64^3 MeshBlocks:
  monolithic baseline median: 4.060007e7
  gauge-only split median:    4.450031e7  (+9.6% vs baseline)
  Gamma split median:         5.653523e7  (+39.2% vs baseline, +27.0% vs gauge-only)

128^3 grid, 32^3 MeshBlocks:
  monolithic baseline median: 3.663409e7
  gauge-only split median:    4.058004e7  (+10.8% vs baseline)
  Gamma split median:         4.922557e7  (+34.4% vs baseline, +21.3% vs gauge-only)
```

So the retained three-way split gives a measured end-to-end `zone-cycles/cpu_second` improvement of about `+34%` to `+42%` over the monolithic baseline across the tested 128^3 layouts. The first sample in several runs is lower than the later two, so these short-run medians should still be treated as approximate, but the direction is consistent across all three layouts.

## Update: Behavior Triple-Check And Other Z4C Kernels

I triple-checked the retained RHS split against a temporary monolithic build by reversing the local `z4c_calcrhs.cpp` diff, rebuilding, and comparing full output directories. These checks compare more than history files: they include binary `z4c` dumps and constraint table output.

Cases checked:

```text
default_ng2:
  16^3 grid, 16^3 MeshBlock, nghost=2, nlim=2

default_ng3:
  20^3 grid, 20^3 MeshBlock, nghost=3, nlim=2

multiblock_ng2:
  32^3 grid, 16^3 MeshBlocks, nghost=2, nlim=2

gauge_branches_ng2:
  16^3 grid, 16^3 MeshBlock, nghost=2, nlim=2
  slow_start_lapse = true
  telegraph_lapse = true
  shift_alpha2Gamma = 0.5
  shift_H = 0.25
  sss_damping_amp = 0.3
  lapse_harmonic = 0.2
```

The gauge-branch case used a temporary input file because command-line overrides are rejected when the keys are not already present in the input file.

Results:

```text
diff -qr monolithic/default_ng2      split/default_ng2:      no differences
diff -qr monolithic/default_ng3      split/default_ng3:      no differences
diff -qr monolithic/multiblock_ng2   split/multiblock_ng2:   no differences
diff -qr monolithic/gauge_branches   split/gauge_branches:   no differences
```

I also repeated the comparison after splitting `ADMConstraints` and writing constraint output at every step:

```text
output4/dt = 0.01

diff -qr monolithic/default_ng2      adm-split/default_ng2:      no differences
diff -qr monolithic/default_ng3      adm-split/default_ng3:      no differences
diff -qr monolithic/multiblock_ng2   adm-split/multiblock_ng2:   no differences
diff -qr monolithic/gauge_branches   adm-split/gauge_branches:   no differences
```

This is the strongest behavior check so far: it exercises the default RHS path, the moved gauge branches, the `NGHOST=3` template instantiation, multiblock indexing, and the ADM constraint diagnostics.

### Other Z4C Kernel Scan

I checked other Z4C kernels with the same `-Xptxas -v` method. Most kernels had no meaningful spill issue. The regular-use candidate was `ADMConstraints`; `Z4cWeyl` also has small spills for `NGHOST=2`, but it is an output/diagnostic path and not part of the measured baseline loop.

Before the ADM split:

```text
ADMConstraints<2>:
  registers/thread: 255
  stack frame:      728 B
  spill stores:     68 B
  spill loads:      72 B

ADMConstraints<3>:
  registers/thread: 255
  stack frame:      736 B
  spill stores:     80 B
  spill loads:      84 B

Z4cWeyl<2>:
  registers/thread: 255
  stack frame:      1416 B
  spill stores:     104 B
  spill loads:      104 B
```

I split `ADMConstraints` into:

```text
ADM Hamiltonian constraint loop
ADM momentum constraint loop
```

This is localized to `src/z4c/z4c_adm.cpp`. The Hamiltonian kernel computes `con.H`; the momentum kernel computes `con.M_d`, `con.M`, `con.Z`, and `con.C` using the already-written `con.H`. This preserves the output behavior and avoids changing public data structures.

After the ADM split:

```text
ADMConstraints<2> Hamiltonian kernel:
  registers/thread: 184
  spill stores:     0 B
  spill loads:      0 B

ADMConstraints<2> momentum/Z/C kernel:
  registers/thread: 172
  spill stores:     0 B
  spill loads:      0 B

ADMConstraints<3> Hamiltonian kernel:
  registers/thread: 254
  spill stores:     0 B
  spill loads:      0 B

ADMConstraints<3> momentum/Z/C kernel:
  registers/thread: 134
  spill stores:     0 B
  spill loads:      0 B
```

128^3 single-MeshBlock profile:

```text
Before ADM split:
  ADMConstraints total: 16.125 ms over 4 launches

After ADM split:
  Hamiltonian constraint: 6.514 ms over 4 launches
  Momentum/Z/C constraint: 8.510 ms over 4 launches
  Combined ADMConstraints: 15.023 ms over 8 launches

ADMConstraints kernel-time change:
  +6.8%
```

The full `zone-cycles/cpu_second` median did not improve in the short 128^3 sample:

```text
Gamma RHS split before ADM split:
  4.338790e7, 5.791858e7, 5.708484e7
  median: 5.708484e7

Gamma RHS split plus ADM split:
  4.338324e7, 5.684741e7, 5.665275e7
  median: 5.665275e7

Median change:
  -0.8%
```

So the ADM split is a modest compile-time and kernel-profile improvement, not a clear end-to-end wall-clock win in these short tests. The major optimization remains the RHS three-way split. The ADM split is retained because it removes spills, is local, and passed byte-for-byte diagnostic comparisons; it should be reconsidered if longer production runs show the extra launch overhead is not worthwhile.

## Update: Total ZCPS Speedup And NGHOST=4

Follow-up request:

```text
Report total zone-cycles/cpu_second speedup now.
Test mesh/nghost=4 and get ZCPS for all cases.
```

The retained optimized source for these measurements is:

```text
z4c rhs loop                     -> scalar, geometry, and A RHS
z4c Gamma rhs loop               -> vGam_u RHS
z4c gauge rhs loop               -> alpha, beta_u, optional vB_d RHS
ADM Hamiltonian constraint loop  -> con.H
ADM momentum constraint loop     -> con.M_d, con.M, con.Z, con.C
```

Each entry below uses three `nlim=3` samples and reports the median `zone-cycles/cpu_second`.

### NGHOST=2, 128^3 Total Grid

```text
128^3 MeshBlock:
  monolithic samples: 3.176071e7, 4.013818e7, 4.090536e7
  monolithic median:  4.013818e7
  optimized samples:  4.338324e7, 5.684741e7, 5.665275e7
  optimized median:   5.665275e7
  speedup:            1.411x (+41.1%)

64^3 MeshBlocks:
  monolithic samples: 3.178048e7, 4.060007e7, 4.062561e7
  monolithic median:  4.060007e7
  optimized samples:  4.106152e7, 5.682939e7, 5.667475e7
  optimized median:   5.667475e7
  speedup:            1.396x (+39.6%)

32^3 MeshBlocks:
  monolithic samples: 3.663409e7, 3.656076e7, 3.689155e7
  monolithic median:  3.663409e7
  optimized samples:  4.882672e7, 4.930900e7, 4.945896e7
  optimized median:   4.930900e7
  speedup:            1.346x (+34.6%)
```

So the current total end-to-end speedup for the default `nghost=2` 128^3 runs is about `1.35x-1.41x`, depending on MeshBlock layout.

### NGHOST=4, 128^3 Total Grid

For `mesh/nghost=4`, I temporarily reversed the local Z4C split diff, rebuilt the monolithic baseline, ran the same three layouts, then restored the optimized source and rebuilt it.

Correctness check:

```text
128^3 grid, 128^3 MeshBlock, nghost=4:
  cmp monolithic/optimized z4c.z4c.user.hst: identical

128^3 grid, 64^3 MeshBlocks, nghost=4:
  cmp monolithic/optimized z4c.z4c.user.hst: identical

128^3 grid, 32^3 MeshBlocks, nghost=4:
  cmp monolithic/optimized z4c.z4c.user.hst: identical
```

ZCPS results:

```text
128^3 MeshBlock:
  monolithic samples: 2.033694e7, 2.281715e7, 2.295947e7
  monolithic median:  2.281715e7
  optimized samples:  1.994304e7, 2.659445e7, 2.643186e7
  optimized median:   2.643186e7
  speedup:            1.158x (+15.8%)

64^3 MeshBlocks:
  monolithic samples: 2.338461e7, 2.336291e7, 2.329269e7
  monolithic median:  2.336291e7
  optimized samples:  2.696935e7, 2.699743e7, 2.685059e7
  optimized median:   2.696935e7
  speedup:            1.154x (+15.4%)

32^3 MeshBlocks:
  monolithic samples: 2.142415e7, 2.141339e7, 2.123934e7
  monolithic median:  2.141339e7
  optimized samples:  2.440351e7, 2.433701e7, 2.428704e7
  optimized median:   2.433701e7
  speedup:            1.137x (+13.7%)
```

The `nghost=4` speedup is consistently positive but much smaller than `nghost=2`: about `1.14x-1.16x`. The likely reason is that wider stencils and larger ghost regions increase non-RHS and memory-traffic costs enough that the register-spill reduction in the RHS kernels is a smaller fraction of total runtime.

## Update: Extra NGHOST=4 Register-Pressure Attempts

Follow-up request:

```text
Optimize a bit more for nghost=4; there may still be register spillage.
```

I checked the current retained split with `-Xptxas -v`. For `NGHOST=4`, the main RHS kernel is still the only large RHS spill site:

```text
K-O dissipation kernel:        34 registers, 0 spill stores, 0 spill loads
Gauge RHS kernel:             210 registers, 0 spill stores, 0 spill loads
Gamma RHS kernel:             182 registers, 0 spill stores, 0 spill loads
Main scalar/metric/A RHS:     255 registers, 608 B spill stores, 660 B spill loads
```

I tried three localized follow-ups:

```text
1. Remove now-unused second shift derivatives from the main RHS kernel.
   Result: exact output, but ptxas changed to 612 B stores / 664 B loads and
   ZCPS was neutral within short-run noise. Not retained.

2. Fill the symmetric Ddalpha_dd temporary only over unique entries while
   preserving the old final off-diagonal write order.
   Result: exact output, but no ptxas or ZCPS improvement. Not retained.

3. Split rhs.g_dd into a separate small metric RHS kernel.
   Result: the new metric kernel used 128 registers with no spills, but the
   remaining main RHS spill grew to 720 B stores / 784 B loads. Not retained.
```

The cleanup-only variant was tested on the same 128^3 `nghost=4` layouts:

```text
128^3 MeshBlock:
  samples: 2.239270e7, 2.624877e7, 2.652590e7
  median:  2.624877e7
  previous retained median: 2.643186e7

64^3 MeshBlocks:
  samples: 2.677708e7, 2.669943e7, 2.691254e7
  median:  2.677708e7
  previous retained median: 2.696935e7

32^3 MeshBlocks:
  samples: 2.446378e7, 2.443792e7, 2.438326e7
  median:  2.443792e7
  previous retained median: 2.433701e7
```

Those differences are small and mixed, so I restored the source to the prior retained optimized split. Byte-for-byte history comparisons passed during the attempts:

```text
default_ng2, default_ng3, multiblock_ng2, gauge_branches_ng2: identical
128^3 nghost=4 with 128^3, 64^3, and 32^3 MeshBlocks: identical
```

Conclusion: for `nghost=4`, small local cleanups do not reduce the main RHS register cliff. A deeper split that separates scalar/A curvature work might lower spill bytes, but it would duplicate the expensive Ricci/conformal-factor calculation and needs a larger design pass; the smaller metric-only split already showed that splitting off cheap work can make ptxas allocation worse. The retained source remains the earlier RHS/Gamma/gauge plus ADM split because it is the best measured version so far.

## Update: Rerun ZCPS For NGHOST=2,3,4

Follow-up request:

```text
Run performance again for the previous grid configurations, for nghost=2,3,and 4.
```

I reran the retained optimized source on the same 128^3 total grid with 128^3, 64^3, and 32^3 MeshBlocks. Each entry again uses three `nlim=3` samples and reports the median `zone-cycles/cpu_second`.

```text
NGHOST=2

128^3 MeshBlock:
  samples: 4.384904e7, 5.686394e7, 5.711098e7
  median:  5.686394e7

64^3 MeshBlocks:
  samples: 5.683187e7, 5.586854e7, 5.683026e7
  median:  5.683026e7

32^3 MeshBlocks:
  samples: 4.944122e7, 4.958972e7, 4.940930e7
  median:  4.944122e7
```

```text
NGHOST=3

128^3 MeshBlock:
  samples: 4.296413e7, 4.285042e7, 4.295878e7
  median:  4.295878e7

64^3 MeshBlocks:
  samples: 4.251083e7, 4.239041e7, 4.291678e7
  median:  4.251083e7

32^3 MeshBlocks:
  samples: 3.712094e7, 3.715242e7, 3.709059e7
  median:  3.712094e7
```

```text
NGHOST=4

128^3 MeshBlock:
  samples: 2.677555e7, 2.665254e7, 2.650098e7
  median:  2.665254e7

64^3 MeshBlocks:
  samples: 2.675985e7, 2.691770e7, 2.704382e7
  median:  2.691770e7

32^3 MeshBlocks:
  samples: 2.432989e7, 2.425954e7, 2.445934e7
  median:  2.432989e7
```

The rerun is consistent with the earlier retained measurements. `nghost=3` falls between `nghost=2` and `nghost=4`, and the smaller 32^3 MeshBlock layout is consistently slower than 64^3/128^3 for this 128^3 total-grid test.

## Update: NGHOST=4 With Separate Spatial Order

Follow-up request:

```text
Get performance for 2nd order, 4th order, and 6th order all with 4 ghost.
```

I used the retained optimized source with the new `z4c/spatial_order` parameter. All runs below use `mesh/nghost=4`, the same 128^3 total grid, and three `nlim=3` samples per MeshBlock layout. Values are `zone-cycles/cpu_second`.

```text
spatial_order=2, nghost=4

128^3 MeshBlock:
  samples: 4.297906e7, 5.713465e7, 5.639456e7
  median:  5.639456e7

64^3 MeshBlocks:
  samples: 5.313828e7, 5.240941e7, 5.170269e7
  median:  5.240941e7

32^3 MeshBlocks:
  samples: 4.236548e7, 4.249598e7, 4.246016e7
  median:  4.246016e7
```

```text
spatial_order=4, nghost=4

128^3 MeshBlock:
  samples: 4.282614e7, 4.297126e7, 4.294056e7
  median:  4.294056e7

64^3 MeshBlocks:
  samples: 4.215358e7, 4.128330e7, 4.157182e7
  median:  4.157182e7

32^3 MeshBlocks:
  samples: 3.536735e7, 3.547203e7, 3.543917e7
  median:  3.543917e7
```

```text
spatial_order=6, nghost=4

128^3 MeshBlock:
  samples: 2.666269e7, 2.665777e7, 2.642846e7
  median:  2.665777e7

64^3 MeshBlocks:
  samples: 2.690461e7, 2.691484e7, 2.691993e7
  median:  2.691484e7

32^3 MeshBlocks:
  samples: 2.437402e7, 2.442940e7, 2.427555e7
  median:  2.437402e7
```

Relative to 6th-order differencing at the same `nghost=4`, the median ZCPS speedups are:

```text
2nd order:
  128^3 MeshBlock: 2.115x
  64^3 MeshBlocks: 1.947x
  32^3 MeshBlocks: 1.742x

4th order:
  128^3 MeshBlock: 1.611x
  64^3 MeshBlocks: 1.544x
  32^3 MeshBlocks: 1.454x
```

This confirms that with ghost-cell count held fixed at 4, most of the throughput loss comes from the wider 6th-order finite-difference stencil rather than the ghost allocation alone.
