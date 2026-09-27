# Z4c overhaul

This branch starts at IAS-Astrophysics/main (`c5a0d7f9155a70149931bf0be5a4ffb673f2532a`).
Each feature group is introduced separately. Upstream's single boosted-puncture test
problem, FastFlow horizon finder, drift controllers, and slow-start **lapse** remain
available. Slow-start **shift** and rolling kappa are not part of this migration.

## Initial data

* `problem/pgen_name=z4c_superposed_punctures` adds two analytic, signed x1 boosts.
  For `N=1,2`, set `punc_N_rest_mass` (positive), `punc_N_velocity_x1` (magnitude
  below one), and `punc_N_center_x1/x2/x3`. Defaults are unit rest masses, zero
  speeds, and centers (-2,0,0)/(2,0,0). `puncture_radius_floor=1e-8` regularizes
  the puncture evaluation. Initial lapse is precollapsed and initial shift is zero.
  **Superposition is approximate initial data, not a solution of the binary
  constraint equations.** Inspect constraints before scientific use.
* For external data, configure `cmake -S . -B build-id -D PROBLEM=id_solve` with
  HDF5 C development libraries installed. Set `problem/id_filename` to the file.
  The schema is `metric` and `extrin` with shape `[6,block,z,y,x]`, component order
  `(xx,xy,xz,yy,yz,zz)`, and `x1v/x2v/x3v` with shape `[block,n_axis]`.
  Coordinates must be finite, strictly increasing, and contain at least five
  samples per axis. Noncubic blocks are supported. Five-point tensor Lagrange
  interpolation prefers finer active coverage over source ghost coverage;
  `problem/id_source_nghost=0` describes source padding independently of the
  simulation's ghost width. Source coordinates must cover destination ghosts.
  Nonfinite interpolated data and non-positive-definite metrics are rejected.
  Optional `z4c/r_fill>0` blends the interior to an analytic puncture metric with
  quintic smoothstep; masses `M_fill_0/1` and centers `co_0/1_x/y/z` control this
  modification. It is disabled by default and does not solve constraints.

## Gauge and discretization

`z4c/telegraph_lapse=false` is opt-in. With finite `telegraph_tau>0` (default 0.1)
and `telegraph_kappa>=0` (default 0.1), it adds

```
(dt - lapse_advect beta^j dj) B_i = (kappa di alpha - B_i)/tau
extra lapse RHS = gamma^{ij} di B_j
```

The added divergence uses partial derivatives, not a covariant divergence.
`B_i` has covector reflection parity. The timestep also accounts for the
relaxation time and the added metric-dependent characteristic speed. A suitable
CFL and a convergence study are still needed for each physical configuration.
The auxiliary fields are `z4c_Bx/By/Bz`; they are zero initially and are checkpointed.

`z4c/spatial_order=2/4/6` selects the Z4c derivative and dissipation stencil
independently of allocated ghost cells. Supported ghost allocations are 2 through 4, with at least `order/2+1` ghosts.
Omitting it retains the upstream mapping `2*(nghost-1)`. The selection also covers
ADM conversion/constraints, Weyl extraction, diagnostics and FastFlow derivatives.
The upstream Sommerfeld boundary stencil remains second order. This option does
not select the fluid reconstruction order or the time integrator order.

## Curvature and refinement

Output `variable=z4c_diag` writes 17 vacuum diagnostics; each name can also be
selected individually: `z4c_E{xx,xy,xz,yy,yz,zz}`, `z4c_B{xx,xy,xz,yy,yz,zz}`,
`z4c_P{x,y,z}`, `z4c_Pnorm`, and `z4c_Kretschmann`.

E and B are symmetric, trace-free Eulerian electric/magnetic curvature tensors.
The convention is `epsilon^{123}=+1/sqrt(det(gamma))` and
`P^i=-epsilon^{ijk} E_jl B_k^l`. The norm is the physical
`sqrt(gamma_ij P^i P^j)`, not the Cartesian component norm. Kretschmann is
`8(E_ij E^ij-B_ij B^ij)` in vacuum. Its identification with the spacetime invariant
assumes the vacuum field equations; off-constraint numerical data need that caveat.
Matter runs are rejected for these vacuum-only diagnostics. `z4c_Bxx`, etc. are
curvature components, distinct from the gauge auxiliary `z4c_Bx`, etc.

`z4c_amr/max_ref_lev=-1` leaves the new cap disabled. Nonnegative values specify
levels above the root grid (zero prevents Z4c refinement). The cap is applied
after chi, dchi, tracker and radius criteria; existing blocks above the cap request
derefinement. Mesh-level balancing and other independently registered criteria
remain under the upstream AMR driver.

## Extraction and restart

* Horizon dumps support `z4c/co_N_dump_cheb=true` and optional
  `z4c/horizon_N_rn` (nonnegative radial regularization power, default zero).
  Uniform and Chebyshev coordinates are used consistently. Center evaluation
  avoids division by zero; changing extents and AMR refresh interpolation weights.
  Dump data are x-fast. Each dump includes `coordinates.txt`. ETK uniform-grid
  parameter files are emitted only for uniform grids; Chebyshev consumers must
  use the explicit coordinates.
* Cartesian and spherical extraction assign shared block faces to a single owner,
  including across MPI ranks, while retaining the outer domain face.
* CCE uses N roots of the second-kind Chebyshev polynomial with denominator N+1,
  consistent angular sampling, and conjugated spherical-harmonic projection.
  Imaginary coefficient signs and radial sample locations therefore change from
  the upstream buggy convention. Files use full-precision scientific time names;
  shells after zero include `shellN_` to avoid overwriting another shell.
* Restarts preserve horizon counters/cadence, CCE cadence, and complete controller
  position/velocity/error/integral/observer/budget history. Parameter real values
  round-trip at full floating-point precision. Tracker and controller logs append.
  Shared restart files and one file per rank are supported.
* **Restart layout changes:** the three new gauge fields increase Z4c storage from
  22 to 25 fields, even when telegraph lapse is disabled. Existing 22-field upstream
  restart files are rejected by the size check; there is no automatic conversion.

## Kernels, conversion and tests

Geometry, connection and gauge RHS kernels are separated; Hamiltonian and
momentum/Z constraint kernels are separated. This reduces simultaneous scratch
storage, with disjoint writes to the same RK stage's RHS. Performance depends on
backend and problem size; the regression comparisons verify numerical equivalence,
not a universal speedup. Binary-to-HDF5 conversion now uses actual output indices
for sliced, ghost-padded and singleton dimensions.

The portable tests in `tst/test_suite/z4c` accept absolute `ATHENA_OVERHAUL_EXE`
and `ATHENA_ID_EXE` paths. `ATHENA_REFERENCE_EXE` enables comparisons to the saved
pre-split executable (`8cf76b1d5` on CPU); omit those tests if no reference is built.
For selected multi-rank extraction/restart tests use `ATHENA_TEST_LAUNCHER='mpiexec -n 2'`.
All subprocesses have timeouts. The feature suite covers analytic puncture limits,
invalid parameters, ghost-independent FD order, Schwarzschild curvature convergence,
independent metric contractions of P, AMR caps, interpolation polynomials, analytic
CCE coefficients, restart equivalence, reflection parity, HDF5 schema/coverage and
block selection, and a bounded integration test of upstream FastFlow.

See the pull request validation record for tested backends and outcomes. These are
short regression and convergence checks, not a long binary-black-hole production
validation.

### Reproduce the targeted suite

```sh
cmake -S . -B build-cpu
cmake --build build-cpu -j 4
cmake -S . -B build-id -DPROBLEM=id_solve
cmake --build build-id -j 4
ATHENA_OVERHAUL_EXE="$PWD/build-cpu/src/athena" \
ATHENA_ID_EXE="$PWD/build-id/src/athena" \
python -m pytest -q tst/test_suite/z4c/test_z4c_overhaul_cpu.py \
  tst/test_suite/z4c/test_z4c_id_import_cpu.py \
  tst/test_suite/z4c/test_z4c_restart_cpu.py \
  tst/test_suite/z4c/test_z4c_conversion_cpu.py \
  tst/test_suite/z4c/test_z4c_fastflow_cpu.py
```

These tests require pytest and NumPy; HDF5/conversion tests additionally require
h5py. The twelve kernel comparisons are a separate optional module,
`test_z4c_kernel_equivalence_cpu.py`, requiring `ATHENA_REFERENCE_EXE`.
Configure CUDA with `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
-DCMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper` and use the corresponding
`_gpu.py` modules. Configure MPI with `-DAthena_ENABLE_MPI=ON` and run
`test_z4c_restart_cpu.py`, `test_z4c_fastflow_cpu.py`, and the `cartesian or cce`
subset of `test_z4c_overhaul_cpu.py` with the launcher environment above.

### Validation record (2026-09-27)

* Apple Silicon serial CPU: all 62 targeted tests passed. Import tests also passed
  with AddressSanitizer and Kokkos bounds/DualView checks (8 tests; macOS leak
  detection is unavailable).
* Two MPI ranks: 14 targeted extraction, restart and FastFlow checks passed.
  The upstream 2D AMR wave convergence case gave RMS L1 errors
  `2.395272e-10` and `5.838634e-11` (ratio `0.2437566`, required <=0.25).
* Upstream CPU AMR/outflow boundary case: L-infinity `2.362894e-14`
  (required <1e-12).
* NVIDIA A100-PCIE-40GB on della-vis1, CUDA 12.8: all 62 targeted tests passed,
  including the HDF5 build and twelve comparisons to the pre-split CUDA kernels.
  The node's 364 source/test files matched the local SHA-256 manifest. Changed
  sources were explicitly rebuilt after transfer to avoid stale timestamp reuse.
* Full upstream 200-step boosted-puncture/FastFlow GPU regression passed all
  original thresholds: C `0.0179735`, H `0.00465582`, M `0.00137659`, Z
  `0.00297762`, Theta `3.0628e-5`, and final horizon residual `0.02537883`.
* Upstream 3D AMR second-order GPU case: RMS L1 errors `1.424621e-10` and
  `3.159706e-11`, passing its error and convergence limits.
* The sixth-order 3D AMR GPU case passed with RMS L1 `5.115880e-12`
  (required <6e-12). Its allocation capacity was reduced from 4096 to 3712
  blocks to fit the shared 40 GB device; physical resolution, AMR criteria,
  integration interval and numerical threshold were unchanged.
* `git diff --check` and Python lint passed. C++ lint on changed files has four
  pre-existing one-line-if formatting findings, verified against upstream; the
  added code has no remaining lint findings under the repository configuration.

GPU executions were limited by subprocess and outer timeouts. The initial
4096-block sixth-order allocation exceeded available shared-device memory; the
3712-block retry completed successfully within the bounded run.
