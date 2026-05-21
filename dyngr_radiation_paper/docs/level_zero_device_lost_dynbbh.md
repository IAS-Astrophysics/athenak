# Dynbbh Level Zero Device-Lost Investigation

This note documents the May 18, 2026 dynbbh reproduction failure seen with
the SYCL Level Zero backend.  The investigation below is based on existing
logs and source inspection only; the GPU workload was not rerun because it can
crash the workstation.

## Symptom

The failing logs report:

```text
There was a synchronous SYCL error:
level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```

The last normal dynbbh output before the failure is:

```text
Found torus outer edge: 12.7515
```

This occurs during dynbbh problem initialization, before the production
evolution loop.

## CPU MPI Reproduction Path

Use a CPU MPI dynbbh executable for paper reproduction:

```sh
cmake -S . -B build-mpi-dynbbh \
  -D CMAKE_BUILD_TYPE=Release \
  -D Athena_ENABLE_MPI=ON \
  -D PROBLEM=dynbbh \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_OPENMP=OFF \
  -D Kokkos_ENABLE_SYCL=OFF
cmake --build build-mpi-dynbbh -j6
```

Do not run `mpirun` with a non-MPI dynbbh executable.  That starts multiple
independent serial Athena processes, each of which believes `nranks=1` and can
write conflicting shared outputs such as tracked-particle files.  The
dynbbh figure driver now checks the executable configuration before launching
an MPI run.

## Likely GPU Failure Mechanism

The strongest static culprit is excessive per-work-item private state in the
dynbbh analytic ADM metric path:

- `src/pgen/dynbbh.cpp:815` calls `SetADMVariablesToBBH()` during
  initialization when `<dyn_radiation>` is enabled.
- `src/pgen/dynbbh.cpp:1687` launches `update_adm_vars` over all cells plus
  ghost zones.
- Each work item calls `analytic_4metric()`, which calls
  `get_metric_and_derivatives()`.
- `src/pgen/dynbbh.cpp:2297` creates local `dual_real gcov[4][4]` and
  `dual_real td[NTRAJ]`.
- `src/pgen/dynbbh.cpp:2274` creates four more local `dual_real[4][4]`
  arrays when `SuperposedBBHTemplate<dual_real>()` is instantiated.

With the current double-precision `dual_real` layout, each value stores one
real value and four derivatives.  Before counting temporaries introduced by
the many overloaded arithmetic operations, the derivative path already carries
several kilobytes of private data per work item.  On Intel Level Zero this can
spill heavily and can exceed runtime/compiler/device limits, causing a device
loss that is reported at a later synchronization point.

The observed "Found torus outer edge" line does not prove that the following
`pgen_torus1` kernel is the sole cause, because SYCL kernel failures can be
reported asynchronously.  It does constrain the crash to dynbbh initialization:
the earlier ADM/dynrad setup kernels and the first torus initialization kernels
are the relevant region.

## Recommended Fix Direction

For GPU support, the dynbbh ADM update should avoid the full dual-number
superposed-BBH derivative evaluation inside one large device kernel.  Plausible
paths are:

- split metric values and derivatives into smaller kernels with fewer live
  local arrays;
- replace the device-side dual-number derivative path with a lower-private-
  memory analytic derivative implementation;
- compute derivatives from stored metric fields using stencil operations when
  that is mathematically acceptable for the target problem;
- add debug fences around `SetADMVariablesToBBH()`, `PrepareADMGeometry()`,
  and `pgen_torus1` only on a machine where GPU crashes are acceptable, to
  confirm the exact reporting point.

Until that refactor is done, dynbbh paper reproduction should use the CPU MPI
path above.
