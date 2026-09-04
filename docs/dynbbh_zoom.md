# Restart-driven circumbinary disk zoom

The `circumbinary_stable` strategy also works with `project/ttorus`'s
`dyn_grmhd/dynbbh`: relax an outer disk on a coarse mesh, restart with stronger
AMR targets, resolve the holes, then gradually shrink the puncture masks. MHD
uses existing conservative/CT AMR; ADM radiation uses existing intensity AMR,
flux correction and restart payloads. The helper supports either GRMHD or
radiative GRMHD **from the start**, without changing physics between stages.

## Generate a plan

Build with `PROBLEM=dyn_grmhd/dynbbh` and your usual Kokkos backend. Start from
your validated disk input, with or without `<dyn_radiation>` (`geometry=adm`).
Copy/edit `inputs/dyn_grmhd/dynbbh_zoom_schedule.json`, then run:

```sh
python3 scripts/setup_dynbbh_zoom.py \
  --input /absolute/path/to/disk.athinput \
  --schedule inputs/dyn_grmhd/dynbbh_zoom_schedule.json \
  --output-dir /absolute/path/to/cbd-zoom
```

The example schedule uses absolute code time (normally total-mass units):

| Stage | End time | BH tracker level | COM refinement | Excision |
| --- | ---: | ---: | --- | --- |
| outer | 1000 | 0 | root only | requested radii |
| inner | 1400 | 2 | r<30: level 1; r<12: level 2 | requested radii |
| resolve | 1600 | 5 | same | requested radii |
| shrink | 2000 | 5 | same | smooth ramp to 0.8 rH |
| settle | 2400 | 5 | same | follows 0.8 rH |

These times/radii/levels are **illustrations, not recommended equilibration
times**. The analytic binary period is `2*pi*sep**1.5` for total mass one; the
outer disk can relax much more slowly. Choose durations from inflow/equilibration
diagnostics. Add intermediate stages if needed. `regions` contains
`[COM_radius, physical_level]` pairs; trackers follow the holes. Level zero is
root, each level halves dx, and generated `num_levels = max_level + 1`.
Adaptive infrastructure is enabled from the beginning, but `outer` requests
only root blocks. The small `tst/inputs/dynbbh_radiation_cbd.athinput` is a
smoke-test base, not a converged disk.

The helper refuses existing output directories, static refinement regions,
non-user AMR criteria and `unresolved_sink=true`. That independent MHD-only
sink does not follow the excision shrink schedule; remove it deliberately from
a copy of your input. If absent, `coord/dexcise` and `pexcise` are copied from
the MHD density/pressure floors; review these and your smooth-excision settings.
Relative `traj_file` paths are resolved against the base
input's directory. The table must cover the **entire** run. Use absolute paths
for any other external input files.

## Launch and advance

`RUN.txt` lists commands for every stage. For example, on a compute node:

```sh
export ATHENA=/absolute/path/to/build/src/athena
export LAUNCHER=srun       # site-specific launcher; empty for serial
$LAUNCHER "$ATHENA" -i /absolute/path/to/cbd-zoom/outer/stage.athinput \
  -d /absolute/path/to/cbd-zoom/outer

# After outer reaches its tlim, select its final checkpoint explicitly:
$LAUNCHER "$ATHENA" -r /absolute/path/to/cbd-zoom/outer/rst/outer.NNNNN.rst \
  -i /absolute/path/to/cbd-zoom/inner/stage.athinput \
  -d /absolute/path/to/cbd-zoom/inner
```

Repeat for `resolve`, `shrink`, `settle`, using each preceding final checkpoint.
For per-rank restarts pass the rank-00000000 file, retain all sibling files and
keep the MPI rank count unchanged. A walltime stop is not stage completion:
resume the same stage with `-r` and its existing `-d`, **without `-i`**.
Do not use the full first-stage deck on a restart.

Later decks override only basename, tlim and AMR targets, preserving fluid,
face-centered B, radiation intensities/angles, mesh layout and output counters
from the checkpoint. The first deck clears stale counters and removes a base
`nlim` cap. Cadences otherwise come from the base, except restart cadence from
the schedule. No helper guesses checkpoints by mtime or parses binary structs.
Do not enable/disable radiation or change quadrature in the middle of this plan.
Checkpoint recovery rebuilds the current puncture masks and does not reapply the
smooth primitive projection to successfully recovered cells. Failed recovery
still retains the existing excision fallback. AMR also rebuilds masks after
recreating coordinates, before primitive recovery.

## Refine before shrinking

Final AMR targets are requested throughout `resolve`, before shrinking starts.
Levels are added on evolution cycles, subject to `refinement_interval` and
`ncycle_check`. **Inspect actual local mesh coverage at both moving holes before
advancing to `shrink`**; a requested maximum level is not proof of resolution.
Allow ample steps for refinement and settling. Size `max_nmb_per_rank` for memory
limits; radiation storage also scales with angular bins.

For an unboosted isolated hole, estimate `rH=M*(1+sqrt(1-|chi|^2))`,
`dx_final=dx_root/2**tracker_level`, and diameter coverage `2*rH/dx_final`.
Use the smallest mass/highest spin in the table and account for Lorentz
contraction and motion. `require_resolved_horizon` is only a weak finest-cell
setup check, disabled for the coarse phase; it is **not** a production resolution
gate. Choose resolution by convergence of observables.

`coord/excise_horizon_fraction` defaults to 1, range `(0,1]`, and scales the target
of automatic, capped, direct-to-horizon and shrinking modes. Explicit positive
fixed radii stay unchanged unless capped/ramped. The generated ramp is:

```
target(t) = horizon_fraction * rH(t)
s = clamp((t - shrink_start_time)/shrink_timescale, 0, 1)
R(t) = (1 - s*s*(3 - 2*s))*max(requested_radius, target(t))
       + s*s*(3 - 2*s)*target(t)
```

Targets use each hole's current mass/spin at the RK stage. The absolute clock
survives restarts; never reset it at `settle`. Time-dependent mass/spin can make
R(t) nonmonotonic despite the smooth blend. Masks use the boosted, spin-oriented
Kerr-Schild radius, not a Cartesian sphere. **rH is an isolated-Kerr estimate in
a superposed binary metric, not a measured binary horizon**: 0.8 rH does not
prove causal excision. Check horizons and characteristic outflow for your binary.
The expanded flux/FOFC mask can extend outside the density-excision radius.

Existing radiation paths zero intensities inside the mask. Newly uncovered
cells refill by evolution, not LTE initialization at each restart. Enlarged masks
deliberately remove material/radiation outside the holes during burn-in. Discard
shrink/refill transients and establish inner inflow equilibrium before measuring
accretion, luminosity or spectra. Monitor floors, magnetization, divB, radiation
positivity, energy exchange and mesh coverage. Short tests do not establish
long-term disk equilibrium.

## Regression commands

```sh
python3 tst/unit/test_dynbbh_zoom.py
PYTHONPATH=vis/python python3 tst/test_suite/dyngrmhd/dynbbh_zoom_common.py \
  --athena /absolute/path/to/build/src/athena --keep-artifacts
```

The end-to-end test requires NumPy. It evolves the small CBD deck through five
stages, both with and without radiation, checks finite positive fields and actual
BH refinement levels, and compares an interrupted/resumed shrink against an
uninterrupted shrink to double-precision tolerance. The integration test is also
enrolled in the CPU/GPU dynbbh radiation suites. It is a workflow regression,
not a disk convergence or equilibrium study. Failed runs retain their temporary
artifacts; successful tests clean them unless `--keep-artifacts` is requested.

Validation for this change (2026-09-04): Serial and four-thread OpenMP completed
all five stages for both models, with block counts `18 -> 32 -> 200 -> 200 -> 200`.
The interrupted shrink changed MHD output by at most `9.8e-17`; the radiation
moment slice was identical. The existing CPU metric/FD-convergence, excision,
refinement, radiation coupling/restart/activation and short CBD regressions also
passed. Fresh CUDA/MPI validation was not available: Della SSH authentication
was rejected. No long-duration equilibration or new GPU performance measurement
is implied by these tests. The implementation adds no device arrays or new kernel
types; it reuses mask updates at restart/AMR and a uniform primitive-recovery flag.
