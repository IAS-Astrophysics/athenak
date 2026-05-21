# Dynamical Radiation Paper Assets

This directory contains the method paper, compact comparison inputs, and plotting
script for the standalone `dyn_radiation` solver.  The ADM beam input runs with
`angular_fluxes=true`, exercising the cached ADM angular geodesic transport.
The beam inputs write both a 1D table slice and a 2D binary XY slice so the
script can produce line-profile, field-map, and residual figures.

The practical solver contract is documented in `docs/dyn_radiation_solver.md`.
The detailed test matrix, commands, expected outcomes, and plot inventory are in
`docs/dyn_radiation_tests.md`.

The intended quick comparison sequence from the repository root is:

```sh
./build/src/athena -i dyngr_radiation_paper/inputs/beam_radiation_cks.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/beam_dynrad_cks.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/beam_dynrad_adm_flat.athinput
./build/src/athena -i inputs/tests/dynrad_bh_beam_adm.athinput
./build/src/athena -i inputs/tests/dynrad_z4c_wave_adm.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/lwave_radiation_cks.athinput job/basename=/tmp/paper_lwave_rad
./build/src/athena -i dyngr_radiation_paper/inputs/lwave_dynrad_cks.athinput job/basename=/tmp/paper_lwave_dyn
python3 dyngr_radiation_paper/scripts/plot_comparisons.py
pdflatex -interaction=nonstopmode dyngr_radiation_method.tex
```

The plotting step requires Python with NumPy and Matplotlib.

The compact stress suite for the passive ADM/Z4c radiation path is:

```sh
python3 dyngr_radiation_paper/scripts/run_stress_tests.py
```

It runs serial, two-rank MPI, restart, unsupported-mode rejection, dynbbh smoke,
positivity-limiter, and nonlinear source-iteration cases with outputs under
`/tmp`.  The current matrix also includes the ADM FLRW redshift,
lapse-gradient, and momentum-source closure regressions and passed 26/26 cases
in the latest run.  In sandboxed environments `mpirun` may need to be run
outside the sandbox so it can bind local sockets.

The crossing-beams comparison can be regenerated with:

```sh
python dyngr_radiation_paper/scripts/run_crossing_beams.py --run-dir /tmp/dynrad_crossing_beams
```

It runs the legacy `radiation` solver, `dyn_radiation` in CKS compatibility
mode, and `dyn_radiation` with analytic Minkowski ADM data
(`<adm dynamic=false>`, no Z4c) on the same straight-line two-beam setup for
`nlevel=1..6`, writes the comparison and convergence figures, and stores the
error table in the run directory.  The pgen initializes one-sided downstream
Gaussian beams from the marked source circles and uses an all-angle positive
moment projection so the angular zeroth moment and requested first moment are
matched on the finite angular grid.

The compact Kerr beam comparison can be regenerated with:

```sh
python dyngr_radiation_paper/scripts/run_kerr_orbit_beam.py --run-dir /tmp/dynrad_kerr_orbit_beam
```

It runs the legacy CKS solver, `dyn_radiation` CKS, and `dyn_radiation` with an
analytic Kerr-Schild ADM metric (`<adm dynamic=false>`, no Z4c) on the paired
`rad_kerr_orbit_beam` inputs with `nx3=1`, injects a projected beam source
tangent to the equatorial Kerr photon orbit, and writes the normalized
legacy/dyn/ADM comparison figure.

The homogeneous gas-radiation equilibration comparison can be regenerated with:

```sh
python dyngr_radiation_paper/scripts/run_equilibration.py --run-dir /tmp/dynrad_equilibration
```

It runs both solvers with the built-in `rad_equilibration` pgen for 1, 10, and
100 source steps, compares gas/radiation temperatures and energies against the
exact energy-conserving relaxation ODE, and writes the comparison figure.

The radiation-fluid linear-wave convergence plot can be regenerated with:

```sh
python dyngr_radiation_paper/scripts/run_linear_wave_convergence.py --run-dir /tmp/dynrad_linear_wave
```

It runs the legacy CKS, `dyn_radiation` CKS, and analytic-flat ADM
`dyn_radiation` inputs over `nx1=16..256`, compares the final density,
velocity, gas pressure, \(R^{tt}\), and \(R^{tx}\) fields against the analytic
eigenmode, and writes the convergence figure plus a CSV error table.  The ADM
case uses `<adm dynamic=false>` and a zero-field `<mhd>` wrapper, since ADM/Z4c
backgrounds are integrated through dynGRMHD rather than `<hydro>`.

The ADM formal-regression figure can be regenerated with:

```sh
python dyngr_radiation_paper/scripts/run_adm_formal_tests.py --run-dir /tmp/dynrad_adm_formal
```

It runs analytic ADM backgrounds without Z4c: an FLRW redshift check
(`E \propto a^{-4}` and `sqrt(gamma) E \propto a^{-1}`), a static
lapse-gradient redshift/source check, and a local Hamiltonian-force versus
Valencia momentum-source closure check.  The script writes
`figures/adm_formal_tests.png` and a CSV diagnostic table in the run directory.

The compact dynbbh beam figure set can be regenerated with a dynbbh-enabled
executable:

```sh
cmake -S . -B build-mpi-dynbbh \
  -D CMAKE_BUILD_TYPE=Release \
  -D Athena_ENABLE_MPI=ON \
  -D PROBLEM=dynbbh \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_OPENMP=OFF \
  -D Kokkos_ENABLE_SYCL=OFF
cmake --build build-mpi-dynbbh -j6
python dyngr_radiation_paper/scripts/run_dynbbh_beam_figures.py \
  --run-dir /tmp/dynbbh_beam_figures \
  --athena build-mpi-dynbbh/src/athena
```

It runs the superposed orbiting Kerr-Schild binary background with
`dyn_radiation/geometry=adm` and `adm/dynamic=true`, so the analytic ADM fields
from the dynbbh pgen are refreshed during evolution.  It writes radiation
slices, the ADM lapse/psi4 background, null tracers, and a combined summary
figure.  The green guide in the plots is the fixed coordinate source axis; the
radiation and null tracers need not remain on that line in the curved,
time-dependent background.

The higher-resolution time-series version used for the current figures is:

```sh
python dyngr_radiation_paper/scripts/run_dynbbh_beam_figures.py \
  --run-dir /tmp/dynbbh_beam_highres_dynamic_t10 \
  --basename paper_dynbbh_beam_hi72z4_t10 \
  --athena build-mpi-dynbbh/src/athena --mpi-ranks 4 \
  --tlim 10.0 --nlim 340 --snapshot-dt 1.0 --track-dt 0.25 \
  --nx1 72 --nx2 72 --nx3 4 --mb-nx1 36 --mb-nx2 36 --mb-nx3 4 \
  --nlevel 4 --ppc 0.004 --ntrack 16 --beam-spread 20.0
```

This run keeps the same angular resolution as the compact Kerr beam diagnostic
(`nlevel=4`) and uses the minimum active z resolution accepted by the mesh
driver (`nx3=4`, `meshblock/nx3=4`).
When `--mpi-ranks` is greater than one, the executable must be built with
`Athena_ENABLE_MPI=ON`; otherwise independent serial jobs can collide on the
same output files.  The plotting script checks this before launching the run.

## Diagnostic plots

| Figure | How to regenerate | Diagnostic |
| --- | --- | --- |
| `figures/beam_comparison.png` | `plot_comparisons.py` | 1D flat beam agreement between legacy CKS, dyn CKS, and dyn ADM-flat. |
| `figures/beam_2d_fields.png` | `plot_comparisons.py` | 2D flat beam morphology for the same three branches. |
| `figures/beam_2d_residuals.png` | `plot_comparisons.py` | Residual maps against the legacy flat beam. |
| `figures/linear_wave_errors.png` | `plot_comparisons.py` | Final scalar error row for legacy and dyn CKS linear waves. |
| `figures/linear_wave_convergence.png` | `run_linear_wave_convergence.py` | Coupled fluid/radiation linear-wave convergence for legacy CKS, dyn CKS, and analytic-flat dyn ADM. |
| `figures/crossing_beams_comparison.png` | `run_crossing_beams.py` | Crossing beams for legacy CKS, dyn CKS, dyn ADM-flat, and the analytic two-beam solution. |
| `figures/crossing_beams_convergence.png` | `run_crossing_beams.py` | Angular convergence of the crossing-beams test in all three solver branches. |
| `figures/kerr_orbit_beam_comparison.png` | `run_kerr_orbit_beam.py` | Compact Kerr photon-orbit beam comparison, including analytic ADM Kerr-Schild transport. |
| `figures/equilibration_comparison.png` | `run_equilibration.py` | Homogeneous gas-radiation source relaxation against the exact ODE. |
| `figures/adm_formal_tests.png` | `run_adm_formal_tests.py` | ADM FLRW redshift, lapse-gradient source sign, and momentum-source closure. |
| `figures/dynbbh_beam_particles.png` | `run_dynbbh_beam_figures.py` | ADM BBH beam source and null-particle beam-edge diagnostic. |
| `figures/dynbbh_beam_rtt_slices.png` | `run_dynbbh_beam_figures.py` | \(R^{tt}\) beam slices on the superposed orbiting Kerr-Schild background. |
| `figures/dynbbh_beam_adm_background.png` | `run_dynbbh_beam_figures.py` | ADM lapse and conformal-factor contours for the same binary slice. |
| `figures/dynbbh_beam_particles_lineout.png` | `run_dynbbh_beam_figures.py` | Beam-core lineout and null beam-edge tracer tracks. |
| `figures/dynbbh_beam_summary.png` | `run_dynbbh_beam_figures.py` | Combined dynbbh beam/background/tracer diagnostic. |
| `figures/dynbbh_beam_timeseries.png` | `run_dynbbh_beam_figures.py` | Eight high-resolution dynbbh snapshots with accumulated null-particle trajectories. |

Current solver notes:

- ADM mode is metric-passive: ADM/Z4c geometry transports and bends radiation,
  but radiation stress-energy is not fed back to the metric solver.
- The ADM geometric energy source is applied with the closed-form exponential
  stage increment rather than a linear forward-Euler multiplier.
- Negative angular bins are handled by a local angular redistribution that
  conserves the angular zeroth moment when the pre-limited angular integral is
  nonnegative. It is not a momentum-conserving angular remap, so limiter
  activations remain a diagnostic for under-resolved transport.
- The radiation-matter source solve iterates temperature-dependent opacities
  with `source_max_iter` and `source_tolerance`; the quartic temperature solve
  uses a bracketed Newton-bisection root finder.
- ADM matter coupling includes the coordinate-to-comoving optical-depth factor
  `alpha * D * dt`, where `D` is the Eulerian-to-fluid Doppler factor.
- For radiation-fluid coupling, `fixed_fluid=true` defaults to
  `affect_fluid=false`; explicitly requesting both `fixed_fluid=true` and
  `affect_fluid=true` is rejected at startup.

The `dynbbh_beam_particles.athinput` input requires a build configured with
`PROBLEM=dynbbh`.  The dynbbh initializer now uses `pdynrad` only and rejects
the legacy `<radiation>` solver because that path is not consistent with the
ADM background.  Its default keeps radiation-matter coupling off for a clean
beam/particle smoke test; set `dyn_radiation/rad_source=true` to exercise the
ADM absorption/scattering coupling coefficients included in the input.
For MPI runs, only the rank that owns the beam source keeps source particles;
the pgen recomputes particle counts and tags after dropping non-source-rank
placeholders so tracked trajectories are not polluted by inactive rank-local
particles.
The null-particle timestep now uses the ADM coordinate-light-speed cap
`|beta^q| + alpha sqrt(gamma^{qq})`, rather than assuming unit coordinate speed.
