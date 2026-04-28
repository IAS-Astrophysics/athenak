# Dynamical Radiation Test Matrix

This file records the tests added for the standalone `dyn_radiation` solver and
how they map onto the verification suite in `radiation_method.tex`.  The solver
contract and runtime-mode notes are in `docs/dyn_radiation_solver.md`.

## Added Inputs

All inputs below use `<dyn_radiation>` rather than `<radiation>`.

| Input | Coverage |
| --- | --- |
| `inputs/tests/dynrad_tetrad_cks.athinput` | CKS-mode construction and orthonormality check through the built-in `rad_beam` test pgen. |
| `inputs/tests/dynrad_tetrad_adm.athinput` | ADM-mode Eulerian tetrad construction and orthonormality check on flat ADM data. |
| `inputs/tests/dynrad_beam_cks.athinput` | CKS-mode transport, beam source term, radiation-moment output, physical boundary fills, and MPI communication. |
| `inputs/tests/dynrad_beam_adm_flat.athinput` | ADM-mode flat-spacetime transport, ADM flux path, and the zero-valued ADM geometric source path. |
| `inputs/tests/dynrad_crossing_beams.athinput` | CKS-mode crossing-beams transport. Used with `inputs/tests/rad_crossing_beams.athinput` by `dyngr_radiation_paper/scripts/run_crossing_beams.py` to compare the legacy CKS, dyn CKS, and dyn ADM branches against the analytic two-beam solution and check angular convergence. |
| `inputs/tests/dynrad_crossing_beams_adm.athinput` | ADM-mode crossing-beams transport on analytic Minkowski ADM data with `<adm dynamic=false>` and no Z4c. Used by `run_crossing_beams.py` as the ADM branch of the angular-convergence comparison. |
| `inputs/tests/dynrad_kerr_orbit_beam.athinput` | CKS-mode Kerr photon-orbit beam. Used with `inputs/tests/rad_kerr_orbit_beam.athinput` by `dyngr_radiation_paper/scripts/run_kerr_orbit_beam.py` to compare the legacy CKS, dyn CKS, and dyn ADM branches on a compact `nx3=1` black-hole bending problem. |
| `inputs/tests/dynrad_kerr_orbit_beam_adm.athinput` | ADM-mode Kerr photon-orbit beam on analytic Kerr-Schild ADM data with `<adm dynamic=false>` and no Z4c. Used by `run_kerr_orbit_beam.py` to exercise the ADM curved-spacetime angular drift and source-direction projection. |
| `inputs/tests/dynrad_lwave.athinput` | Radiation-hydro coupling with the linear-wave pgen and the copied local implicit matter update. |
| `inputs/tests/dynrad_lwave_smr.athinput` | Same linear-wave coupling with static mesh refinement, covering radiation restriction/prolongation and AMR buffer packing for `pdynrad`. |
| `inputs/tests/dynrad_lwave_convergence.athinput` | CKS-mode radiation-hydro linear-wave convergence. Used with `inputs/tests/rad_lwave_convergence.athinput` by `dyngr_radiation_paper/scripts/run_linear_wave_convergence.py` to compare both solvers against the analytic eigenmode in density, velocity, gas pressure, \(R^{tt}\), and \(R^{tx}\). |
| `inputs/tests/dynrad_lwave_adm_convergence.athinput` | ADM-mode analytic-flat linear-wave convergence. It uses `<adm dynamic=false>` and a zero-field `<mhd>` wrapper because ADM/Z4c metric backgrounds are coupled through dynGRMHD rather than hydro. |
| `inputs/tests/dynrad_positivity_floor.athinput` | Conservative angular positivity limiter. The test seeds one negative primitive angular bin and checks nonnegative output plus conservation of the cell angular zeroth moment. It is run in CKS mode and with `dyn_radiation/geometry=adm` to cover both limiter branches. |
| `inputs/tests/dynrad_source_iteration.athinput` | Iterated nonlinear source solve with temperature-dependent opacity in fixed-fluid mode. The test checks positive primitive intensity and verifies that the source step changes the radiation field. |
| `inputs/tests/dynrad_equilibration.athinput` | Homogeneous gas-radiation thermal relaxation. Used with `inputs/tests/rad_equilibration.athinput` by `dyngr_radiation_paper/scripts/run_equilibration.py` to compare the legacy and new solvers against the exact energy-conserving relaxation ODE for 1, 10, and 100 source steps. |
| `inputs/tests/dynrad_flrw_redshift.athinput` | ADM FLRW redshift test with analytic `a(t)` and `K_ij`, checking `E ~ a^-4` and `sqrt(gamma) E ~ a^-1`. |
| `inputs/tests/dynrad_lapse_gradient.athinput` | Static analytic ADM lapse-gradient source test, checking the sign and shape of `-F^i partial_i alpha`. |
| `inputs/tests/dynrad_momentum_source.athinput` | Local ADM momentum-source closure test comparing the angular Hamiltonian-force sum against the Valencia momentum source. |

## Detailed Test Inventory

### Tetrad Construction

Inputs:

- `inputs/tests/dynrad_tetrad_cks.athinput`
- `inputs/tests/dynrad_tetrad_adm.athinput`

Purpose:

- Checks that CKS compatibility mode constructs the same orthonormal tetrad used
  by the legacy solver.
- Checks that ADM mode constructs the Eulerian tetrad on flat ADM data.
- Exercises the metric-normalization paths used later by transport, output, and
  matter coupling.

Expected result:

- Runs complete without non-finite values.
- Built-in tetrad checks pass at initialization.

### Flat Beam Transport

Inputs:

- `dyngr_radiation_paper/inputs/beam_radiation_cks.athinput`
- `dyngr_radiation_paper/inputs/beam_dynrad_cks.athinput`
- `dyngr_radiation_paper/inputs/beam_dynrad_adm_flat.athinput`
- `inputs/tests/dynrad_beam_cks.athinput`
- `inputs/tests/dynrad_beam_adm_flat.athinput`

Purpose:

- Verifies that CKS-compatible `dyn_radiation` reproduces the legacy beam exactly.
- Verifies that ADM-flat transport reproduces the same flat result when
  `geometry=adm`.
- Covers physical boundary fills, moment output, beam source normalization, MPI
  decomposition, and restart I/O.

Diagnostics:

- `dyngr_radiation_paper/figures/beam_comparison.png`
- `dyngr_radiation_paper/figures/beam_2d_fields.png`
- `dyngr_radiation_paper/figures/beam_2d_residuals.png`

Expected result:

- 1D and 2D CKS-compatible residuals against the legacy solver are zero to
  reported precision.
- ADM-flat residuals against the legacy solver are zero to reported precision.
- Two-rank MPI and restart/resume smoke tests complete.

### Linear Wave

Inputs:

- `dyngr_radiation_paper/inputs/lwave_radiation_cks.athinput`
- `dyngr_radiation_paper/inputs/lwave_dynrad_cks.athinput`
- `inputs/tests/rad_lwave_convergence.athinput`
- `inputs/tests/dynrad_lwave_convergence.athinput`
- `inputs/tests/dynrad_lwave_adm_convergence.athinput`
- `inputs/tests/dynrad_lwave.athinput`
- `inputs/tests/dynrad_lwave_smr.athinput`

Purpose:

- Verifies radiation-hydro source coupling and transport against the analytic
  damped linear eigenmode.
- Confirms that the standalone CKS-compatible branch reproduces the legacy error
  row.
- Runs the same eigenmode through the ADM geometry path using analytic flat ADM
  data and a zero-field MHD wrapper. This is not expected to be bitwise identical
  because it uses dynGRMHD primitives, but it should converge to the same
  analytic solution.
- Runs a resolution sweep over `nx1=16,32,64,128,256` and checks density,
  velocity, gas pressure, \(R^{tt}\), and \(R^{tx}\) against the analytic
  eigenmode.
- Covers SMR/AMR packing through `dynrad_lwave_smr.athinput`.

Diagnostics:

- `dyngr_radiation_paper/figures/linear_wave_errors.png`
- `dyngr_radiation_paper/figures/linear_wave_convergence.png`

Expected result:

- Legacy and `dyn_radiation` CKS error rows match.
- Legacy, `dyn_radiation` CKS, and `dyn_radiation` ADM convergence curves
  overlap to plotting precision.
- Current last-three-point `dyn_radiation` CKS convergence rates are
  `RMS=2.071`, `rho=1.849`, `u^x=2.129`, `Pgas=2.111`, `Rtt=2.121`,
  and `Rtx=2.247`.
- Current last-three-point `dyn_radiation` ADM convergence rates are
  `RMS=2.070`, `rho=1.849`, `u^x=2.129`, `Pgas=2.105`, `Rtt=2.121`,
  and `Rtx=2.247`.

### Crossing Beams

Inputs:

- `inputs/tests/rad_crossing_beams.athinput`
- `inputs/tests/dynrad_crossing_beams.athinput`
- `inputs/tests/dynrad_crossing_beams_adm.athinput`

Purpose:

- Verifies that two beams free-stream through one another without artificial
  interaction between angular bins.
- Runs the same flat-space problem through the analytic ADM path with
  `<adm dynamic=false>` and no Z4c.
- Initializes one-sided downstream Gaussian beams whose centerlines start at the
  marked source circles and pass through the requested crossing point.  Physical
  boundary fills keep the same analytic beam state in the ghost zones.
- Uses a positive all-angle maximum-entropy angular projection.  The injected
  angular energy is normalized exactly and the injected first moment is aligned
  with the requested beam direction at the finite-grid realizable flux factor,
  rather than being assigned to the nearest angular cell.
- Checks angular convergence for `nlevel=1..6` against the analytic
  straight-line Gaussian-beam solution.

Diagnostics:

- `dyngr_radiation_paper/figures/crossing_beams_comparison.png`
- `dyngr_radiation_paper/figures/crossing_beams_convergence.png`

Expected result:

- Legacy and `dyn_radiation` CKS maps agree to reported precision for each
  angular resolution.
- The analytic-flat ADM branch overlays the CKS branch visually; the current
  normalized \(L_\infty\) ADM-CKS differences range from `3.757567e-4` at
  `Nang=12` to `2.443248e-5` at `Nang=362`.
- Relative \(L_1\) error decreases with angular refinement; the current run
  decreases from `1.956930e-2` at `Nang=12` to `1.504625e-2` at `Nang=362`
  for CKS and from `1.956859e-2` to `1.504618e-2` for ADM.

### Kerr Photon-Orbit Beam

Inputs:

- `inputs/tests/rad_kerr_orbit_beam.athinput`
- `inputs/tests/dynrad_kerr_orbit_beam.athinput`
- `inputs/tests/dynrad_kerr_orbit_beam_adm.athinput`

Purpose:

- Checks stationary Kerr-Schild geodesic bending in the legacy solver and
  CKS-compatible `dyn_radiation`.
- Runs the same compact black-hole beam through the analytic Kerr-Schild ADM
  path with `<adm dynamic=false>` and no Z4c.
- Uses a compact `nx3=1` grid so the test can run locally.
- Projects the source direction onto the angular mesh rather than choosing only
  the nearest angular bin.

Diagnostic:

- `dyngr_radiation_paper/figures/kerr_orbit_beam_comparison.png`

Expected result:

- The high-intensity beam ridge follows the equatorial photon-orbit guide in
  all three branches.
- Current CKS `dyn_radiation` versus legacy normalized infinity-norm difference
  is `7.298162e-2`; relative \(L_1\) difference is `1.595626e-2`.
- Current ADM versus CKS `dyn_radiation` normalized infinity-norm difference is
  `4.497598e-2`; relative \(L_1\) difference is `7.477185e-2`.

### Gas-Radiation Equilibration

Inputs:

- `inputs/tests/rad_equilibration.athinput`
- `inputs/tests/dynrad_equilibration.athinput`

Purpose:

- Initializes homogeneous gas and radiation out of thermal equilibrium.
- Evolves only local absorption/emission coupling.
- Compares 1-step, 10-step, and 100-step source integrations to the exact
  energy-conserving relaxation ODE.

Diagnostic:

- `dyngr_radiation_paper/figures/equilibration_comparison.png`

Expected result:

- Legacy and `dyn_radiation` final states match.
- Current 100-step `dyn_radiation` errors are `max |Delta T|=1.712653e-2` and
  `max |Delta u|=2.568980e-2`.
- Final state in the current run is `Tgas=1.216323`, `Trad=1.214481`,
  `utot=4.000000`.

### ADM Formal Checks

Inputs:

- `inputs/tests/dynrad_flrw_redshift.athinput`
- `inputs/tests/dynrad_lapse_gradient.athinput`
- `inputs/tests/dynrad_momentum_source.athinput`

Purpose:

- Isolates ADM-only source terms on analytic ADM data with `<adm dynamic=false>`
  and no Z4c.
- Checks FLRW radiation redshift with `a(t)=1+0.2t`: `E ~ a^-4` and
  `sqrt(gamma) E ~ a^-1`.
- Checks the static lapse-gradient source sign using
  `alpha=1+0.1 sin(2 pi x)` and a positive x-directed flux.
- Checks the local angular Hamiltonian-force sum against the Valencia
  momentum source using the same finite-difference ADM metric-gradient caches
  used by the solver.

Diagnostic:

- `dyngr_radiation_paper/figures/adm_formal_tests.png`

Expected result:

- FLRW final relative errors at `t=0.5` are `1.18618e-4` for both `E` and
  `sqrt(gamma) E`.
- Lapse-gradient perturbation has correlation `0.999307` with the first-order
  prediction and relative RMS mismatch `4.68375e-2`.
- Momentum-source absolute residual decreases from `1.80196e-5` at `Nx=32` to
  `1.14669e-6` at `Nx=128`, consistent with second-order finite-difference
  cache error. The max relative residual at `Nx=128` is `8.47848e-4`.

### Positivity Limiter

Input:

- `inputs/tests/dynrad_positivity_floor.athinput`

Purpose:

- Seeds one negative primitive angular bin.
- Exercises the conservative angular redistribution limiter.
- Runs in both CKS compatibility mode and ADM mode.

Expected result:

- Output primitive intensities are nonnegative.
- The local angular zeroth moment is conserved by the limiter in both branches.

### Iterated Source Coupling

Input:

- `inputs/tests/dynrad_source_iteration.athinput`

Purpose:

- Exercises the nonlinear source iteration with temperature-dependent opacity.
- Keeps the fluid fixed while requiring the radiation field to change, testing
  the source solve independently of fluid feedback.

Expected result:

- Primitive intensities remain positive.
- Radiation field changes by more than the configured minimum and less than the
  configured maximum.
- The fixed-fluid guard rejects incompatible `fixed_fluid=true,
  affect_fluid=true` configurations.

### ADM/Z4c and Binary-Background Smoke Tests

Inputs:

- `inputs/tests/dynrad_bh_beam_adm.athinput`
- `inputs/tests/dynrad_z4c_wave_adm.athinput`
- `dyngr_radiation_paper/inputs/dynbbh_beam_particles.athinput`

Purpose:

- Exercises ADM angular geodesic drift, ADM timestep reductions, and excision
  masking on non-flat backgrounds.
- Checks that Z4c-to-ADM metric data can refresh the radiation geometry cache.
- Tests the ADM beam source and null-particle beam-edge diagnostic on the
  superposed binary black-hole background.  The dynbbh beam input uses
  `<adm dynamic=true>` because its analytic ADM callback depends on
  `pmesh->time`; the plotted dashed guide is the fixed coordinate source axis,
  not an expected geodesic in the curved, time-dependent spacetime.

Diagnostics:

- `dyngr_radiation_paper/figures/dynbbh_beam_particles.png`
- `dyngr_radiation_paper/figures/dynbbh_beam_rtt_slices.png`
- `dyngr_radiation_paper/figures/dynbbh_beam_adm_background.png`
- `dyngr_radiation_paper/figures/dynbbh_beam_particles_lineout.png`
- `dyngr_radiation_paper/figures/dynbbh_beam_summary.png`

Expected result:

- Stationary black-hole ADM beam completes with finite output.
- Z4c linear-wave ADM smoke test completes without NaNs.
- dynbbh beam and coupled MHD/radiation smoke tests complete in serial; the
  compact dynbbh smoke also completes under two-rank MPI.
- The longer figure run on the same compact dynbbh input reaches `t=10` on four
  MPI ranks with `72 x 72 x 4` cells and `nlevel=4`.  The current run has
  lineout `max Rtt=6.247600e0` and lineout integral `6.491427e0`.

## Commands Used In This Port

```bash
cmake --build build -j6
./build/src/athena -i inputs/tests/dynrad_tetrad_cks.athinput
./build/src/athena -i inputs/tests/dynrad_tetrad_adm.athinput
./build/src/athena -i inputs/tests/dynrad_lwave.athinput job/basename=/tmp/dynrad_lwave_input_smoke
./build/src/athena -i inputs/tests/dynrad_lwave_smr.athinput job/basename=/tmp/dynrad_lwave_smr_smoke
python dyngr_radiation_paper/scripts/run_linear_wave_convergence.py --run-dir /tmp/dynrad_linear_wave
./build/src/athena -i inputs/tests/dynrad_positivity_floor.athinput job/basename=/tmp/dynrad_positivity_floor_smoke
./build/src/athena -i inputs/tests/dynrad_positivity_floor.athinput job/basename=/tmp/dynrad_positivity_floor_adm_smoke dyn_radiation/geometry=adm
./build/src/athena -i inputs/tests/dynrad_source_iteration.athinput job/basename=/tmp/dynrad_source_iteration_smoke
./build/src/athena -i inputs/tests/dynrad_beam_cks.athinput job/basename=dynrad_beam_input_smoke
./build/src/athena -i inputs/tests/dynrad_beam_adm_flat.athinput job/basename=dynrad_beam_adm_flat_smoke
./build/src/athena -i inputs/tests/dynrad_crossing_beams.athinput job/basename=dynrad_crossing_beams_smoke
./build/src/athena -i inputs/tests/dynrad_crossing_beams_adm.athinput job/basename=dynrad_crossing_beams_adm_smoke
python dyngr_radiation_paper/scripts/run_crossing_beams.py --run-dir /tmp/dynrad_crossing_beams
./build/src/athena -i inputs/tests/dynrad_kerr_orbit_beam_adm.athinput -d /tmp/dynrad_kerr_adm_smoke job/basename=dynrad_kerr_adm_smoke time/nlim=1 time/tlim=0.01 output1/dt=0.01
python dyngr_radiation_paper/scripts/run_kerr_orbit_beam.py --run-dir /tmp/dynrad_kerr_orbit_beam
./build/src/athena -i inputs/tests/dynrad_equilibration.athinput -d /tmp/equil_smoke job/basename=dynrad_equil_smoke time/nlim=1 time/tlim=0.01 output1/dt=0.01
python dyngr_radiation_paper/scripts/run_equilibration.py --run-dir /tmp/dynrad_equilibration
python dyngr_radiation_paper/scripts/run_adm_formal_tests.py --run-dir /tmp/dynrad_adm_formal
mpirun -n 2 ./build/src/athena -i inputs/tests/dynrad_beam_cks.athinput job/basename=dynrad_beam_mpi_smoke
./build/src/athena -i inputs/tests/dynrad_beam_cks.athinput job/basename=dynrad_restart_test output1/file_type=rst output1/dt=0.1 time/tlim=0.1
./build/src/athena -r rst/dynrad_restart_test.00001.rst time/tlim=0.12 job/basename=dynrad_restart_resume output1/file_type=tab output1/dt=0.12
/Users/hengrui/miniforge3/envs/jupyterenv/bin/python dyngr_radiation_paper/scripts/run_stress_tests.py --run-dir /tmp/dynrad_stress_latest --keep-going
```

The generated diagnostic plots can be reproduced with:

```bash
python3 dyngr_radiation_paper/scripts/plot_comparisons.py
python dyngr_radiation_paper/scripts/run_linear_wave_convergence.py --run-dir /tmp/dynrad_linear_wave
python dyngr_radiation_paper/scripts/run_crossing_beams.py --run-dir /tmp/dynrad_crossing_beams
python dyngr_radiation_paper/scripts/run_kerr_orbit_beam.py --run-dir /tmp/dynrad_kerr_orbit_beam
python dyngr_radiation_paper/scripts/run_equilibration.py --run-dir /tmp/dynrad_equilibration
python dyngr_radiation_paper/scripts/run_adm_formal_tests.py --run-dir /tmp/dynrad_adm_formal
cmake --build build_dynbbh_rad -j6
python dyngr_radiation_paper/scripts/run_dynbbh_beam_figures.py --run-dir /tmp/dynbbh_beam_figures --athena build_dynbbh_rad/src/athena
```

The beam output was checked for finite values and for a positive transported
`r00` signal.  The flat CKS and flat ADM beam outputs were also compared
component-by-component and matched to tabular precision.  The linear-wave output
was checked for finite errors.  The current full stress harness passed 26/26
cases, including serial solver-quality inputs, two-rank MPI cases, restart,
expected unsupported-mode rejections, and dynbbh smoke tests.

## Mapping To `radiation_method.tex`

| Paper test | Current dyn-radiation status |
| --- | --- |
| Colliding beams | Covered by the new `rad_crossing_beams` pgen for legacy CKS, dyn CKS, and analytic-flat dyn ADM, with an analytic straight-line Gaussian-beam comparison and angular-convergence plot. |
| Beams in curvilinear coordinates | The current committed dyn-radiation input covers CKS/Minkowski beam transport. The old snake-coordinate pgen is not part of the default built-in pgen set and still assumes the legacy radiation object. |
| Beams around black holes | Covered by the compact Kerr photon-orbit beam pgen for legacy CKS, dyn CKS, and analytic Kerr-Schild dyn ADM. The test uses a continuous projected source on an equatorial Kerr photon orbit, `nx3=1`, and a dashed analytic orbit guide in the generated figure. |
| Hohlraums | Not yet ported to a built-in dyn-radiation test input; the existing hohlraum pgen is a custom pgen outside the default test binary. |
| Radiating disk | Not yet ported. The `gr_torus`/`dynbbh` initializers still initialize the legacy radiation object only. |
| Equilibration | Covered by the new `rad_equilibration` pgen for both solvers. The runner compares gas/radiation temperatures and energies against the exact homogeneous relaxation ODE for 1, 10, and 100 timesteps. |
| Diffusion | Covered indirectly by the optically thick linear wave. The dedicated diffusion pgen remains legacy-only. |
| Shocks | No dedicated dyn-radiation shock input exists in this pass. |
| Linear waves | Covered by `dynrad_lwave.athinput`, `dynrad_lwave_smr.athinput`, `dynrad_lwave_convergence.athinput`, and `dynrad_lwave_adm_convergence.athinput`. The convergence runner checks the coupled fluid/radiation eigenmode against the analytic solution and shows matching legacy, dyn CKS, and dyn ADM behavior with near-second-order convergence. |
| Schwarzschild atmosphere | Not yet ported; there is no compact built-in atmosphere test in this tree. |

The current implementation therefore provides regression coverage for every new
code path introduced in `dyn_radiation`: CKS transport, ADM tetrad construction,
ADM geometric energy source setup, FLRW redshift, lapse-gradient source signs,
ADM momentum-source closure, ADM flat crossing-beams free streaming, Kerr ADM beam bending,
homogeneous gas-radiation equilibration,
conservative positivity limiting, iterated
source coupling, output, restart, MPI, and SMR/AMR data motion.
It does not yet claim a full one-for-one reproduction of every production-scale
test in the paper.
