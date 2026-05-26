# Z4c Gravitational-Wave Collapse Free Data

Build this problem generator with:

```sh
cmake -S . -B build-gw-collapse -DPROBLEM=z4c_gravitational_wave_collapse
cmake --build build-gw-collapse --target athena
```

For the native CTS-solved critical-Kerr path, build:

```sh
cmake -S . -B build-critical-kerr -DPROBLEM=z4c_critical_kerr_formation
cmake --build build-critical-kerr --target athena
```

The `z4c_critical_kerr_formation` pgen fills the conformal metric and
trace-free conformal metric derivative described in the local notes, then the
`<id_solve>` task solves the CTS constraints and applies the resulting ADM/Z4c
data before evolution.

The pgen initializes smooth, compactly supported vacuum gravitational-wave free
data for Z4c collapse studies.  The parameterization follows the finite
dimensional conformal-data scaffold in
`/Users/hengrui/Desktop/research/gr/SPEK/critical_kerr_formation`: an
`ell=m` helical tensor seed, radial basis coefficients, a smooth truncation
envelope, an ingoing radial contribution, and a rotating quadrature
contribution.  The angular/time-parity choice follows the ingoing/rotating
prescription used in arXiv:2401.00805 at the pgen level: the radial part enters
through a one-way derivative and rotation uses the phase-quadrature tensor.

This is an AthenaK-native free-data construction followed by the native
`<id_solve>/formulation=cts` relaxation solve.  The conformal metric is kept
positive by a local symmetric matrix exponential and determinant normalization
before the CTS solve applies the conformal factor and shift.

The relaxation controls follow the NRPyElliptic hyperbolic-relaxation form
`du/dtau = v - eta*u`, `dv/dtau = c^2 R`.  For uniform meshes
`wavespeed_mode=local_dx` gives a constant relaxation wave speed and
`relax_cfl` is the usual wave CFL.  Large damping also constrains the explicit
RK step through `eta*dtau`; if necessary the solver caps `dtau` using
`damping_stability_limit` and reports the effective value.

AthenaK uses Cartesian meshblocks for these runs.  Domain size, resolution,
boundary conditions, and any refinement are therefore controlled by the usual
Cartesian `<mesh>` and `<meshblock>` inputs.

Key controls:

- `amplitude`: overall conformal-wave amplitude.
- `ell`, `m`: helical angular mode, normally `ell=m=2` or higher.
- `radial_coeff_N`: finite radial-basis coefficients.
- `support_radius`: exact truncation radius of the compact support.
- `bump_steepness`: smooth cutoff strength at the truncation radius.
- `ingoing_sign`: sign of the one-way radial contribution.
- `omega`, `helicity`: rotating wave controls.
- `<z4c>/spatial_order` and `<z4c>/diss`: finite-difference order and
  Kreiss-Oliger dissipation used by both Z4c evolution and the relaxation
  residual.
