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

The relaxation controls default to the paper-style hyperbolic-relaxation form
`du/dtau = v`, `dv/dtau = c^2 R - eta*v`, where the auxiliary variable is the
pseudo-time velocity and its L2 norm is a direct steady-state diagnostic.  The
NRPyElliptic form remains available with `<id_solve>/damping_form=nrpy`.  For
uniform meshes
the default smooth-box wave speed is constant and the default damping is
estimated from the high-frequency characteristic scale of the selected Z4c
finite-difference operator.  On refined meshes this uses the same smooth speed
envelope that avoids discontinuous relaxation speeds across refinement
boundaries, so the default tracks the effective maximum `c/dx` rather than the
box-crossing time.  `relax_cfl` is the usual wave CFL.  Large damping also
constrains the explicit RK step through `eta*dtau`; if necessary the solver caps
`dtau` using `damping_stability_limit` and reports the effective value.
The optional `eta_schedule=adaptive_rate` controller starts from the box-scale
damping estimate, monitors the residual log-decay rate, and relaxes `eta`
toward a high damping cap when convergence slows relative to the best decay
rate seen so far.  If `eta_final` is not specified, the adaptive-rate cap is
half the explicit damping stability ceiling; set `eta_final` to override it.
`eta_schedule=adaptive_curvature` is also available as a simpler
curvature-trigger prototype.

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
