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

This is the free-data construction and Z4c handoff, not a full elliptic XCTS
solve.  The conformal metric is kept positive by a local trace-free exponential
approximation and determinant normalization, then converted through the existing
ADM-to-Z4c path.  A later XCTS solve should reuse the same radial/angular seed
and replace the local handoff with an elliptic solve for the conformal factor,
lapse, and shift.

AthenaK currently uses Cartesian meshblocks.  The
`<problem_gw_collapse_domain>` block therefore approximates a filled sphere
surrounded by spherical shells through radial AMR bands on the Cartesian mesh.
It is not a true filled-ball plus spherical-shell multipatch topology.

Key controls:

- `amplitude`: overall conformal-wave amplitude.
- `ell`, `m`: helical angular mode, normally `ell=m=2` or higher.
- `metric_coeff_N`, `momentum_coeff_N`: finite radial-basis coefficients.
- `support_radius`: exact truncation radius of the compact support.
- `radial_center`, `radial_width`: Gaussian localization within the support.
- `bump_steepness`: smooth cutoff strength at the truncation radius.
- `ingoing_weight`: radial one-way contribution to the extrinsic curvature.
- `omega`, `rotation_weight`, `helicity`: rotating wave controls.
- `<problem_gw_collapse_domain>`: filled central AMR region plus shell-band
  levels.
