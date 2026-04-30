# Dynamical Radiation Solver Notes

`dyn_radiation` is a passive-radiation transport module for dynamical or
analytic ADM backgrounds.  Radiation propagates on the supplied geometry and can
exchange energy-momentum with the fluid, but it does not currently add radiation
stress-energy back to Z4c.

## Current Contract

- `dyn_radiation/geometry=cks` preserves the legacy stationary-CKS signed
  radiation normalization for regression tests. In the `(-+++)` convention
  `-k_0` is the positive Killing energy, while the legacy conserved variable
  uses the signed product `k^0 k_0 I`.
- `dyn_radiation/geometry=adm` transports the densitized Eulerian intensity
  using cached ADM lapse, shift, spatial metric, tetrad, face metric factors,
  and angular geodesic drift.
- ADM data may come from analytic `<adm dynamic=false>` backgrounds or from Z4c.
  The current paper tests use analytic ADM backgrounds for flat, single-Kerr,
  and superposed orbiting Kerr-Schild cases.
- The ADM geometric source is applied with a closed-form exponential stage
  factor, avoiding the old forward-Euler positivity loss.
- Negative angular bins are repaired by local angular redistribution that
  preserves the angular zeroth moment in the solver's conserved variable when
  possible. This is not a momentum-conserving angular remap.
- Radiation-matter coupling iterates temperature-dependent opacities with
  `source_max_iter` and `source_tolerance`; ADM coupling includes the
  `alpha * D * dt` coordinate-to-comoving optical-depth factor.
  `fixed_fluid=true` defaults to `affect_fluid=false`.

## Particle Diagnostics

The dynbbh paper diagnostic initializes null geodesic particles at the beam
source and advances them with the ADM null pusher.  In MPI, only the rank that
owns the beam source retains these particles.  The pgen then recomputes particle
counts and tags so tracked particle output follows the real source particles,
not placeholder particles from ranks that do not contain the source.

The particle timestep is capped by the ADM coordinate light speeds,
`|beta^q| + alpha sqrt(gamma^{qq})`, so a null diagnostic particle cannot cross
more cells than the boundary exchange expects in shifted or small-lapse
backgrounds.
