# STS Conduction Follow-up

## Benchmark Correctness

- The earlier Hydro conduction benchmark discrepancy was not evidence that STS itself was
  inaccurate.
- The benchmark exposed a Hydro explicit-path bug instead: `Hydro::Fluxes()` must clear
  the face-flux scratch before appending explicit diffusion.
- That matters for conduction because the conduction operator writes only the
  total-energy-flux slot, so stale face-flux components can corrupt the exact conduction
  profile if the scratch array is not explicitly initialized.
- With that fix in place, both explicit and STS Hydro conduction recover clean
  second-order convergence on the exact Gaussian diffusion problem and land in the same
  small-error regime as the viscosity and resistivity exact tests.

## Relation To The Landau-Fluid/CGL Question

- Jim's point is still relevant: AthenaK's current conduction operator is coupled through
  the total energy flux, which makes both exact benchmarking and future extensions more
  delicate than viscosity or resistivity.
- Steve's Landau-fluid/CGL concern is a deeper variable-choice and operator-splitting
  issue. The preferred LF heat-flux variable is not the same one AthenaK's current
  hydro-flux infrastructure naturally updates.
- Those are separate issues. The benchmark bug above should not be used as evidence for
  or against whether STS is acceptable for future Landau-fluid heat-flux work.
