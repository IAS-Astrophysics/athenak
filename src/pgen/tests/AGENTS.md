# Directory Guide

## Role
Built-in problem generators used mainly by the regression suite and method verification problems.

## Important Files
- `linear_wave.cpp`, `shock_tube.cpp`, `cpaw.cpp`, `advection.cpp`: compact regression-style initial conditions.
- `diffusion.cpp`, `rad_beam.cpp`, `rad_linear_wave.cpp`: feature-targeted generators for diffusion and radiation.
  `diffusion.cpp` now carries the Step 7 exact STS modes selected by
  `problem/diffusion_test = hydro_viscosity|hydro_conduction|mhd_resistivity`.
- `gr_bondi.cpp`, `gr_monopole.cpp`, `z4c_*`, `mri3d.cpp`, `orszag_tang.cpp`: higher-level tests spanning GR, NR, and MHD.

## Read This Next
- Match a problem generator here with the corresponding deck in `inputs/tests/`.
- For STS validation, pair `diffusion.cpp` with `inputs/tests/viscosity.athinput`,
  `sts_conduction.athinput`, and `sts_resistivity.athinput`, then read
  `tst/test_suite/diffusion/AGENTS.md`.
