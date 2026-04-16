# Directory Guide

## Role
Optional diffusive physics layered on top of hydro and MHD.

## Important Files
- `viscosity.hpp/.cpp`: isotropic Navier-Stokes viscosity; now also parses
  `viscosity_integrator`.
- `conduction.hpp/.cpp`: thermal conduction support; now also parses
  `conductivity_integrator`.
- `resistivity.hpp/.cpp`: magnetic resistivity terms; now also parses
  `ohmic_resistivity_integrator`.
- `sts_types.hpp`, `parabolic_process.hpp`, `sts_rkl2.hpp/.cpp`: shared metadata and
  math helpers for planned super-time-stepping infrastructure.
- `current_density.hpp`: helper used by magnetic diffusion/current-derived quantities.

## Read This Next
- These classes are instantiated and registered for STS bookkeeping from `src/hydro/`
  or `src/mhd/`; read those callers before changing interfaces here.
