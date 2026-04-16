# Directory Guide

## Role
Newtonian hydrodynamics example inputs spanning common verification and benchmark problems.

## Important Files
- `sod.athinput`, `shu_osher.athinput`, `lw_implode.athinput`: compact 1D/2D shock and Riemann-style tests.
- `blast_hydro*.athinput`, `shock_cloud.athinput`, `rt2d.athinput`, `kh2d-*.athinput`: multidimensional instability and blast problems.
- `slotted_cyl.athinput`, `turb.athinput`, `viscosity.athinput`: advection, turbulence-driving, and diffusion-related checks.

## Read This Next
- For hydro task flow, read `src/hydro/AGENTS.md`.
- For viscosity or source-term parameters, also check `src/diffusion/AGENTS.md` and `src/srcterms/AGENTS.md`.
