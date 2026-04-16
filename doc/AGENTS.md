# Directory Guide

## Role
Design notes and implementation guides for AthenaK changes. The STS guide now reflects
the fully landed seven-step implementation, the post-v1 Hydro and MHD shearing-box/
orbital-advection extensions, and the remaining unsupported runtime scopes.

## Important Files
- `sts_implementation_guide.md`: decision-complete STS design plus the current landed
  Hydro/MHD status, the diffusion and shearing-box regression suites, and the still-fenced
  `ion-neutral` scope.
- `sts_diffusion_benchmark.tex`: concise colleague-facing benchmark note comparing
  explicit and STS accuracy/cost for the exact 1D viscosity, conduction, and resistivity
  problems. It consumes generated data from `doc/data/sts_diffusion/` and figures from
  `doc/figures/sts_diffusion/`.
- `sts_conduction_followup.md`: short internal note separating the Hydro conduction
  benchmark-audit fix from the deeper future Landau-fluid/CGL heat-flux design question.

## Read This Next
- Pair any guide here with the owning runtime directories under `src/`, especially `src/driver/AGENTS.md`, `src/mesh/AGENTS.md`, `src/diffusion/AGENTS.md`, `src/hydro/AGENTS.md`, and `src/mhd/AGENTS.md`.
- For the scripts that regenerate the benchmark artifacts, read `tst/scripts/diffusion/AGENTS.md`
  and `vis/python/AGENTS.md`.
