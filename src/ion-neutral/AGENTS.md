# Directory Guide

## Role
Two-fluid ion-neutral coupling module layered on top of the hydro/MHD infrastructure.

## Important Files
- `ion-neutral.hpp`: task IDs and coupling coefficients for drag, ionization, and recombination.
- `ion-neutral.cpp`: setup and integration wiring.
- `ion-neutral_tasks.cpp`: task-list assembly and stage execution.

## Read This Next
- This module leans on the generic task runtime and fluid solvers, so cross-check `src/tasklist/AGENTS.md` and `src/hydro/AGENTS.md`.
- STS is currently fenced for all `ion-neutral` runs because this module assembles its
  own coupled Hydro/MHD task graph and does not yet populate the parabolic task lists;
  check `src/driver/AGENTS.md` and `doc/sts_implementation_guide.md` before widening
  that scope.
