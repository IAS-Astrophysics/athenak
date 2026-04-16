# Directory Guide

## Role
Radiation transport module and radiation-fluid coupling.

## Important Files
- `radiation.hpp`: best overview of intensity storage, angular mesh/tetrad state, source/coupling flags, and task IDs.
- `radiation.cpp`: construction and setup.
- `radiation_fluxes.cpp`: transport flux calculation.
- `radiation_source.cpp`: source and coupling terms.
- `radiation_tetrad.hpp/.cpp`: orthonormal tetrad machinery for relativistic radiation.
- `radiation_newdt.cpp`, `radiation_tasks.cpp`, `radiation_update.cpp`: timestep control, task glue, and RK update.
- `radiation_opacities.hpp`: opacity helpers and constants.

## Read This Next
- For angular discretization, read `src/geodesic-grid/AGENTS.md`.
- For output/analysis of radiation fields, read `src/outputs/AGENTS.md` and `vis/python/AGENTS.md`.
