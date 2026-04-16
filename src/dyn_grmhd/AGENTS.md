# Directory Guide

## Role
Matter evolution for GRMHD in dynamical spacetimes, coupled to Z4c and advanced primitive solvers.

## Important Files
- `dyn_grmhd.hpp`: main class hierarchy, task IDs, policy selection, and factory entrypoint.
- `dyn_grmhd.cpp`: construction and task queue wiring.
- `dyn_grmhd_fluxes.cpp`: core flux calculation path.
- `dyn_grmhd_fofc.cpp`: first-order flux correction fallback.
- `dyn_grmhd_util.hpp`: shared helpers used by kernels and task code.

## Important Subdirectories
- `rsolvers/`: LLF/HLLE implementations specialized for dynamical GRMHD.

## Read This Next
- For EOS/c2p behavior, continue into `src/eos/AGENTS.md`.
- For spacetime coupling and NR tasks, also read `src/z4c/AGENTS.md` and `src/tasklist/AGENTS.md`.
