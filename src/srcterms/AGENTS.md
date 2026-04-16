# Directory Guide

## Role
Shared physical source terms and turbulence-driving logic for fluid modules.

## Important Files
- `srcterms.hpp`, `srcterms.cpp`, `srcterms_newdt.cpp`: generic source-term flags, application, and timestep constraints.
- `turb_driver.hpp/.cpp`: turbulence forcing implementation.
- `ismcooling.hpp`: cooling helper declarations.

## Read This Next
- For callers, read `src/hydro/AGENTS.md`, `src/mhd/AGENTS.md`, or `src/radiation/AGENTS.md`.
