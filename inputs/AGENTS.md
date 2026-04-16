# Directory Guide

## Role
Repository-wide catalog of runtime input decks. These files define problem parameters, mesh/output blocks, and solver options without touching compiled code.

## Important Subdirectories
- `hydro/`, `mhd/`, `srhydro/`, `srmhd/`: fluid and MHD examples across Newtonian and relativistic regimes.
- `grhydro/`, `grmhd/`, `dyngr/`: stationary-GR and dynamical-spacetime relativistic problems.
- `radiation/`, `shearing_box/`, `ion-neutral/`, `particles/`: feature-specific examples.
- `tests/`, `unit_tests/`: compact fixtures used by automated testing.
- `z4c/`: numerical-relativity-only inputs.

## Read This Next
- To change a runtime setup, start from the closest existing `.athinput` rather than editing from scratch.
- For how these decks are consumed, read `src/parameter_input.hpp`, then the corresponding solver directory under `src/`.
