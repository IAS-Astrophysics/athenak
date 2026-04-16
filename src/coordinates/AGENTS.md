# Directory Guide

## Role
Coordinate geometry, metric support, and horizon/excision helpers used by relativistic modules.

## Important Files
- `coordinates.hpp`, `coordinates.cpp`: lightweight coordinate object passed into kernels; owns metric flags and source-term helpers.
- `adm.hpp`, `adm.cpp`: ADM variable support used by numerical relativity and dynamical spacetime coupling.
- `cartesian_ks.hpp`: inline Cartesian Kerr-Schild metric helpers.
- `cell_locations.hpp`: cell/face position utilities.
- `excision.cpp`: runtime updates for excision masks.

## Read This Next
- For stationary-GR hydro/MHD behavior, pair this directory with `src/hydro/AGENTS.md` or `src/mhd/AGENTS.md`.
- For dynamical spacetime evolution, continue into `src/z4c/AGENTS.md`.
