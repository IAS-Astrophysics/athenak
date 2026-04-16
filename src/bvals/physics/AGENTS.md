# Directory Guide

## Role
Contains the actual physical boundary-condition kernels applied after the generic communication/prolongation layer has moved data into place.

## Important Files
- `hydro_bcs.cpp`: hydro primitive/conserved boundary kernels.
- `bfield_bcs.cpp`: magnetic-field boundary handling.
- `radiation_bcs.cpp`: radiation intensity boundary kernels.
- `z4c_bcs.cpp`: spacetime-boundary routines for Z4c variables.

## Read This Next
- For buffer setup and MPI exchange, back up to `../AGENTS.md`.
