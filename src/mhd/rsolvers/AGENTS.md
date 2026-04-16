# Directory Guide

## Role
Header-only Riemann solvers for MHD across Newtonian, SR, and GR formulations.

## Important Files
- `advect_mhd.hpp`: advection-style baseline.
- `llf_*.hpp`, `hlle_*.hpp`, `hlld_mhd.hpp`: approximate MHD solvers by regime.
- `llf_mhd_singlestate.hpp`: specialized simplified-state path.

## Read This Next
- For constrained transport and task integration, return to `../AGENTS.md`.
