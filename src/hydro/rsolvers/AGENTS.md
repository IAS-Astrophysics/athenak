# Directory Guide

## Role
Header-only Riemann solver implementations for hydro across Newtonian, SR, and GR regimes.

## Important Files
- `advect_hyd.hpp`: simplest advection-only flux path.
- `llf_*.hpp`, `hlle_*.hpp`, `hllc_*.hpp`, `roe_hyd.hpp`: approximate solvers grouped by regime.
- `llf_hyd_singlestate.hpp`: specialized fast path for reduced state handling.

## Read This Next
- Solver selection happens in `../hydro.hpp`; return there before changing interfaces.
