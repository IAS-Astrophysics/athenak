# Directory Guide

## Role
Pytest coverage for stationary-spacetime GR hydro/MHD problems.

## Important Files
- `test_gr_bondi_mpicpu.py`, `test_gr_monopole_gpu.py`: representative production-like GR checks.
- `test_gr_lwave*.py`, `test_gr_shocktube_cpu.py`: linear-wave and shock-style regressions.

## Read This Next
- For solver internals, inspect `src/hydro/AGENTS.md` or `src/mhd/AGENTS.md` together with `src/coordinates/AGENTS.md`.
