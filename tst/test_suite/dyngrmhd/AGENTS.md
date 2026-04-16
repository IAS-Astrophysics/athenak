# Directory Guide

## Role
Pytest coverage for dynamical-spacetime GRMHD.

## Important Files
- `test_dyngrmhd_lwave*.py`: linear-wave regressions across CPU, MPI CPU, and GPU targets.
- `test_dyngrmhd_shocktube_cpu.py`, `test_dyngrmhd_nqt_shocktube_cpu.py`, `test_dyngrmhd_tab_shocktube_cpu.py`: shock-tube and EOS-path coverage for dynamical GRMHD.

## Read This Next
- These tests usually pair `tst/inputs/` fixtures with `src/dyn_grmhd/AGENTS.md`.
