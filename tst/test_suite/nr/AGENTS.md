# Directory Guide

## Role
Pytest coverage for numerical-relativity evolutions beyond the narrower Z4c-only smoke tests.

## Important Files
- `test_nr_isolwave1d_cpu.py`, `test_nr_lwave*.py`: isolated and coupled line-wave regressions.
- `test_nr_rj2a_cpu.py`, `test_nr_sod_cpu.py`: more problem-specific NR validation cases.

## Read This Next
- Cross-check `src/z4c/AGENTS.md` and, when matter is present, `src/dyn_grmhd/AGENTS.md`.
