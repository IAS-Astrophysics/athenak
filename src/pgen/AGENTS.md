# Directory Guide

## Role
Problem-generator layer: sets initial conditions, user hooks, and built-in verification problems.

## Important Files
- `pgen.hpp`, `pgen.cpp`: central `ProblemGenerator` API, built-in dispatch, user BC/source/history hooks, and restart handling.
- Top-level `*.cpp` files such as `blast.cpp`, `current_sheet.cpp`, `gr_torus.cpp`, `dynbbh.cpp`, `dyngr_tov.cpp`: larger maintained setup implementations beyond the compact regression fixtures.
- `elliptica_bns.cpp`, `lorene_bns.cpp`, `disk-magnetosphere.cpp`: integrations with external or specialized initial-data workflows.

## Important Subdirectories
- `tests/`: regression-focused built-in generators.
- `gr_analytic/`: analytic GR helper routines.
- `unit_tests/`: very small test-only generators.

## Read This Next
- For runtime input wiring, compare with `inputs/AGENTS.md`.
- For problem finalization or analysis hooks, also inspect `src/outputs/AGENTS.md`.
