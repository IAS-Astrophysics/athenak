# Directory Guide

## Role
Regression and style testing infrastructure for AthenaK. This tree contains both the current pytest-based suite and an older script-based harness.

## Important Files
- `run_test_suite.py`: current entrypoint used by CI. Selects style, CPU, MPI CPU, GPU, or a single test and routes through pytest.
- `run_tests.py`: older regression entrypoint that imports `tst/scripts/` modules with `run()`/`analyze()` functions.
- `inputs/`: test-specific input decks and tabulated EOS fixtures.

## Important Subdirectories
- `test_suite/`: maintained pytest suites and shared helpers.
- `scripts/`: legacy regression modules and utilities.

## Read This Next
- If CI is failing, start with `run_test_suite.py` and `test_suite/AGENTS.md`.
- If an older developer script or wiki page mentions `run_tests.py`, use `scripts/AGENTS.md`.
