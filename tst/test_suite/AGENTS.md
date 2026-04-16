# Directory Guide

## Role
Maintained pytest-based regression suite. CI uses this tree through `tst/run_test_suite.py`.

## Important Files
- `testutils.py`: shared helpers for build, execution, cleanup, logging, and data loading.
- `diffusion/`: Step 7 STS regression coverage, including exact diffusion problems,
  mixed explicit/STS mode tests, runtime fences, and the MPI smoke.
- Area subdirectories (`dyngrmhd/`, `gr/`, `ion-neutral/`, `nr/`, `rad/`, `sbox/`, `sr/`, `z4c/`): physics-specific pytest suites.
- `style/`: pytest wrapper around style enforcement.

## Read This Next
- Add new automated coverage here unless you specifically need compatibility with the older `tst/scripts/` harness.
