# Directory Guide

## Role
Defines the repository's GitHub Actions automation.

## Important Files
- `main.yml`: single CI workflow. Runs style checks first, then CPU, MPI CPU, and GPU regression jobs on self-hosted runners with the `kokkos` submodule checked out.

## Read This Next
- For test failures in CI, read `tst/AGENTS.md` after this file.
- For runner-specific build flags, compare the job commands here with `tst/run_test_suite.py`.
