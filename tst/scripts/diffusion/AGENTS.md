# Directory Guide

## Role
Standalone benchmark scripts for the exact 1D STS diffusion package. This is separate from
the maintained pytest regression suite: use this directory when the task is to generate
benchmark data, figures, or a short report rather than to gate correctness in CI.

## Important Files
- `benchmark_sts_diffusion.py`: builds or reuses the CPU AthenaK binary, runs the explicit
  and STS diffusion benchmark matrix, writes raw and summarized CSV data under
  `doc/data/sts_diffusion/`, emits the small TeX tables used by the benchmark note, and
  now refreshes `tst/build` automatically when the benchmark binary is older than the
  relevant source or input files.

## Read This Next
- For the exact problem-generator side of the benchmark, read `src/pgen/tests/AGENTS.md`.
- For the plotting step, read `vis/python/AGENTS.md`.
- For the write-up, read `doc/AGENTS.md`.
