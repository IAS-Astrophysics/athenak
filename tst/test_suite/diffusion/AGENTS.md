# Directory Guide

## Role
STS-specific regression coverage for exact diffusion problems, mixed explicit/STS mode
selection, multilevel smoke runs, and runtime-fence behavior.

## Important Files
- `test_sts_diffusion_cpu.py`: CPU exact-solution regressions, mixed-mode timestep-budget
  coverage, SMR smoke coverage, and full-run fence checks for the remaining global and
  `ion-neutral` STS failures. The Hydro conduction checks now enforce clean exact
  convergence for both explicit and STS runs in the corrected `1e-12` error regime.
- `test_sts_diffusion_mpicpu.py`: narrow 4-rank MPI smoke for resistive STS.

## Read This Next
- Pair these tests with `inputs/tests/AGENTS.md` and `src/pgen/tests/AGENTS.md` so the
  exact fixtures and the diffusion problem generator stay aligned.
