# Directory Guide

## Role
Magnetohydrodynamics module, including constrained transport and optional diffusion/source physics.

## Important Files
- `mhd.hpp`: primary map of MHD state, task IDs, boundary objects, and optional extras.
- `mhd.cpp`: construction and task assembly, including diffusion-object creation and
  parabolic-process registration plus the MHD STS mode flags/history allocation.
- `mhd_fluxes.cpp`: finite-volume flux calculations.
- `mhd_ct.cpp`, `mhd_corner_e.cpp`: constrained transport and corner electric fields.
- `mhd_update.cpp`, `mhd_newdt.cpp`: RK updates and timestep selection.
- `mhd_sts.cpp`: MHD-owned STS helpers, `RKL2` updates for conserved and face-centered
  state, diffusion/EMF selection helpers, and the post-sweep timestep refresh hook.
- `mhd_tasks.cpp`, `mhd_fofc.cpp`: task glue and low-order fallback; `mhd_tasks.cpp`
  now also owns the sweep-aware shearing-box/orbital wrappers used by the MHD STS loop.

## Important Subdirectories
- `rsolvers/`: concrete Riemann solvers.

## Read This Next
- For magnetic boundary handling, read `src/bvals/AGENTS.md`.
- For resistive/viscous extensions, read `src/diffusion/AGENTS.md`.
- For runtime STS activation and the remaining scope fence (`ion-neutral`), read
  `src/driver/AGENTS.md`.
- For shearing-box/orbital-advection boundary semantics, read `src/shearing_box/AGENTS.md`.
- For the exact resistive STS fixture and mixed explicit/STS regression, read
  `inputs/tests/AGENTS.md`, `tst/test_suite/diffusion/AGENTS.md`, and
  `tst/test_suite/sbox/AGENTS.md`.
