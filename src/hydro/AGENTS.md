# Directory Guide

## Role
Core hydrodynamics module for Newtonian, SR, and stationary-GR flows.

## Important Files
- `hydro.hpp`: best overview of hydro-owned state, task IDs, optional extras, and solver selection.
- `hydro.cpp`: construction and task-list assembly, including diffusion-object creation
  and parabolic-process registration, plus Hydro STS register allocation.
- `hydro_fluxes.cpp`: main flux path.
- `hydro_update.cpp`, `hydro_newdt.cpp`: RK update and timestep control; `hydro_newdt.cpp`
  now also owns the shared Hydro timestep refresh used after the final STS post sweep.
- `hydro_tasks.cpp`: glue between the generic task system and hydro routines, including
  the Hydro-owned parabolic task-list enrollment plus the sweep-aware shearing-box/
  orbital-advection wrappers used by the STS loop. The explicit Hydro flux path now also
  owns the face-flux scratch initialization before explicit diffusion is appended, which
  is required for exact thermal-conduction benchmarks because conduction writes only the
  energy-flux slot.
- `hydro_sts.cpp`: Hydro STS helper logic, selected-diffusion split, RKL2 stage update,
  and the post-sweep timestep refresh hook.
- `hydro_fofc.cpp`: first-order fallback when higher-order states become invalid.

## Important Subdirectories
- `rsolvers/`: concrete Riemann solvers.

## Read This Next
- For EOS behavior, read `src/eos/AGENTS.md`.
- For boundary exchange and AMR communication, read `src/bvals/AGENTS.md`.
- For driver-owned STS validation and sweep sequencing, read `src/driver/AGENTS.md`.
- For shearing-box/orbital-advection boundary semantics, read `src/shearing_box/AGENTS.md`.
- For the exact Hydro STS fixtures and the static-refinement smoke, read
  `inputs/tests/AGENTS.md`, `tst/test_suite/diffusion/AGENTS.md`, and
  `tst/test_suite/sbox/AGENTS.md`.
