# Directory Guide

## Role
Boundary exchange, prolongation, restriction, and flux-correction infrastructure shared by fluid, radiation, particles, and spacetime modules.

## Important Files
- `bvals.hpp`: core types for boundary faces/flags, communication buffers, and the `MeshBoundaryValues*` classes.
- `bvals.cpp`, `bvals_tasks.cpp`: construction and task-facing orchestration.
- `bvals_cc.cpp`, `bvals_fc.cpp`: cell-centered and face-centered send/receive paths.
- `buffs_cc.cpp`, `buffs_fc.cpp`: buffer packing layout details.
- `prolongation.cpp`, `prolong_prims.cpp`, `flux_correct_*.cpp`: AMR prolongation and flux correction.
- `bvals_part.cpp`: particle-specific boundary migration.

## Important Subdirectories
- `physics/`: actual BC kernels by subsystem. Read `physics/AGENTS.md` when changing face behavior rather than transport plumbing.

## Read This Next
- For AMR ownership, also read `src/mesh/AGENTS.md`.
- For task ordering, see `src/tasklist/AGENTS.md`.
