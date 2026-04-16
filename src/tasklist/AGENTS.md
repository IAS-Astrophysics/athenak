# Directory Guide

## Role
Dynamic task-graph runtime used to orchestrate solver stages and module dependencies.

## Important Files
- `task_list.hpp`: generic task and task-list implementation; many modules depend on the interfaces defined here.
- `numerical_relativity.hpp`, `numerical_relativity.cpp`: NR-specific queue builder that composes Z4c and dynamical GRMHD tasks with dependency filtering.

## Read This Next
- `MeshBlockPack::tl_map` now carries both the explicit stage lists and the empty STS/parabolic stage shells: `before_parabolic_stagen`, `parabolic_stagen`, and `after_parabolic_stagen`.
- For the code that drives these task lists each timestep, read `src/driver/AGENTS.md`.
- For the place where those named lists are created, read `src/mesh/meshblock_pack.cpp`.
