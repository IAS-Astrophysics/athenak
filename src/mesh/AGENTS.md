# Directory Guide

## Role
Mesh, meshblock, AMR tree, and load-balancing ownership. If a change affects domain decomposition or refinement, start here.

## Important Files
- `mesh.hpp`, `mesh.cpp`: top-level mesh object, runtime sizes, meshblock packs, setup
  flow, and the global timestep reduction including STS budget bookkeeping; Step 5 now
  switches `dt` to the split cycle candidate whenever STS is enabled.
- `meshblock.hpp/.cpp`, `meshblock_pack.hpp/.cpp`: local patch storage and pack-level organization for accelerator-friendly execution.
- `meshblock_tree.hpp/.cpp`, `build_tree.cpp`: logical tree representation of AMR/SMR
  structure, plus mesh-owned parsing of timestep controls such as `cfl_number` and raw
  STS config values.
- `mesh_refinement.hpp/.cpp`, `refinement_criteria.cpp`: refinement policy and criteria hooks.
- `load_balance.cpp`: repartitioning/load-balance logic.

## Read This Next
- Driver now owns the global STS activation checks and controller state, so read
  `src/driver/AGENTS.md` for the runtime fence and sweep orchestration.
- Hydro and MHD now refresh their local timestep estimates after the final STS post
  sweep, so read `src/hydro/AGENTS.md` and `src/mhd/AGENTS.md` when a timestep question
  depends on live STS state.
- The Step 7 multilevel STS smoke uses static refinement from
  `inputs/tests/sts_viscosity_smr.athinput`; check `tst/test_suite/diffusion/AGENTS.md`
  before changing STS-related prolongation/restriction behavior.
- For boundary communication, also read `src/bvals/AGENTS.md`.
- For problem-specific refinement hooks, check `src/pgen/AGENTS.md`.
