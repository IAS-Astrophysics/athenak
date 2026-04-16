# Directory Guide

## Role
Owns the compiled AthenaK executable: core types, input parsing, runtime driver, mesh/AMR infrastructure, physics modules, and outputs.

## Important Files
- `main.cpp`: executable entrypoint. Handles MPI/Kokkos initialization, CLI flags, input/restart loading, and the top-level run lifecycle.
- `athena.hpp`: central type aliases, enums, array wrappers, and execution-space abstractions used almost everywhere.
- `athena_tensor.hpp`: tensor utilities used heavily by relativistic modules.
- `parameter_input.hpp`, `parameter_input.cpp`: input deck parser and command-line overrides.
- `globals.hpp`, `globals.cpp`: process-global runtime state such as MPI rank metadata.
- `CMakeLists.txt`: authoritative compilation inventory for maintained source files.

## Important Subdirectories
- `mesh/`, `bvals/`, `driver/`, `tasklist/`: runtime infrastructure and execution plumbing.
- `hydro/`, `mhd/`, `radiation/`, `ion-neutral/`, `particles/`: major physics modules.
- `dyn_grmhd/`, `z4c/`, `coordinates/`, `eos/`: relativistic and numerical-relativity stack.
- `outputs/`, `utils/`, `units/`, `reconstruct/`, `diffusion/`, `geodesic-grid/`, `shearing_box/`, `srcterms/`: shared support and optional physics.
- `pgen/`: built-in problem generators and initialization hooks.

## Read This Next
- For a feature change, jump directly into the owning subdirectory before reading unrelated modules.
- For runtime behavior changes that span modules, read in this order: `main.cpp`, `driver/AGENTS.md`, `mesh/AGENTS.md`, then the relevant physics directory.
