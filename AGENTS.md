# Repository Guide

## What This Repository Does
AthenaK is a block-structured AMR astrophysical simulation code written in C++17 on top of Kokkos. It targets performance-portable fluid, MHD, radiation, particle, and numerical-relativity workloads, including Newtonian, special-relativistic, general-relativistic, and dynamical-spacetime regimes.

The codebase is organized around a small executable core in `src/`, runtime-configurable problem decks in `inputs/`, automated regression coverage in `tst/`, lightweight design notes in `doc/`, and Python post-processing in `vis/python/`. Most workflow changes involve touching one owning solver directory plus the matching inputs/tests rather than editing the whole tree.

## Architecture At A Glance
- `src/main.cpp` + `src/driver/`: process startup, MPI/Kokkos initialization, time integration, and output scheduling.
- `src/mesh/` + `src/bvals/` + `src/tasklist/`: mesh/AMR layout, boundary exchange, and dynamic task execution.
- `src/hydro/`, `src/mhd/`, `src/radiation/`, `src/particles/`, `src/ion-neutral/`: major physics modules.
- `src/coordinates/`, `src/eos/`, `src/diffusion/`, `src/reconstruct/`, `src/shearing_box/`, `src/srcterms/`, `src/units/`, `src/utils/`: shared physics support.
- `src/z4c/` + `src/dyn_grmhd/`: numerical relativity and matter evolution in dynamical spacetimes.
- `src/pgen/` + `inputs/`: initial conditions, user hooks, and runtime problem configuration.
- `tst/`: pytest-based regression suite plus older script-based tests.
- `vis/python/`: output readers, converters, and plotting scripts.

## Where To Work
| Task | Start Here | Notes |
| --- | --- | --- |
| Change executable startup, CLI flags, or run lifecycle | `src/AGENTS.md` | Read `src/main.cpp`, then `src/driver/AGENTS.md`. |
| Modify mesh, AMR, or boundary communication | `src/mesh/AGENTS.md` | `src/bvals/AGENTS.md` is usually the next stop. |
| Change hydro or MHD numerics | `src/hydro/AGENTS.md` or `src/mhd/AGENTS.md` | Solver choices live in each module's `rsolvers/` child. |
| Work on GR, Z4c, or dynamical-spacetime coupling | `src/z4c/AGENTS.md` | Pair with `src/dyn_grmhd/AGENTS.md` for matter-coupled runs. |
| Add or tune a problem setup | `inputs/AGENTS.md` | Matching setup code usually lives in `src/pgen/AGENTS.md`. |
| Adjust outputs or derived diagnostics | `src/outputs/AGENTS.md` | Python readers live in `vis/python/AGENTS.md`. |
| Add regression coverage or debug CI | `tst/AGENTS.md` | CI workflow itself lives in `.github/workflows/AGENTS.md`. |
| Understand a planned architecture change before coding | `doc/AGENTS.md` | Pair the guide with the owning `src/` directories before implementing it. |
| Update plotting or analysis scripts | `vis/python/AGENTS.md` | Tests often import `vis/python/athena_read.py` directly. |

## Top-Level Map
- `.github/`: GitHub Actions workflow definitions.
- `doc/`: design notes and implementation guides for planned or cross-cutting changes.
- `inputs/`: maintained `.athinput` example and regression decks by physics area.
- `scripts/`: one-off operator scripts; not the main testing harness.
- `src/`: compiled code, including runtime infrastructure, physics modules, and outputs.
- `tst/`: automated testing, with both pytest and legacy script-based harnesses.
- `vis/`: Python-side readers, converters, and visualization tools.

## Skipped Or Generated Areas
- `kokkos/`: external git submodule; intentionally not documented as repository-owned source.
- `.git/`: VCS metadata, intentionally omitted.
