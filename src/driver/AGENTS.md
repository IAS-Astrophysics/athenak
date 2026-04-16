# Directory Guide

## Role
Top-level runtime driver that initializes physics, executes task lists, advances time, and triggers outputs.

## Important Files
- `driver.hpp`: authoritative overview of the `Driver` state, explicit integrator bookkeeping, and the STS controller scaffolding.
- `driver.cpp`: implementation of initialization, execution loop, diagnostics, finalization, and driver-owned STS validation/sweep helpers; the current branch allows Hydro and single-fluid MHD STS on both the standard and shearing/orbital paths, and keeps all `ion-neutral` STS runs fenced.
  Regression coverage now spans `tst/test_suite/diffusion/` and `tst/test_suite/sbox/`.

## Read This Next
- For task graph details, read `src/tasklist/AGENTS.md`.
- For mesh-owned timestep reduction and raw STS config parsing, read `src/mesh/AGENTS.md`.
- For the live STS module paths, read `src/hydro/AGENTS.md` and `src/mhd/AGENTS.md`.
- For executable startup and CLI behavior, read `src/main.cpp`.
