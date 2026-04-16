# Directory Guide

## Role
Equation-of-state layer and conserved-to-primitive conversion for hydro, MHD, and relativistic variants.

## Important Files
- `eos.hpp`, `eos.cpp`: base EOS API and common conversions.
- `ideal_*.cpp`: ideal-gas implementations across Newtonian, SR, and GR hydro/MHD.
- `isothermal_*.cpp`: isothermal variants.
- `noop_dyngrmhd.cpp`: placeholder wiring for cases where the dynamical GRMHD path owns c2p.
- `ideal_c2p_*.hpp`: inline helpers used by the standard solvers.

## Important Subdirectories
- `primitive-solver/`: policy-based advanced EOS stack for dynamical GRMHD and tabulated matter.

## Read This Next
- For callers, inspect `src/hydro/AGENTS.md`, `src/mhd/AGENTS.md`, or `src/dyn_grmhd/AGENTS.md`.
