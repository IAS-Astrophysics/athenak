# Directory Guide

## Role
Advanced primitive-solver framework for tabulated and hybrid equations of state, mainly used by dynamical GRMHD.

## Important Files
- `eos_policy_interface.hpp`, `error_policy_interface.hpp`: policy contracts.
- `eos_compose.*`, `eos_hybrid.*`, `piecewise_polytrope.cpp`: concrete EOS backends.
- `eos.hpp`: local umbrella header for the primitive-solver layer.
- `unit_system.*`, `ps_types.hpp`, `geom_math.hpp`, `numtools_root.hpp`, `logs.hpp`: support types, units, root-finding, and diagnostics.
- `idealgas.hpp`: simplest policy implementation in the same framework.

## Read This Next
- For how this plugs into the rest of AthenaK, back up to `../AGENTS.md`.
