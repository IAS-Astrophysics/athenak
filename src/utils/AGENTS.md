# Directory Guide

## Role
Small cross-cutting utilities that do not belong to a single physics module.

## Important Files
- `utils.hpp`: umbrella declarations for the most commonly used helpers.
- `show_config.cpp`, `change_rundir.cpp`: executable/runtime convenience helpers used from `main.cpp`.
- `derived_vars.cpp`, `cart_grid.cpp`, `spherical_surface.cpp`, `current.hpp`: analysis/output support routines.
- `lagrange_interpolator.*`, `finite_diff.hpp`, `chebyshev.hpp`, `legendre_roots.hpp`: numerical helper code.
- `random.hpp`, `tr_table.*`: assorted utilities used by specialized physics paths.

## Important Subdirectories
- `tov/`: neutron-star/TOV helper routines.

## Read This Next
- For output-facing helpers, also inspect `src/outputs/AGENTS.md`.
