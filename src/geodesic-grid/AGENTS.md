# Directory Guide

## Role
Angular discretization and spherical-grid helpers used by radiation transport and some analysis/problem-generator paths.

## Important Files
- `geodesic_grid.hpp/.cpp`: geodesic angular mesh with neighbor topology, solid angles, and edge directions.
- `spherical_grid.hpp/.cpp`: spherical-grid support used by problem generators and analysis.
- `gauss_legendre.hpp/.cpp`: Gauss-Legendre quadrature support.

## Read This Next
- Radiation code depends on this heavily; read `src/radiation/AGENTS.md` next.
