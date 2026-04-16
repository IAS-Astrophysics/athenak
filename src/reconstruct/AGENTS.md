# Directory Guide

## Role
Header-only spatial reconstruction kernels shared by hydro, MHD, and radiation.

## Important Files
- `dc.hpp`: donor-cell / piecewise-constant reconstruction.
- `plm.hpp`: piecewise-linear reconstruction.
- `ppm.hpp`: piecewise-parabolic reconstruction.
- `wenoz.hpp`: higher-order WENO-Z reconstruction.

## Read This Next
- Solver modules select these via enums in `src/athena.hpp`; callers live in `src/hydro/`, `src/mhd/`, and `src/radiation/`.
