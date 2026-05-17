# Z4c Initial-Data Import Utilities

`convert_speck_adm_cart_to_id_solve.cpp` converts a uniform SpECK ADM
Cartesian HDF5 dump into the HDF5 layout consumed by the `z4c_id_solve`
problem generator.

`run_id_solve_collapse_convergence.py` runs a sequence of converted
critical-collapse imports and reports the volume-weighted ADM/Z4c constraint
RMS values written by the id-solve reader.

The converter intentionally rejects Chebyshev-node SpECK dumps. Convert the
SpECK data to a uniform AthenaK-centered Cartesian dump first, then feed that
uniform dump into this converter.
