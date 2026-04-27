# Dynamical Radiation Paper Assets

This directory contains the method paper, compact comparison inputs, and plotting
script for the standalone `dyn_radiation` solver.  The ADM beam input runs with
`angular_fluxes=true`, exercising the cached ADM angular geodesic transport.
The beam inputs write both a 1D table slice and a 2D binary XY slice so the
script can produce line-profile, field-map, and residual figures.

The intended quick comparison sequence from the repository root is:

```sh
./build/src/athena -i dyngr_radiation_paper/inputs/beam_radiation_cks.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/beam_dynrad_cks.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/beam_dynrad_adm_flat.athinput
./build/src/athena -i inputs/tests/dynrad_bh_beam_adm.athinput
./build/src/athena -i inputs/tests/dynrad_z4c_wave_adm.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/lwave_radiation_cks.athinput job/basename=/tmp/paper_lwave_rad
./build/src/athena -i dyngr_radiation_paper/inputs/lwave_dynrad_cks.athinput job/basename=/tmp/paper_lwave_dyn
python3 dyngr_radiation_paper/scripts/plot_comparisons.py
pdflatex -interaction=nonstopmode dyngr_radiation_method.tex
```

The plotting step requires Python with NumPy and Matplotlib.

The compact stress suite for the passive ADM/Z4c radiation path is:

```sh
python3 dyngr_radiation_paper/scripts/run_stress_tests.py
```

It runs serial, two-rank MPI, restart, unsupported-mode rejection, and dynbbh
smoke cases with outputs under `/tmp`.  In sandboxed environments `mpirun` may
need to be run outside the sandbox so it can bind local sockets.

The `dynbbh_beam_particles.athinput` input requires a build configured with
`PROBLEM=dynbbh`.  The dynbbh initializer now uses `pdynrad` only and rejects
the legacy `<radiation>` solver because that path is not consistent with the
ADM background.  Its default keeps radiation-matter coupling off for a clean
beam/particle smoke test; set `dyn_radiation/rad_source=true` to exercise the
ADM absorption/scattering coupling coefficients included in the input.
