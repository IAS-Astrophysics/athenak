# Dynamical Radiation Paper Assets

This directory contains the method paper, compact comparison inputs, and plotting
script for the standalone `dyn_radiation` solver.  The beam inputs write both a
1D table slice and a 2D binary XY slice so the script can produce line-profile,
field-map, and residual figures.

The intended quick comparison sequence from the repository root is:

```sh
./build/src/athena -i dyngr_radiation_paper/inputs/beam_radiation_cks.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/beam_dynrad_cks.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/beam_dynrad_adm_flat.athinput
./build/src/athena -i dyngr_radiation_paper/inputs/lwave_radiation_cks.athinput job/basename=/tmp/paper_lwave_rad
./build/src/athena -i dyngr_radiation_paper/inputs/lwave_dynrad_cks.athinput job/basename=/tmp/paper_lwave_dyn
python3 dyngr_radiation_paper/scripts/plot_comparisons.py
pdflatex -interaction=nonstopmode dyngr_radiation_method.tex
```

The plotting step requires Python with NumPy and Matplotlib.

The `dynbbh_beam_particles.athinput` input requires a build configured with
`PROBLEM=dynbbh`.
