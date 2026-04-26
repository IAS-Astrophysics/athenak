# Dynamical Radiation Paper Assets

This directory contains the method note, compact comparison inputs, and plotting
script for the standalone `dyn_radiation` solver.

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

The `dynbbh_beam_particles.athinput` input requires a build configured with
`PROBLEM=dynbbh`.
