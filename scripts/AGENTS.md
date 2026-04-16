# Directory Guide

## Role
Holds ad hoc operator scripts that sit outside the compiled code and test harnesses.

## Important Files
- `run_slurm.sh`: example SLURM submission script that shows the expected runtime invocation pattern for the `athena` executable on a cluster.

## Read This Next
- For reusable testing or build automation, prefer `tst/` over adding logic here.
- For production launch parameters, pair this file with a nearby `inputs/*.athinput` example.
