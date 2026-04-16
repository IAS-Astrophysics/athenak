# Directory Guide

## Role
Example input decks for dynamical-spacetime GRMHD and matter-coupled numerical relativity runs.

## Important Files
- `mag_tov.athinput`, `whisky_tov.athinput`: TOV star setups used for dynamical GR matter evolution.
- `sgrid_bns_amr.athinput`: binary neutron star input with AMR and external initial-data coupling.
- `sod.athinput`: compact shock-tube style setup for fast solver iteration.

## Read This Next
- For solver behavior, pair these decks with `src/dyn_grmhd/AGENTS.md`.
- For problem-generator wiring, check `src/pgen/AGENTS.md`.
