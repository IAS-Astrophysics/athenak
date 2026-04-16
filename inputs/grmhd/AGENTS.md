# Directory Guide

## Role
Stationary-spacetime GRMHD examples used for production-like torus runs and regression coverage.

## Important Files
- `gr_fm_torus_sane_8_4.athinput`, `gr_fm_torus_mad_8_4.athinput`: torus accretion setups with different magnetic states.
- `gr_chakrabarti_torus_sane_8_4.athinput`: alternative torus initial condition.
- `blast_grmhd_amr.athinput`, `mub1-gr.athinput`: smaller tests for GRMHD shock/blast behavior.

## Read This Next
- For GRMHD fluxes and constrained transport, see `src/mhd/AGENTS.md`.
- For output/analysis expectations, compare with `vis/python/AGENTS.md`.
