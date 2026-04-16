# Directory Guide

## Role
Small, purpose-built input decks used by automated regression tests and convergence checks.

## Important Files
- `advect_*`, `linear_wave_*`, `cpaw3d*`: transport and wave convergence inputs.
- `bondi.athinput`, `hohlraum_1d.athinput`, `z4c_linear_wave.athinput`: representative GR, radiation, and NR test fixtures.
- `mub*`, `mb2.athinput`: relativistic MHD and hydro reference cases.
- `viscosity.athinput`, `sts_conduction.athinput`, `sts_resistivity.athinput`:
  exact STS diffusion fixtures for Hydro viscosity, Hydro conduction, and MHD
  resistivity.
- `sts_mhd_mixed_modes.athinput`, `sts_viscosity_smr.athinput`: focused mixed-mode and
  multilevel STS regression inputs.

## Read This Next
- When a pytest or legacy regression names an input under `tests/`, start here before looking at broader example decks.
- Cross-reference `tst/AGENTS.md` for the harness that invokes these files.
- For the exact-solution side of the STS fixtures, also read `src/pgen/tests/AGENTS.md`.
