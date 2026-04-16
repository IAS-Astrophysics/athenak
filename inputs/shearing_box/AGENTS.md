# Directory Guide

## Role
Local shearing-box and orbital-advection example problems.

## Important Files
- `epicycle.athinput`: simplest orbital/shear sanity check.
- `hydro_*shwave.athinput`, `mhd_compress_shwave.athinput`: shearing-wave regression inputs.
- `hydro_orb_adv.athinput`, `mhd_orb_adv.athinput`: orbital advection checks.
- `mri2d.athinput`, `mri3d_*.athinput`: MRI setups, including stratified and unstratified variants.

## Read This Next
- For shearing boundary implementation, read `src/shearing_box/AGENTS.md`.
- For MHD variants, also check `src/mhd/AGENTS.md`.
- The tracked `hydro_orb_adv.athinput` and `mhd_orb_adv.athinput` templates are now also
  used by the Hydro/MHD STS smoke tests in `tst/test_suite/sbox/`, with pytest injecting
  the default-build problem-generator choices needed for those smokes.
