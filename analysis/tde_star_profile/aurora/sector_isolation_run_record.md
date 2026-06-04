# TOV Symmetry Sector Isolation Run Record

## Classification

The y-reflection asymmetry was not reproduced in either isolated sector tested here:

- Stage A, GRMHD fluid on frozen analytic Schwarzschild ADM, stayed symmetric for the uniform 12-rank, SMR 12-rank, and SMR 8-rank runs.
- Stage B, Z4c spacetime evolution with fluid frozen and `Tmunu` explicitly zeroed, stayed symmetric for the uniform 12-rank and SMR 12-rank runs. The SMR 8-rank follow-up was not run because the SMR 12-rank case did not break.

This points away from a standalone fluid-only or spacetime-only origin in these reduced tests. The next most likely class is the coupled path: evolved Z4c/ADM data and matter feedback feeding the GRMHD reconstruction/flux calculation, especially the previously localized stage-2 x3 flux path.

## Stage A: Fluid On Frozen Schwarzschild ADM

| Case | Job | Ranks | Deck | Result |
| --- | --- | ---: | --- | --- |
| `stageA_tov_frozen_ks_uniform_12r` | `8522541` | 12 | `inputs/tde/aurora/tov_frozen_ks_n3_schwarzschild_sym_uniform_small_aurora.athinput` | Symmetric; final density relative L2 `2.31e-12`, relative Linf `6.95e-08` at floor/roundoff scale. |
| `stageA_tov_frozen_ks_smr_12r` | `8522556` | 12 | `inputs/tde/aurora/tov_frozen_ks_n3_schwarzschild_sym_smr_small_aurora.athinput` | Symmetric; density L2/Linf exactly zero in the checked slice through cycle 8. |
| `stageA_tov_frozen_ks_smr_8r` | `8522577` | 8 | `inputs/tde/aurora/tov_frozen_ks_n3_schwarzschild_sym_smr_small_aurora.athinput` | Symmetric; density L2/Linf exactly zero in the checked slice through cycle 8. |

Metric and plot directories:

- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/stageA_tov_frozen_ks_uniform_12r`
- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/stageA_tov_frozen_ks_smr_12r`
- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/stageA_tov_frozen_ks_smr_8r`

## Stage B: Z4c With Zero Matter Source

| Case | Job | Ranks | Deck | Result |
| --- | --- | ---: | --- | --- |
| `stageB_z4c_zero_tmunu_uniform_12r` | `8522607` | 12 | `inputs/tde/aurora/z4c_tov_ks_zero_tmunu_sym_uniform_small_aurora.athinput` | Symmetric; checked MHD density and Z4c fields have zero L2/Linf asymmetry. |
| `stageB_z4c_zero_tmunu_smr_12r` | `8522618` | 12 | `inputs/tde/aurora/z4c_tov_ks_zero_tmunu_sym_smr_small_aurora.athinput` | Symmetric to roundoff; largest observed relative Linf is `1.18e-07` in `z4c_Gamz`, with absolute differences around `1e-12` to `3e-11`. |

Metric and plot directories:

- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/stageB_z4c_zero_tmunu_uniform_12r`
- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/stageB_z4c_zero_tmunu_smr_12r`

## Recommended Next Discriminators

1. Run coupled-control cases that keep Z4c active but remove one coupling channel at a time:
   - GRMHD on Z4c-produced ADM with matter feedback disabled.
   - Z4c evolution with fluid frozen but initial nonzero `Tmunu` retained.
   - Z4c evolution with fluid frozen and `Tmunu` refreshed from fixed primitives.
2. Add a parity probe at the ADM fields consumed by `DynGRMHD_CalcFluxes`, immediately before x3 reconstruction, to compare full coupled ADM against frozen analytic ADM.
3. If a coupled-control case breaks, instrument x3 left/right reconstructed primitive and metric states at the same target cells already used by `ATHENA_FLUX_DEBUG`.
4. Add AMR only after the coupled discriminator that breaks is identified; log regrid cycles and compare norm jumps against regrid times.
