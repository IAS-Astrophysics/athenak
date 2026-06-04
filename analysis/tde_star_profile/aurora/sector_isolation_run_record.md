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

Run the new dense-output matrix in this order, submitting one Aurora job at a time:

| Step | Case | Deck | Interpretation |
| --- | --- | --- | --- |
| 0 | `minkowski_static_uniform_dense` | `inputs/tde/aurora/z4c_tov_ks_minkowski_static_sym_uniform_dense_aurora.athinput` | Establish uniform-grid static-star numerical symmetry drift. |
| 1 | `minkowski_static_smr_dense` | `inputs/tde/aurora/z4c_tov_ks_minkowski_static_sym_smr_dense_aurora.athinput` | Establish refinement-only static-star symmetry drift. |
| 2 | `schwarzschild_infall_smr_dense` | `inputs/tde/aurora/z4c_tov_ks_schwarzschild_infall_sym_smr_dense_aurora.athinput` | Reproduce the coupled infall asymmetry with dense `xy`/`xz` slices. |
| 3 | `schwarzschild_zero_feedback_smr_dense` | `inputs/tde/aurora/z4c_tov_ks_schwarzschild_infall_zero_feedback_smr_dense_aurora.athinput` | Fluid evolves on Z4c-produced ADM, but matter feedback is disabled. A break here points to ADM-to-GRMHD, reconstruction, ghost/communication, or refinement infrastructure. |
| 4 | `schwarzschild_fixed_mhd_tmunu_smr_dense` | `inputs/tde/aurora/z4c_tov_ks_schwarzschild_fixed_mhd_tmunu_smr_dense_aurora.athinput` | Z4c evolves with frozen fluid and retained initial nonzero `Tmunu`. A break here points to matter-feedback-to-Z4c or Z4c source/refinement path. |
| 5 | `schwarzschild_fixed_mhd_refresh_tmunu_smr_dense` | `inputs/tde/aurora/z4c_tov_ks_schwarzschild_fixed_mhd_refresh_tmunu_smr_dense_aurora.athinput` | Same as step 4, but `Tmunu` is refreshed from fixed primitives each stage to test the refresh path. |

Use `analysis/tde_star_profile/aurora/z4c_bg_validation_metrics.py` for all dense runs.
It reports absolute, local-relative, and peak-relative L2/Linf metrics for `xy`
y-reflection and `xz` z-reflection, including high-density masks
`rho > 1e-12`, `1e-10`, and `1e-8`. A case should be classified as breaking only
when absolute and peak-relative metrics grow coherently in time and exceed the
Minkowski static baseline in high-density material or relevant ADM/Z4c fields.

If a coupled-control case breaks, add the next probe at the ADM fields consumed
by `DynGRMHD_CalcFluxes`, immediately before x3 reconstruction, then instrument
x3 left/right reconstructed primitive and metric states at the same target cells
used by `ATHENA_FLUX_DEBUG`.
