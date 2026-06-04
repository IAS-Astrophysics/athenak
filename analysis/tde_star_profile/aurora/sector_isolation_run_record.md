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

## Dense Matrix Submissions

| Case | Job | Status | Run directory | Metric directory |
| --- | --- | --- | --- | --- |
| `minkowski_static_uniform_dense` | `8522794` | Complete; MHD `dens`, `velx`, `vely`, and `velz` are exactly symmetric in the `all`, `rho > 1e-12`, and `rho > 1e-10` masks. The `rho > 1e-8` mask has no valid central-window pairs. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/minkowski_static_uniform_dense` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_uniform_dense` |
| `minkowski_static_smr_dense` | `8522837` | Complete; MHD density is exactly symmetric in all masks. High-density (`rho > 1e-8`) velocity differences are small: max Linf abs `1.14e-11` (`velx`), `1.36e-11` (`vely`), `5.91e-12` (`velz`), with max Linf peak-relative `2.09e-6`. ADM checked fields remain symmetric. Z4c has a refinement-only baseline: `z4c_Theta` max Linf abs `1.44e-5`, max Linf peak-relative `1.0`, and `z4c_Gamy/Gamz` max Linf abs about `1e-10`. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/minkowski_static_smr_dense` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_smr_dense` |
| `schwarzschild_infall_smr_dense` | `8522851` | Complete; reproduced a coherent coupled-infall MHD symmetry break above the Minkowski baselines. High-density (`rho > 1e-8`) density asymmetry grows monotonically from zero to cycle 20/time `0.25` in both `xy` and `xz`: max Linf abs `3.40e-8`, Linf local-relative `1.97e-3`, Linf peak-relative `3.55e-4`, L2 peak-relative `8.14e-5`. High-density velocity max Linf abs at final time is `4.07e-4` (`velx`), `8.20e-6` (`vely` in `xy` / `velz` in `xz`), and `4.21e-7` (`velz` in `xy` / `vely` in `xz`). | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_infall_smr_dense` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_infall_smr_dense` |
| `schwarzschild_zero_feedback_smr_dense` | `8522863` | Complete; disabling matter feedback does not remove the MHD break. High-density (`rho > 1e-8`) density asymmetry at cycle 20/time `0.25` is nearly identical to the full coupled infall in both `xy` and `xz`: max Linf abs `3.42e-8`, Linf local-relative `1.97e-3`, Linf peak-relative `3.57e-4`, L2 peak-relative `8.15e-5`. High-density velocity asymmetry also matches the full coupled run. Z4c/ADM mirror differences stay small: `z4c_Theta` max Linf abs `1.46e-11`, `z4c_Gamy/Gamz` max Linf abs `2.91e-11`/`5.82e-11`, checked ADM fields zero. This points away from matter-feedback-to-Z4c as the necessary trigger and toward the ADM-to-GRMHD / x3 reconstruction path or coupled communication/task-order context. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_zero_feedback_smr_dense` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_zero_feedback_smr_dense` |
| `schwarzschild_fixed_mhd_tmunu_smr_dense` | `8522867` | Complete; frozen MHD remains exactly symmetric in the high-density masks. Z4c/ADM source-path asymmetry stays at absolute roundoff scale: largest Z4c all-mask Linf abs is `2.33e-10` (`Azz/Ayy`), `Gamz` max Linf abs `5.82e-11`, `Theta` max Linf abs `3.64e-12`; largest ADM all-mask Linf abs is `1.82e-12`. This does not reproduce the MHD density break and further points away from matter-source-to-Z4c as the main source. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_fixed_mhd_tmunu_smr_dense` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_fixed_mhd_tmunu_smr_dense` |
| `schwarzschild_fixed_mhd_refresh_tmunu_smr_dense` | `8522883` | Complete; refreshing `Tmunu` from fixed primitives also keeps frozen MHD exactly symmetric. Z4c/ADM asymmetry remains at absolute roundoff scale, similar to the retained-source fixed-fluid case: largest Z4c all-mask Linf abs is `2.33e-10`, `Theta` max Linf abs `7.28e-12`; largest ADM all-mask Linf abs is `1.46e-11` (`Kxz`). This does not reproduce the MHD density break. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_fixed_mhd_refresh_tmunu_smr_dense` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_fixed_mhd_refresh_tmunu_smr_dense` |

## Dense Matrix Classification

- Fluid-only on frozen analytic Schwarzschild ADM: no break in the prior Stage A runs.
- Spacetime-only / zero-matter Z4c: no macroscopic break in the prior Stage B runs.
- Minkowski static baselines: uniform is exactly symmetric; SMR has zero density asymmetry and only tiny velocity differences, plus a refinement-only `z4c_Theta` baseline.
- Full coupled Schwarzschild infall: reproduces coherent MHD density/velocity symmetry growth.
- Coupled infall with `Tmunu` feedback disabled: reproduces the MHD break nearly identically while Z4c/ADM fields remain symmetric to small absolute error.
- Fixed-fluid matter-source-to-Z4c cases, with retained and refreshed `Tmunu`: do not reproduce the MHD break.

Current best classification: the necessary path is not matter feedback into Z4c.
The break is isolated to the fluid evolution running in the coupled Z4c/ADM
context, most likely the ADM-to-GRMHD data path, dynamic x3 reconstruction/flux
inputs, or coupled boundary/refinement/task-order communication feeding those
inputs. The next decisive probe should compare parity pairs for the ADM fields
actually consumed by `DynGRMHD_CalcFluxes` immediately before x3 reconstruction,
then dump the x3 left/right primitive and metric states at the same target
locations used by `ATHENA_FLUX_DEBUG`.
