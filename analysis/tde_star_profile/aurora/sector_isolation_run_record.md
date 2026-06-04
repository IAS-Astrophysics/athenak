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

## Zero-Feedback X3 Diagnostic

| Case | Job | Status | Run directory | Metric directory |
| --- | --- | --- | --- | --- |
| `schwarzschild_zero_feedback_x3debug_c19` | `8522976` | Complete; diagnostic built from commit `69bdf213` and enabled with `mhd/dyngr_x3_debug=true mhd/dyngr_x3_debug_cycle=19`. The run reproduces the prior zero-feedback break: high-density (`rho > 1e-8`) final density Linf abs `3.421155270189047e-8`, local-relative Linf `1.974173482905793e-3`, peak-relative Linf `3.5737975125244427e-4`, L2 peak-relative `8.150517976511866e-5` in `xy` and the same values in `xz`. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_zero_feedback_x3debug_c19` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_zero_feedback_x3debug_c19` |
| `schwarzschild_zero_feedback_cycle0_debug_n1` | `8523001` | Complete; enabled `ATHENA_SYM_DEBUG=1`, `ATHENA_FLUX_DEBUG=1`, `ATHENA_SYM_X_TARGET=20.03125`, `ATHENA_SYM_Z_TARGET=0.0`, and `mhd/dyngr_x3_debug=true mhd/dyngr_x3_debug_cycle=0` with `time/nlim=1`. The first high-density density break appears after one full cycle: Linf abs `2.473825588822365e-10`, local-relative Linf `6.328741409524629e-6`, peak-relative Linf `2.6777820876067915e-6` at `x=20.03125`, mirror `y=+-0.03125`. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_zero_feedback_cycle0_debug_n1` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_zero_feedback_cycle0_debug_n1` |
| `schwarzschild_zero_feedback_c2pdebug_c0_n1` | `8523033` | Complete; built from commit `254a131e` and enabled `ATHENA_C2P_DEBUG=1`, `ATHENA_C2P_DEBUG_CYCLE=0`, `ATHENA_SYM_X_TARGET=20.03125`, and `ATHENA_SYM_Z_TARGET=0.0` with `time/nlim=1`. The final high-density density metric matches the cycle-0 debug run: Linf abs `2.473825588822365e-10`, local-relative Linf `6.328741409524629e-6`, peak-relative Linf `2.6777820876067915e-6`. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_zero_feedback_c2pdebug_c0_n1` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_zero_feedback_c2pdebug_c0_n1` |
| `schwarzschild_zero_feedback_rkvars_c0_n1` | `8523062` | Complete; built from commit `628cecc2` and enabled expanded `ATHENA_SYM_DEBUG`, `ATHENA_FLUX_DEBUG`, and `ATHENA_C2P_DEBUG` at cycle 0 with `time/nlim=1`. The final high-density density metric again matches the one-cycle break: Linf abs `2.473825588822365e-10`, local-relative Linf `6.328741409524629e-6`, peak-relative Linf `2.6777820876067915e-6`. The first nonzero conserved-variable asymmetry at the target appears after stage-1 `MHD_SrcTerms`, not in flux/RK. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_zero_feedback_rkvars_c0_n1` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_zero_feedback_rkvars_c0_n1` |
| `schwarzschild_zero_feedback_coordsrc_c0_n1` | `8523105` | Complete; built from commit `a9499420` and enabled `ATHENA_COORDSRC_DEBUG=1`, `ATHENA_COORDSRC_DEBUG_CYCLE=0`, `ATHENA_COORDSRC_DEBUG_STAGE=1`, `ATHENA_C2P_DEBUG=1`, `ATHENA_SYM_X_TARGET=20.03125`, and `ATHENA_SYM_Z_TARGET=0.0` with `time/nlim=1`. The final high-density density metric again matches the one-cycle break: Linf abs `2.473825588822365e-10`, local-relative Linf `6.328741409524629e-6`, peak-relative Linf `2.6777820876067915e-6`. `COORDSRCDBG` shows the high-density mirror pair has symmetric pre-source conserved variables, but the below-side source sees `alpha=0` and `beta=(0,0,0)` with zero lapse/shift derivatives and therefore adds zero momentum/energy source, while the above side sees the expected full lapse/shift and adds the nonzero source. | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_zero_feedback_coordsrc_c0_n1` | `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_zero_feedback_coordsrc_c0_n1` |

Diagnostic stdout:

- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_zero_fb_x3dbg.o8522976`
- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_zero_fb_c2pdbg.o8523033`
- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_zero_fb_rkvars.o8523062`
- `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_zero_fb_coordsrc.o8523105`

Key cycle-19 result at `x=20.03125`, `y=+-0.03125`, `zface=0`:

- ADM/metric inputs consumed by GRMHD are symmetric at the sampled x3 face:
  `alpha`, `beta_x`, parity-adjusted `beta_y`, `gxx`, parity-adjusted `gxy`,
  and `detg` differ only at roundoff (`0` to `~6e-16`; `gxy` local relative
  `~4e-13`).
- Primitive cell states are already asymmetric before x3 reconstruction:
  stage 1 `cc_km1_rho`/`cc_k_rho` parity-adjusted differences are
  `3.102e-8` with local relative `5.049e-4`; stage 2 differences grow to
  `3.415e-8` with local relative `5.717e-4`.
- Reconstructed x3 states inherit the asymmetry:
  stage 1 `wl_rho`/`wr_rho` differences are `7.808e-9` with local relative
  `1.155e-4`; stage 2 differences are `8.611e-9` with local relative
  `1.311e-4`.
- The sampled x3 mass flux has opposite sign across the y mirror pair
  (`-2.40e-11` versus `+2.40e-11` at stage 1; `-2.71e-11` versus `+2.71e-11`
  at stage 2). Since the primitive states are already asymmetric, this is not
  yet evidence that the Riemann solver is the first source.

Interpretation: the late-time zero-feedback break is upstream of x3
reconstruction/Riemann and not caused by asymmetric ADM values at the GRMHD
flux-consumption point. The next discriminator should target cycle 0/1 with
`ATHENA_SYM_DEBUG`/`ATHENA_FLUX_DEBUG` and the x3 probe enabled to identify
whether the first nonzero density asymmetry appears in flux computation,
flux correction, RK update, source terms, conservative-variable communication,
prolongation, or the subsequent conserved-to-primitive conversion.

Cycle-0/one-step result:

- Stage 0 and stage 1 are symmetric at the target through `MHD_RecvFlux`,
  `MHD_RKUpdate`, `MHD_SrcTerms`, `MHD_RecvU`, and `MHD_Prolongate`.
- At the start of stage 2, `MHD_RecvFlux` sees primitive density already
  asymmetric while conserved density is still symmetric:
  `rho_w` differs by `1.97733922145020349e-10` (local relative
  `2.195557e-6`), but `rho_u` differs by zero.
- The stage-2 RK update then propagates the primitive-state asymmetry into
  conserved density: `rho_u` differs by `5.61108990217509818e-11`.
- Stage-2 `DynGRMHD_CalcFluxes` and `RKDebug` flux differences are therefore
  downstream of asymmetric primitives entering stage 2, not the first source.

Interpretation update: the first observed break at the target is in the
primitive refresh between stage-1 `MHD_Prolongate` and the start of stage 2.
The existing `MHD_AfterC2P` probe only runs inside `MHD_Newdt`, which skips
non-final RK stages, so the next decisive diagnostic must instrument the
dynamic `DynGRMHDPS::ConToPrim` task directly before and after the EOS
conversion.

Cycle-0 C2P result:

- Stage 0 C2P is exactly symmetric in conserved density, momenta, energy,
  primitives, and cell-centered ADM quantities at the target pair.
- Stage 1 C2P input has symmetric conserved density and symmetric
  cell-centered ADM determinant/metric values, but conserved momentum and
  energy are already asymmetric: `momx_u` differs by
  `2.09709683795105968e-9` (local relative `1.997031e-4`), `momy_u` and
  `momz_u` differ by `2.67772438672822527e-12` in parity-adjusted comparison,
  and `tau_u` differs by `2.03922646536067729e-10`.
- Stage 1 C2P converts those already-asymmetric momentum/energy inputs into a
  primitive density difference of `1.97733922145020349e-10` while conserved
  density remains exactly symmetric.
- Stage 2 then carries the primitive asymmetry forward and the RK update
  propagates it into conserved density.

Interpretation update: C2P is the first observed place where density becomes
asymmetric, but it is not the first source. The upstream source is a
stage-1 update to conserved momentum and energy, most likely either GRMHD flux
divergence or dynamic-coordinate source terms. The next diagnostic must expand
`SYMDBG`, `FLUXDBG`, and `RKDBG` from density-only logging to all core MHD
variables so stage-1 `MHD_RKUpdate` and `MHD_SrcTerms` can be separated.

Cycle-0 conserved-variable result:

- Stage-1 `DynGRMHD_CalcFluxes`, `MHD_RecvFlux`, `MHD_RKUpdate`, and `RKDBG`
  flux divergence are exactly parity-symmetric at the target for `rho`,
  `momx`, parity-adjusted `momy`, `momz`, and `tau`.
- Stage-1 `MHD_RKUpdate` leaves the target pair symmetric in the core
  conserved variables.
- Stage-1 `MHD_SrcTerms` introduces the first nonzero conserved asymmetry:
  `momx_u` differs by `2.09709683795105968e-9`, parity-adjusted `momy_u` by
  `-2.67772438672822527e-12`, parity-adjusted `momz_u` by
  `2.67772438672822527e-12`, and `tau_u` by
  `-2.03922646536067729e-10`; `rho_u` remains symmetric.
- Stage-1 `MHD_RecvU`, `MHD_Prolongate`, and C2P then carry the source-term
  asymmetry forward, and stage 2 propagates it into conserved density.

Interpretation update: the first breaking task is dynamic GRMHD coordinate
source terms, specifically `DynGRMHDPS::AddCoordTermsEOS`, rather than
flux/RK, x3 reconstruction/Riemann, C2P, or matter feedback. The next probe is
the env-gated `COORDSRCDBG` diagnostic inside `DynGRMHD_AddCoordTerms`, targeted
to cycle 0/stage 1, to split the source into geometry, shift-gradient,
lapse-gradient, and energy-source subcomponents.

Cycle-0 coordinate-source result:

- The one-cycle high-density density break is reproduced again: Linf abs
  `2.473825588822365e-10`, local-relative Linf `6.328741409524629e-6`,
  peak-relative Linf `2.6777820876067915e-6`.
- At the high-density target pair (`x=20.03125`, `z=0.03125`,
  `y=+-0.03125`), the stage-1 pre-source conserved variables are parity
  symmetric: `pre_rho=9.4981435955234346e-05`,
  `pre_momx=-1.0498977562073042e-05`, parity-adjusted
  `pre_momy=+-1.4737333629859224e-09`, `pre_momz=-1.4737333629859224e-09`,
  and `pre_tau=5.2656730781440339e-07`.
- The above-side source (`rank=10`, `gid=232`) receives full lapse/shift:
  `alpha=0.95353031364429219`, `beta_x=0.09077972002248523`,
  `beta_y=0.00014162202811620161`, `beta_z=0.00014162202811620161`. It adds
  `src_momx=-2.0970968379518207e-09`,
  `src_momy=-2.6777243867284305e-12`,
  `src_momz=-2.6777243867284305e-12`, and
  `src_tau=2.039226465360487e-10`.
- The below-side source (`rank=8`, `gid=194`) receives `alpha=0` and
  `beta_x=beta_y=beta_z=0`; the lapse/shift derivatives are also zero. It
  therefore adds `src_momx=src_momy=src_momz=src_tau=0`, while the cached
  metric determinant and metric derivatives are nonzero and parity-consistent.

Interpretation update: the coordinate source formula is not the first suspect.
The break is caused by `DynGRMHDPS::AddCoordTermsEOS` consuming zero/stale ADM
lapse/shift data on only part of the mirror pair. Since `adm.g_dd`,
`adm.vK_dd`, and `adm.psi4` are cached in `ADM::u_adm`, while
`adm.alpha`/`adm.beta_u` currently alias `pz4c->u0` when Z4c is present, the
most likely root is that MHD coordinate sources are reading Z4c residual/RK
stage storage instead of a stable reconstructed-full ADM cache. A focused fix
should make the ADM cache own lapse and shift as well, or otherwise enforce a
single stable full-ADM data path for GRMHD source/flux consumption.

## Dense Matrix Classification

- Fluid-only on frozen analytic Schwarzschild ADM: no break in the prior Stage A runs.
- Spacetime-only / zero-matter Z4c: no macroscopic break in the prior Stage B runs.
- Minkowski static baselines: uniform is exactly symmetric; SMR has zero density asymmetry and only tiny velocity differences, plus a refinement-only `z4c_Theta` baseline.
- Full coupled Schwarzschild infall: reproduces coherent MHD density/velocity symmetry growth.
- Coupled infall with `Tmunu` feedback disabled: reproduces the MHD break nearly identically while Z4c/ADM fields remain symmetric to small absolute error.
- Fixed-fluid matter-source-to-Z4c cases, with retained and refreshed `Tmunu`: do not reproduce the MHD break.

Current best classification: the necessary path is not matter feedback into Z4c.
The break is isolated to the fluid evolution running in the coupled Z4c/ADM
context. The cycle-19 x3 diagnostic shows the sampled ADM values are symmetric
at the GRMHD consumption point, while primitive and reconstructed fluid states
are already asymmetric before x3 reconstruction/Riemann. The cycle-0
conserved-variable diagnostic further excludes the stage-1 flux/RK path at the
target and localizes the first conserved asymmetry to dynamic GRMHD coordinate
source terms. The coordinate-source diagnostic shows the source inputs are not
geometrically asymmetric; instead, one side of the high-density mirror pair
receives zero lapse/shift from `adm.alpha`/`adm.beta_u`, while the other side
receives the expected full ADM values. Current best classification:
coupled ADM-to-GRMHD data path, specifically stale or unstable lapse/shift
storage exposed to GRMHD coordinate source terms through the Z4c-backed ADM
aliasing path.
