# Residual Z4c RHS Term Debug Run Log

Follow-up to `background_adapted_residual_gauge_run_log.md`. Goal: identify the
dominant residual RHS term driving the `Khat_res`/constraint blow-up at
`t = 3.2` in the Schwarzschild Kerr-Schild no-advection residual-lapse debug
cases.

## 2026-06-10 Step 1: control comparison at common times

Sources (raw `.hst` files):

- standard-subtract no-advection failing control:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_lapse_noadvect_hi2n/`
- background-adapted no-advection failing control:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_1h/`
- lapse-source-off control (lapse residual effectively frozen at `~9.7e-6`):
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_lapse_sourceoff_hi2n/`

| quantity | t | std_noadv | bgadapt_noadv | sourceoff |
|---|---|---|---|---|
| alpha-res | 1.9 | 4.236580e-04 | 4.250618e-04 | 9.728425e-06 |
| alpha-res | 2.5 | 5.432862e-03 | 5.421427e-03 | 9.728378e-06 |
| alpha-res | 2.9 | 7.974381e-02 | 7.667682e-02 | 9.728346e-06 |
| alpha-res | 3.1 | 5.763329e-01 | 3.376073e-01 | 9.728331e-06 |
| Gam-res | 1.9 | 3.611463e-02 | 3.622381e-02 | 7.252685e-02 |
| Gam-res | 2.5 | 4.740656e-01 | 4.752473e-01 | 1.093685e-01 |
| Gam-res | 2.9 | 6.080018e+00 | 6.068306e+00 | 1.353402e-01 |
| Gam-res | 3.1 | 3.517488e+01 | 2.071571e+01 | 1.485893e-01 |
| Khat-res | 2.9 | n/a (column absent) | 7.206033e+00 | n/a |
| Khat-res | 3.1 | n/a | 3.219173e+01 | n/a |
| C-norm2 | 1.9 | 8.862064e-01 | 8.914475e-01 | 1.099895e-03 |
| C-norm2 | 2.5 | 4.097288e+02 | 4.128206e+02 | 4.332060e-04 |
| C-norm2 | 2.9 | 2.545820e+04 | 2.575238e+04 | 2.974856e-04 |
| C-norm2 | 3.1 | 4.477951e+05 | 4.229222e+05 | 2.560065e-04 |
| H-norm2 | 3.1 | 3.540312e+05 | 3.458668e+05 | 2.146742e-04 |
| M-norm2 | 3.1 | 9.376006e+04 | 7.705234e+04 | 4.125608e-05 |
| Theta-norm | 3.1 | 8.565386e-01 | 6.945383e-01 | 3.502072e-08 |
| alpha-min | 3.1 | 6.778344e-01 | 6.778344e-01 | 6.778344e-01 |
| alpha-max | 3.1 | 1.551387e+00 | 1.312670e+00 | 9.802937e-01 |
| detg-min | 3.1 | 4.222263e-01 | 5.574791e-01 | 1.040609e+00 |
| bad-metric | 3.1 | 0 | 0 | 0 |

Failure bracketing:

- std_noadv: last finite `t = 3.0999999999999450`, first bad `t = 3.1999999999999420`.
- bgadapt_noadv: last finite `t = 3.0999999999999450`, first bad `t = 3.1999999999999420`.
- sourceoff: finite through end of run at `t = 4.91`; constraints *decay*
  (C-norm2 from `1.1e-3` at `t=1.9` to `2.6e-4` at `t=3.1`).

Observations:

- std_noadv and bgadapt_noadv are nearly identical at every common time
  (differences < 2% until `t = 2.9`). Removing the `Khat_bg * alpha_res` lapse
  source changed nothing measurable: the instability does not run through that
  term.
- Constraint growth is smooth and exponential: C-norm2 multiplies by ~460 per
  `Delta t = 0.4-0.6`, i.e. e-folding time ~0.1 code units, already active by
  `t ~ 1.9` (C-norm2 0.89 vs 1.1e-3 in sourceoff at the same time).
- The growth requires the closed loop `alpha_res <-> (Khat_res, geometry)`.
  With the lapse residual frozen (sourceoff) the residual geometry evolution on
  the same data is constraint-damping-stable.

## 2026-06-10 Step 4: background-subtraction audit of `z4c_calcrhs.cpp`

Audited at commit `45eda6de` (tree clean apart from untracked handoff note).

- All geometric residual RHS terms are exact pairwise differences
  `RHS_full[full] - RHS_bg[bg]` built from two independent `GeometryData`
  evaluations with the same stencils: `Ddalpha`, `alpha(AA + K^2/3)`, Lie
  terms, kappa1/kappa2 damping pairs, `chi` source, `Gam` terms (`DA_u`,
  `A^{ab} d_b alpha`, damping with `Gamma_u` recomputed per state), `g_dd`
  and `A_dd` terms including Ricci/Rphi and trace parts.
- Matter terms enter with full-state lapse/metric only and are not
  background-subtracted; this is correct because the Kerr-Schild background is
  vacuum (`RHS_bg` matter terms are identically zero).
- KO dissipation is applied to the evolved residual `u0` only, so background
  truncation error is not injected through dissipation.
- The algebraic constraint projection (`EnforceAlgConstr` task) operates on the
  reconstructed full state and recasts the residual afterwards; the background
  itself is projected in `UpdateBackgroundState`. Consistent.
- No term was found where the residual RHS mixes full and background fields
  inconsistently in the standard-subtract path. The background-adapted path
  only modifies the gauge RHS.

Conclusion of the audit: the residual evolution is algebraically equivalent to
evolving the full state and subtracting a static background, so the `t = 3.2`
blow-up should be interpreted as an instability of the *full* scheme in this
gauge configuration (1+log without advection, shift frozen at Kerr-Schild,
`damp_kappa1 = 0`), unless the term diagnostics show otherwise.

## 2026-06-10 new gated diagnostics

Commit `45eda6de Add gated term-by-term residual Z4c RHS diagnostics`:

- `<z4c>/rhs_term_debug = true|false` (default false),
  `<z4c>/rhs_term_debug_stride = N` (default 20, cycles, stage-1 only).
- `Z4C_RHS_TERM_MAX` lines: domain-wide max |contribution| for 23 geometry
  term categories (Khat: dda/alg/adv/damp/mat; Theta: adv/Ht/damp/mat;
  chi: adv/src; Gam: DA/Adal/adv/damp/mat; g: A/adv; A: ric/tr/alg/adv/mat),
  KO dissipation per field group, and counts of nonfinite `u_rhs`/`u0`
  entries.
- `Z4C_RHS_TERM_LOC` lines: full local term breakdown plus
  full/background/residual field values, coordinates, meshblock gid/level and
  bounds, and distance to the BH at the argmax of `|rhs Khat_res|`,
  `|Khat_res|`, `|Gam_res|`, `|Theta_res|` (rank owning the global max prints).

Local validation: tiny deck
`inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_rhsdbg_tiny_local.athinput`
run to `t = 1.0` with `build_sym_debug/src/athena`; diagnostics print finite,
consistent values (`rhs_Khat` equals the sum of the Khat term slots plus the
KO contribution at the sampled points; e.g. at cycle 2 the term sum is
`-8.90e-7` vs stored `rhs_Khat = -8.55e-7` with `KO_Khat <= 4.9e-8`; the
stored `u_rhs` includes KO dissipation).

Local build: `cmake --build build_sym_debug --target athena --parallel 8`,
passed.

Aurora GPU build (2026-06-10T13:40Z):
`BUILD_PARALLELISM=16 analysis/tde_star_profile/aurora/configure_intel_gpu_z4c_tov_ks.sh`,
passed. Executable:
`/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks/src/athena`.

## 2026-06-10 Aurora debug job: bgadapt no-advection + term diagnostics

- Case: `z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_rhsdbg_hi2n_1h`
- Deck: `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_rhsdbg_hi2n_aurora.athinput`
  (identical to the failing bgadapt hi2n deck plus `rhs_term_debug = true`,
  `rhs_term_debug_stride = 20`)
- Job id: `8534357`
- Queue/project: `debug` / `MHDTidal`
- Nodes/ranks: 2 nodes, 12 ranks per node
- PBS walltime: `01:00:00`, Athena walltime limit: `00:55:00`
- Commit: `45eda6de`, tree clean (untracked handoff note only)
- Physics/gauge mode: dynamical GRMHD + Z4c residual evolution,
  `residual_gauge = background_adapted`, `evolve_lapse_residual = true`,
  `evolve_shift_residual = false`, `lapse_advect = 0`, `damp_kappa1 = 0`,
  `diss = 0.5`, CFL 0.2, rk2
- Status:
  - `2026-06-10T13:41:59Z`: submitted.
  - `2026-06-10T13:47:18Z`: started running.
  - `2026-06-10T14:20Z`: failure reproduced and fully captured; job cancelled
    with `qdel` after the blow-up to free the debug slot (job had reached
    `t ~ 3.5`).
- Run dir:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_rhsdbg_hi2n_1h`
- Post dir:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_rhsdbg_hi2n_1h`
  (`residual_gauge_history_summary.csv`, `residual_gauge_plots/`,
  `pbs/bgadapt_rhsdbg.o8534357`)
- Failure bracketing (identical to the non-instrumented control, so the
  diagnostics did not perturb the dynamics; last finite history values agree
  digit-for-digit):
  - history: last finite `t = 3.0999999999999450`, first bad
    `t = 3.1999999999999420`.
  - term diagnostics: all categories finite at cycle 1240 (`t = 3.100`);
    first nonfinite at cycle 1260 (`t = 3.150`), with
    `nonfinite_rhs = 2.9e6` and `nonfinite_u0 = 4.8e5` entries. So the first
    bad time is bracketed to `3.100 < t_bad <= 3.150` and the state `u0` is
    already nonfinite at `t = 3.150`.

### Term hierarchy (domain-wide max |contribution|, stage 1)

| t | Khat_dda | Khat_alg | Khat_adv | Theta_Ht | Gam_DA | A_ric | A_tr | KO_Khat |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 6.18e-01 | 1.08e-05 | 1.40e-02 | 2.74e-01 | 4.07e-01 | 4.38e-01 | 3.91e-01 | 3.52e-02 |
| 1.50 | 2.14e+00 | 4.30e-05 | 3.88e-02 | 7.87e-01 | 1.11e+00 | 1.50e+00 | 1.26e+00 | 7.47e-02 |
| 2.00 | 5.32e+01 | 7.86e-03 | 6.81e-01 | 1.06e+01 | 1.82e+01 | 3.34e+01 | 2.56e+01 | 1.30e+00 |
| 2.50 | 2.30e+02 | 6.28e-01 | 4.88e+00 | 8.01e+01 | 1.53e+02 | 1.46e+02 | 1.24e+02 | 9.37e+00 |
| 3.00 | 4.37e+03 | 1.62e+02 | 8.18e+01 | 1.18e+03 | 2.42e+03 | 2.58e+03 | 2.08e+03 | 2.21e+02 |

- Dominant `Khat_res` term at every sampled time: `Khat_dda`, the residual of
  the lapse second covariant derivative `-(DDalpha_full - DDalpha_bg)`. It
  exceeds the algebraic term `Khat_alg` by 1.5-4 orders of magnitude and the
  advection term by ~1.5 orders.
- `Gam_DA` (contains `dKhat` derivatives through `DA_u`) and the `A_dd`
  Ricci/lapse-second-derivative terms `A_ric`/`A_tr` track `Khat_dda` at a
  factor ~0.5; this is the lapse <-> Khat <-> (Gam, A) derivative loop, not an
  algebraic or damping term.
- Damping terms are identically zero (`damp_kappa1 = 0` deck).
- Matter terms stay at `Khat_mat ~ 1.5e-3`, `Theta_mat ~ 3.0e-3`,
  `A_mat ~ 2e-5` for the entire run: matter sources are NOT the driver.
- KO dissipation contributions stay ~5% of `Khat_dda`: KO is not injecting the
  growth and at `diss = 0.5` cannot remove it either.
- e-folding time of `Khat_dda` over `t in [1.5, 3.0]`: ~0.20 code units
  (consistent with C-norm2 e-folding 0.11 for the squared norm).

### First-bad-location

- The argmax of |rhs Khat_res|, |Khat_res|, |Gam_res| and |Theta_res| sits at
  the star from cycle 0 onward and never near the BH:
  - cycle 0: `x ~ 39.6-40.5`, `|y|,|z| < 0.5` (star interior; star center
    `x = 40`), level 8 (finest).
  - t = 1.9: all maxima at `x ~ 38.41`, `|y|,|z| < 0.16`, level 8.
  - t = 3.1 (last finite): all four maxima cluster within ~0.04 of
    `(x, y, z) ~ (38.59, 0.0, 0.0)`, level 8, `r_bh ~ 38.6`. Star center at
    `t = 3.1` is `x ~ 39.69` (boost `-0.1`), so the hotspot is ~1.1 behind the
    center, at/behind the trailing stellar surface, in floored-atmosphere
    cells (`tmunu_E ~ 1.3e-16`).
- Local state at the `t = 3.1` hotspot: `alpha_full` swings 0.93 -> 1.15
  between cells ~0.03 apart; `Khat_res = -32.2` and `+27.5` in adjacent cells;
  `K_full = -29` vs `+44.8`; `R_full = -4616` vs `+1384`. The unstable mode
  has grid-scale (zone-to-zone sign-flipping) structure.
- The argmax repeatedly lands in the first interior cell layer of level-8
  meshblocks (x = 38.40625, 39.20625 with block faces at multiples of 0.4),
  i.e. adjacent to meshblock/refinement ghost zones in the wake of the
  star-following AMR region.
- Initial-data seed: at cycle 0 the diagnostics show `Theta_Ht ~ -0.99` and
  discrete `R_full ~ -2.04` (vs `R_bg ~ 3e-6`) at star-interior level-8 points
  where the physical conformal Ricci scalar should be `~ 16 pi rho ~ 6e-3`.
  The initial C-norm2 is 0.104 at `t = 0`, drops to ~3e-3 by `t = 0.7` (the
  seed transient mostly disperses), then grows exponentially from
  `t ~ 0.7-1.2` onward. So the boosted-TOV-on-KS initial data carries an O(1)
  local discrete constraint defect at the star which seeds the unstable mode.

### Interpretation after run 1

- The instability is a star-localized, grid-scale, exponentially growing
  (`tau ~ 0.2`) coupled mode of the residual lapse and residual geometry
  (`alpha_res <-> Khat_res` via `DDalpha`, feeding `Gam_res`/`A_res` via
  `DA_u` and the Ricci terms). It is not: BH-related, background-subtraction
  leakage (BH region stays quiescent; bg terms cancel cleanly), matter source
  terms, constraint damping, or KO dissipation.
- Both failing gauges (standard subtract and background adapted) reduce to
  nearly the same dynamics at the star because `alpha_bg ~ 0.975` and
  `Khat_bg ~ 1.3e-3` there; this explains the identical failure times.
- Remaining discriminator: whether the star's *motion* across the grid (with
  frozen shift, no lapse advection, and a moving AMR refinement wake) is
  required, or whether a static star in this residual gauge is also unstable.

## 2026-06-10 Aurora debug job: no-boost discriminator

- Case: `z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_noboost_rhsdbg_hi2n_1h`
- Deck: `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_noboost_rhsdbg_hi2n_aurora.athinput`
  (identical to the term-debug deck except `star_boost_x = 0.0`)
- Job id: `8534419`
- Queue/project: `debug` / `MHDTidal`
- Nodes/ranks: 2 nodes, 12 ranks per node
- PBS walltime: `01:00:00`, Athena walltime limit: `00:55:00`
- Purpose: single-discriminator test of whether star motion (frozen shift +
  no lapse advection + moving AMR wake) is required for the instability.
- Status:
  - `2026-06-10T14:22:42Z`: submitted.
