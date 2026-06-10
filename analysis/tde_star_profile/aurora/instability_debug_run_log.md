# Residual Z4c TOV Instability Debug Log

Date: 2026-06-05

## Current Classification

The static self-gravitating TOV relaxation failure is best classified as an interior residual-Z4c metric blow-up, not star dispersion and not a simple outer-boundary reflection.

Evidence:

- Half-box run `8525879` reached `t = 26.93` cleanly before wall-clock exit, retained the stellar density profile, and showed growing Z4c norms.
- Double-box run `8525983` reproduced failure around `t = 33`. It was clean through `t = 32.5`; by `t = 33.0`, histories contained NaNs and fluid had reset to atmosphere.
- First invalid locations reported by the existing runtime diagnostic were near the stellar center, e.g. `(-0.00625, 0.00625, -0.01875)`, with negative `detg`, negative `psi4`, and very large `K_dd`.

## Background-Subtraction Audit

| Component | Current behavior | Assessment |
| --- | --- | --- |
| Geometry/Ricci/derivative terms | `CalcRHS` reconstructs full and analytic-background Z4c states, computes geometry for both using the same finite-difference operators, and subtracts background RHS pieces for residual evolution. | Background truncation error is mostly factored out for evolved non-gauge Z4c fields. |
| Constraint damping | Khat, Theta, and Gamma damping are expressed as full-minus-background combinations, e.g. `alpha_full*Theta_full - alpha_bg*Theta_bg` and `((Gam - Gamma)_full - (Gam - Gamma)_bg)`. | Constraint damping is background-subtracted in the residual path. |
| Gauge terms | If `evolve_gauge_residual = true`, the standard full and background gauge RHS are computed and subtracted. If false, lapse/shift/B residual RHS are zero. | Background-subtracted when evolved. The non-evolved path previously discarded the initial TOV lapse residual. |
| KO dissipation | KO dissipation is applied to the residual state `u0`, not the reconstructed full state. | Appropriate for a residual formulation because it does not dissipate the analytic background into the residual. |
| Algebraic projection | Projection reconstructs full state, enforces constraints, then recasts residuals. Existing determinant guard uses `detg > 0 ? detg : 1` before rescaling, which avoids immediate arithmetic failure but does not repair an already invalid conformal metric. | Potential late-stage symptom amplifier; not treated as the root cause yet. |
| ADM conversion validity | `ComputeGeometryData`, `ADMToZ4c`, and `Z4cToADM` invert determinants without broad validity guards. | Correct for production, but diagnostics now track when determinant/psi4 validity is first lost. |

## Code Changes Added For This Pass

- Added `<problem>/metric_diag_history = true`, also env-gated by `ATHENA_METRIC_DIAG_HISTORY`, to extend the `z4c_tov_ks` user history from 2 columns to 20 columns:
  `rho-max`, alpha extrema, chi extrema, conformal determinant extrema, ADM `psi4` extrema, ADM determinant extrema, max `|K_dd|`, max `|A_dd|`, max `|Theta|`, max `|Khat|`, bad-metric cell count, beta/B maxima, and mean density proxy.
- Added `<z4c>/preserve_lapse_residual = true` as a controlled discriminator. It preserves the initial static TOV lapse residual while keeping shift and auxiliary shift residuals prescribed to the analytic background path.

## Two-Node Debug Matrix

All new decks use project storage under `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation`, regular Aurora `debug`, `select=2`, and `ATHENA_WALLTIME=00:50:00`.

| Case | Deck | Purpose |
| --- | --- | --- |
| Baseline diagnostic | `inputs/tde/aurora/z4c_tov_ks_n3_minkowski_static_selfgrav_starfloor_smallbox_diag_2n_aurora.athinput` | Reproduce instability on a smaller box with metric validity history. |
| Preserve TOV lapse residual | `inputs/tde/aurora/z4c_tov_ks_n3_minkowski_static_selfgrav_starfloor_smallbox_lapse_2n_aurora.athinput` | Test whether forcing full lapse to flat background lapse drives the static-star blow-up. |
| Light damping | `inputs/tde/aurora/z4c_tov_ks_n3_minkowski_static_selfgrav_starfloor_smallbox_kappa005_2n_aurora.athinput` | Check whether small residual-consistent constraint damping changes the growth. |
| Static refinement | `inputs/tde/aurora/z4c_tov_ks_n3_minkowski_static_selfgrav_starfloor_smallbox_smr_2n_aurora.athinput` | Remove AMR refine/derefine operations as a discriminator. |

Submit helper:

`analysis/tde_star_profile/aurora/prepare_instability_debug_2n_qsub.sh`

Default dry run:

```bash
analysis/tde_star_profile/aurora/prepare_instability_debug_2n_qsub.sh
```

Default submit:

```bash
analysis/tde_star_profile/aurora/prepare_instability_debug_2n_qsub.sh --submit
```

Override example:

```bash
CASE_NAME=minkowski_static_selfgrav_starfloor_smallbox_lapse_2n \
INPUT_DECK=/home/hzhu/athenak_tde/inputs/tde/aurora/z4c_tov_ks_n3_minkowski_static_selfgrav_starfloor_smallbox_lapse_2n_aurora.athinput \
JOB_NAME=z4c_lapse_2n \
analysis/tde_star_profile/aurora/prepare_instability_debug_2n_qsub.sh --submit
```

## Verification

- Local build: `cmake --build build_sym_debug -j 4` completed.
- Aurora build: `cmake --build /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks -j 8` completed.
- Parse-only checks passed for all four new 2-node decks.
- Local one-cycle smoke with `ATHENA_METRIC_DIAG_HISTORY=1` wrote a 20-column user history:
  `/tmp/athena_metric_smoke/z4c_tov_ks_n3_schwarzschild_sym_tiny_local.user.hst`

## Submitted Jobs

| Job ID | Case | State at submission check |
| --- | --- | --- |
| `8526059` | `minkowski_static_selfgrav_starfloor_smallbox_diag_2n` | Finished cleanly on Athena wall-clock limit |
| `8526097` | `minkowski_static_selfgrav_starfloor_smallbox_lapse_2n` | Finished cleanly on Athena wall-clock limit |

## Baseline Two-Node Result

Job `8526059` ran in the regular `debug` queue on 2 nodes and exited cleanly with PBS `Exit_status = 0`. Athena stopped on the wall-clock guard at `t = 8.33`, cycle `3332`, before reaching the earlier observed failure time near `t = 33`.

The run did not produce invalid metric cells in the elapsed physical time, but it already shows a strong monotone metric drift:

- `bad-metric = 0`
- `detg-min = 8.978599473925195e-01`
- `psi4-min = 9.647235187036189e-01`
- `chi-max = 1.036566415778673e+00`
- `Kdd-max = 4.174683313916232e-03`
- `Add-max = 1.744123345912786e-03`
- `Theta-max = 5.728295969411345e-05`
- `Khat-max = 1.297890672914063e-02`
- `rho-max = 1.221939540213900e-04`

Postprocess outputs:

- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_smallbox_diag_2n/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_smallbox_diag_2n/metric_history_trends.png`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_smallbox_diag_2n/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_smallbox_diag_2n/radial_density_profile_overlay.png`

The next discriminator is the preserve-lapse residual run over the same 2-node debug envelope. The comparison target is whether it suppresses or delays the early `detg-min` drop and `Khat-max` growth over `0 <= t <= 8.33`.

## Preserve-Lapse Two-Node Result

Job `8526097` ran in the regular `debug` queue on 2 nodes and exited cleanly with PBS `Exit_status = 0`. Athena stopped on the wall-clock guard at `t = 8.30`, cycle `3320`, matching the baseline time coverage closely enough for a direct early-growth comparison.

The final metric state is still clean and close to static:

- `bad-metric = 0`
- `detg-min = 9.992970606246744e-01`
- `psi4-min = 9.997656319508000e-01`
- `chi-max = 1.000234422990459e+00`
- `Kdd-max = 1.361558378710262e-04`
- `Add-max = 1.418434038003163e-04`
- `Theta-max = 6.349133459154538e-05`
- `Khat-max = 4.967913989885362e-05`
- `rho-max = 1.221300186599803e-04`

Postprocess outputs:

- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_smallbox_lapse_2n/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_smallbox_lapse_2n/metric_history_trends.png`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_smallbox_lapse_2n/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_smallbox_lapse_2n/radial_density_profile_overlay.png`
- Direct baseline-vs-lapse comparison CSV: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/instability_debug_comparison/baseline_vs_preserve_lapse_metric_comparison.csv`
- Direct baseline-vs-lapse comparison plot: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/instability_debug_comparison/baseline_vs_preserve_lapse_metric_comparison.png`

At the preserve-lapse final time, interpolating the baseline history gives:

| Field | Baseline at `t = 8.30` | Preserve lapse at `t = 8.30` |
| --- | ---: | ---: |
| `detg-min` | `8.985557436617602e-01` | `9.992970606246744e-01` |
| `psi4-min` | `9.649725512949093e-01` | `9.997656319508000e-01` |
| `Khat-max` | `1.293124061977084e-02` | `4.967913989885362e-05` |
| `Kdd-max` | `4.159371360319506e-03` | `1.361558378710262e-04` |
| `rho-max` | `1.221934805147615e-04` | `1.221300186599803e-04` |

## Ranked Conclusion

The first decisive discriminator supports the fixed-flat-gauge hypothesis. With `<z4c>/evolve_gauge_residual = false` and no lapse preservation, the initial TOV lapse residual is discarded and the run evolves a self-gravitating TOV matter profile with the full lapse pinned to the Minkowski background value. That produces rapid interior residual-Z4c metric drift: large `Khat` growth and a falling ADM determinant before any invalid metric cell is counted.

Preserving only the initial TOV lapse residual while continuing to prescribe shift and auxiliary shift residuals almost eliminates the early drift over the same 2-node, same-walltime envelope. The density maximum remains comparable, so the improvement is not from losing the star.

Recommended next fix path:

1. Promote `<z4c>/preserve_lapse_residual = true` as the default for static-TOV-gauge residual self-gravity decks that use `evolve_gauge_residual = false`.
2. Rerun a longer reproduction case, preferably the double-box setup that failed at `t ~ 33`, with `preserve_lapse_residual = true` to verify the late-time failure is removed or substantially delayed.
3. Only after the gauge path is verified at late time, run the light damping and fixed-refinement discriminators as stability margin checks.

## Next Decision Rule

1. If baseline reproduces the interior blow-up, inspect the metric history for the first field to depart: `Kdd-max`, `Add-max`, `Theta-max`, `Khat-max`, `psi4-min`, `detg-min`, or `bad-metric`.
2. If the preserved-lapse case removes or delays the failure substantially, the leading hypothesis becomes static-gauge mismatch: the non-evolved gauge path was evolving TOV matter with the full lapse forced to the flat background lapse.
3. If the static-refinement case removes the failure while adaptive baseline fails, inspect AMR restriction/prolongation/ghost consistency for residual Z4c variables.
4. If damping changes the failure time but not the qualitative growth, use it as a stabilizing knob only after the gauge-path discriminator is resolved.
