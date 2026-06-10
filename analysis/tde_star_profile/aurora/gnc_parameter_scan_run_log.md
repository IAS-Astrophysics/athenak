# Gaussian-Normal Parameter Scan Run Log

Date: 2026-06-05

Objective: run the 2-node Gaussian-normal/frozen-flat-gauge scan from `gnc_stability_parameter_scan_prompt.md`, explicitly with `<z4c>/evolve_gauge_residual = false` and `<z4c>/preserve_lapse_residual = false`.

Reference fixed-flat-gauge baseline:

- Job `8526059`, case `minkowski_static_selfgrav_starfloor_smallbox_diag_2n`, reached `t = 8.33`.
- Final early-drift metrics: `detg-min = 8.978599473925195e-01`, `psi4-min = 9.647235187036189e-01`, `Khat-max = 1.297890672914063e-02`.

## Scan Matrix

| Case | Integrator | CFL | KO diss | kappa1 | Deck |
| --- | --- | ---: | ---: | ---: | --- |
| `gnc_rk2_cfl010_diss030` | `rk2` | `0.1` | `0.3` | `0.0` | `inputs/tde/aurora/z4c_tov_ks_n3_gnc_rk2_cfl010_diss030_aurora.athinput` |
| `gnc_rk2_cfl030_diss030` | `rk2` | `0.3` | `0.3` | `0.0` | `inputs/tde/aurora/z4c_tov_ks_n3_gnc_rk2_cfl030_diss030_aurora.athinput` |
| `gnc_rk3_cfl020_diss030` | `rk3` | `0.2` | `0.3` | `0.0` | `inputs/tde/aurora/z4c_tov_ks_n3_gnc_rk3_cfl020_diss030_aurora.athinput` |
| `gnc_rk4_cfl020_diss030` | `rk4` | `0.2` | `0.3` | `0.0` | `inputs/tde/aurora/z4c_tov_ks_n3_gnc_rk4_cfl020_diss030_aurora.athinput` |
| `gnc_rk2_cfl020_diss000` | `rk2` | `0.2` | `0.0` | `0.0` | `inputs/tde/aurora/z4c_tov_ks_n3_gnc_rk2_cfl020_diss000_aurora.athinput` |
| `gnc_rk2_cfl020_diss050` | `rk2` | `0.2` | `0.5` | `0.0` | `inputs/tde/aurora/z4c_tov_ks_n3_gnc_rk2_cfl020_diss050_aurora.athinput` |
| `gnc_rk2_cfl020_diss030_kappa005` | `rk2` | `0.2` | `0.3` | `0.05` | `inputs/tde/aurora/z4c_tov_ks_n3_gnc_rk2_cfl020_diss030_kappa005_aurora.athinput` |
| `gnc_rk2_cfl020_diss030_kappa010` | `rk2` | `0.2` | `0.3` | `0.10` | `inputs/tde/aurora/z4c_tov_ks_n3_gnc_rk2_cfl020_diss030_kappa010_aurora.athinput` |

All cases keep `nghost = 4`, `reconstruct = wenoz`, `rsolver = llf`, `metric_diag_history = true`, and `damp_kappa2 = 0.0`.

Implementation note: the first CFL 0.3 attempt hit the inherited AMR capacity cap (`max_nmb_per_rank = 64`) when one rank needed 69 MeshBlocks. To avoid an infrastructure failure unrelated to the scan physics, the seven uncompleted scan decks were raised to `max_nmb_per_rank = 128` and re-validated with parse-only checks. The completed CFL 0.1 run used the original cap and did not hit it.

## Parse Checks

The requested Aurora GPU executable aborts on the login node with `Error: no GPU available for execution`, so parse-only validation used the existing local debug executable:

`build_sym_debug/src/athena -i <deck> -n`

All eight decks passed parse-only validation:

- `gnc_rk2_cfl010_diss030`
- `gnc_rk2_cfl030_diss030`
- `gnc_rk3_cfl020_diss030`
- `gnc_rk4_cfl020_diss030`
- `gnc_rk2_cfl020_diss000`
- `gnc_rk2_cfl020_diss050`
- `gnc_rk2_cfl020_diss030_kappa005`
- `gnc_rk2_cfl020_diss030_kappa010`

## Submitted Jobs

| Job ID | Case | Status |
| --- | --- | --- |
| `8526379` | `gnc_rk2_cfl010_diss030` | Finished cleanly on Athena wall-clock limit |
| `8526464` | `gnc_rk2_cfl030_diss030` | Failed early: AMR `max_nmb_per_rank` cap |
| `8526480` | `gnc_rk2_cfl030_diss030` | Failed by `t = 0.5025`; terminated early |
| `8526496` | `gnc_rk3_cfl020_diss030` | Finished cleanly on Athena wall-clock limit |
| `8526584` | `gnc_rk4_cfl020_diss030` | Invalid discriminator; stopped after identifying incomplete MHD RK4 register update |
| `8526605` | `gnc_rk2_cfl020_diss000` | Failed by `t = 0.5`; terminated early |
| `8526623` | `gnc_rk2_cfl020_diss050` | Finished cleanly on Athena wall-clock limit |
| `8526717` | `gnc_rk2_cfl020_diss030_kappa005` | Finished cleanly on Athena wall-clock limit |
| `8526792` | `gnc_rk2_cfl020_diss030_kappa010` | Finished cleanly on Athena wall-clock limit |

Constraint note: KO dissipation is capped at `diss <= 0.5`; the active scan only uses `0.0`, `0.3`, and `0.5`.

## Case Results

### `gnc_rk2_cfl010_diss030`

- Job: `8526379`
- Final time/cycle: `t = 4.19875`, cycle `3359`
- PBS state/status: finished, `Exit_status = 0`
- Stop reason: Athena wall-clock guard
- Final diagnostics:
  - `bad-metric = 0`
  - `detg-min = 9.730167528944440e-01`
  - `psi4-min = 9.909234497878505e-01`
  - `Khat-max = 6.500035582238242e-03`
- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl010_diss030/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl010_diss030/metric_history_trends.png`
- Baseline common-time comparison: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl010_diss030/baseline_common_time_comparison.csv`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl010_diss030/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl010_diss030/radial_density_profile_overlay.png`

### `gnc_rk2_cfl030_diss030`

- Initial job `8526464` failed from the inherited AMR capacity cap, not from the scan physics.
- Retry job: `8526480`, with `max_nmb_per_rank = 128`
- Stop reason: early failure detected, then `qdel` requested to avoid wasting allocation
- First nonfinite metric-history time: `t = 0.5025`
- Final available history time: `t = 1.75125`
- Failure diagnostics:
  - `bad-metric = 2.4068544e+07`
  - `detg-min = -inf`
  - `Khat-max = 4.453469565884941e+305`
  - primitive recovery reports `NANS_IN_CONS`
- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl030_diss030/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl030_diss030/metric_history_trends.png`
- Baseline common-time comparison: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl030_diss030/baseline_common_time_comparison.csv`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl030_diss030/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl030_diss030/radial_density_profile_overlay.png`

### `gnc_rk3_cfl020_diss030`

- Job: `8526496`
- Final time/cycle: `t = 5.0`, cycle `2000`
- PBS state/status: finished, `Exit_status = 0`
- Stop reason: Athena wall-clock guard
- Final diagnostics:
  - `bad-metric = 0`
  - `detg-min = 9.620222525507242e-01`
  - `psi4-min = 9.871770249836247e-01`
  - `Khat-max = 7.737845314771991e-03`
- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk3_cfl020_diss030/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk3_cfl020_diss030/metric_history_trends.png`
- Baseline common-time comparison: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk3_cfl020_diss030/baseline_common_time_comparison.csv`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk3_cfl020_diss030/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk3_cfl020_diss030/radial_density_profile_overlay.png`

### `gnc_rk4_cfl020_diss030`

- Job: `8526584`
- PBS state/status: stopped with `qdel`, `Exit_status = 271`
- Classification: invalid as a coupled MHD/Z4c stability discriminator
- Reason: `MHD::CopyCons` only copies `u0 -> u1` at stage 1 and does not implement the RK4 low-storage `u1 += delta*u0` update used by Hydro and Z4c. The early history reflects this: `rho-max` drops to atmosphere by `t = 0.25`, so the case is not comparable to RK2/RK3.
- Final finite metric-history time before stopping: `t = 0.75`
- Final finite diagnostics:
  - `bad-metric = 0`
  - `detg-min = 9.998254951549364e-01`
  - `psi4-min = 9.999418242830241e-01`
  - `Khat-max = 5.831290052473904e-05`
- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk4_cfl020_diss030/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk4_cfl020_diss030/metric_history_trends.png`
- Baseline common-time comparison: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk4_cfl020_diss030/baseline_common_time_comparison.csv`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk4_cfl020_diss030/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk4_cfl020_diss030/radial_density_profile_overlay.png`

### `gnc_rk2_cfl020_diss000`

- Job: `8526605`
- PBS state/status: stopped with `qdel`, `Exit_status = 271`
- Stop reason: early metric failure detected
- First nonfinite metric-history time: `t = 0.5`
- Last fully finite metric-history time: `t = 0.25`
- Last fully finite diagnostics:
  - `bad-metric = 0`
  - `detg-min = 9.479626550149397e-01`
  - `psi4-min = 9.823443302204897e-01`
  - `Khat-max = 1.770006464777144e-01`
- Failure-tail diagnostics include `detg-min = -inf`, `psi4-min = -6.102438519472217e+12` at `t = 0.5`, and Z4c constraint-history NaNs from `t = 0.5`.
- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss000/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss000/metric_history_trends.png`
- Baseline common-time comparison: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss000/baseline_common_time_comparison.csv`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss000/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss000/radial_density_profile_overlay.png`

### `gnc_rk2_cfl020_diss050`

- Job: `8526623`
- Final time: `t = 7.8575`
- PBS state/status: finished, `Exit_status = 0`
- Stop reason: Athena wall-clock guard
- Final diagnostics:
  - `bad-metric = 0`
  - `detg-min = 9.088054283237599e-01`
  - `psi4-min = 9.686278925537823e-01`
  - `chi-max = 1.032388193327270e+00`
  - `Kdd-max = 3.942555563036658e-03`
  - `Add-max = 1.608599889515903e-03`
  - `Theta-max = 1.538118085042127e-05`
  - `Khat-max = 1.221311109494026e-02`
  - `rho-max = 1.221756493937704e-04`
  - `rho-mass = 5.189810148625040e-01`
- Common-time comparison to baseline at `t = 7.8575`: defect ratios are near unity (`detg-min` `0.9976`, `psi4-min` `0.9975`, `Khat-max` `0.9987`), so `diss = 0.5` does not materially suppress the fixed-flat gauge drift relative to the baseline.
- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss050/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss050/metric_history_trends.png`
- Baseline common-time comparison: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss050/baseline_common_time_comparison.csv`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss050/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss050/radial_density_profile_overlay.png`

### `gnc_rk2_cfl020_diss030_kappa005`

- Job: `8526717`
- Final time: `t = 7.7025`
- PBS state/status: finished, `Exit_status = 0`
- Stop reason: Athena wall-clock guard
- Final diagnostics:
  - `bad-metric = 0`
  - `detg-min = 9.120197360900936e-01`
  - `psi4-min = 9.697685124759484e-01`
  - `chi-max = 1.031173921544294e+00`
  - `Kdd-max = 3.874709439436750e-03`
  - `Add-max = 1.596555869045252e-03`
  - `Theta-max = 4.191198591247975e-05`
  - `Khat-max = 1.198410466807650e-02`
  - `rho-max = 1.221844664059364e-04`
  - `rho-mass = 5.187157764345153e-01`
- Common-time comparison to baseline at `t = 7.7025`: defect ratios are essentially unity (`detg-min` `0.9999`, `psi4-min` `0.9998`, `Khat-max` `1.0000`), so `kappa1 = 0.05` does not materially suppress the fixed-flat gauge drift.
- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa005/metric_history_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa005/metric_history_trends.png`
- Baseline common-time comparison: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa005/baseline_common_time_comparison.csv`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa005/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa005/radial_density_profile_overlay.png`

### `gnc_rk2_cfl020_diss030_kappa010`

- Job: `8526792`
- Final time: `t = 7.8250`
- PBS state/status: finished, `Exit_status = 0`
- Stop reason: Athena wall-clock guard
- Final diagnostics:
  - `bad-metric = 0`
  - `detg-min = 9.093264605008207e-01`
  - `psi4-min = 9.688129669626173e-01`
  - `chi-max = 1.032190974007252e+00`
  - `Kdd-max = 3.916341783861281e-03`
  - `Add-max = 1.594325433140325e-03`
  - `Theta-max = 4.532745585553935e-05`
  - `Khat-max = 1.217783700912506e-02`
  - `rho-max = 1.221866774085763e-04`
  - `rho-mass = 5.189399536820712e-01`
- Final Z4c constraints:
  - `C-norm2 = 2.242614240726838e-04`
  - `H-norm2 = 1.563792730521881e-04`
  - `M-norm2 = 6.787466020202405e-05`
  - `Z-norm2 = 1.370696983790461e-09`
  - `Theta-norm = 2.008030536381906e-09`
- Metric history summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa010/metric_history_summary.csv`
- Z4c constraint summary: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa010/z4c_constraint_summary.csv`
- Metric history trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa010/metric_history_trends.png`
- Z4c constraint trends: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa010/z4c_constraint_trends.png`
- Baseline common-time comparison: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa010/baseline_common_time_comparison.csv`
- Radial density profiles: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa010/radial_density_profiles.csv`
- Radial density overlay: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/gnc_rk2_cfl020_diss030_kappa010/radial_density_profile_overlay.png`

## Postprocess Outputs

Combined outputs target:

`/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/`
