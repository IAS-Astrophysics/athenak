# Goal-Mode Prompt: 2-Node Gaussian-Normal Stability Parameter Scan

Repository: `/home/hzhu/athenak_tde`

Goal: Run a controlled 2-node Aurora debug-queue parameter scan for the static TOV residual-Z4c instability in Gaussian-normal/frozen-flat gauge, explicitly **without preserving the TOV lapse residual**, to determine whether RK integrator, CFL, KO dissipation, or constraint damping materially affects the early metric drift and late instability.

## Summary

Use the existing 2-node small-box diagnostic setup as the baseline, but keep the intentionally problematic gauge path:

```text
<z4c>/evolve_gauge_residual = false
<z4c>/preserve_lapse_residual = false
```

This means the full lapse is pinned to the Minkowski background value, `alpha = 1`, and the star's initial lapse residual is not retained. The purpose is not to fix the gauge issue, but to test whether numerical-method choices change the stability/growth rate in this Gaussian-normal setup.

Existing reference runs:

- Baseline 2-node fixed-flat-gauge run: job `8526059`, reached `t = 8.33`.
  - `detg-min = 8.978599473925195e-01`
  - `psi4-min = 9.647235187036189e-01`
  - `Khat-max = 1.297890672914063e-02`
- Preserve-lapse control: job `8526097`, reached `t = 8.30`.
  - `detg-min = 9.992970606246744e-01`
  - `psi4-min = 9.997656319508000e-01`
  - `Khat-max = 4.967913989885362e-05`

Do not use the preserve-lapse setting in this scan except as an already-completed comparison control.

## Constraints

- Do not change `NGHOST`.
- Do not revert unrelated dirty-tree changes.
- Keep all new diagnostics/input variants input- or env-gated.
- Use `source ~/athenak_env` before Python analysis.
- Use project `MHDTidal`.
- Use storage under `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation`.
- Use regular Aurora `debug` queue with 2 nodes.
- Submit one job at a time.
- Use local parse/smoke tests only before submitting.
- Monitor jobs at roughly 10-minute intervals to conserve tokens.
- Do not claim stability from raw low-density relative errors; report absolute and normalized metric-history changes.

## Starting Point

Use this deck as the base template:

`inputs/tde/aurora/z4c_tov_ks_n3_minkowski_static_selfgrav_starfloor_smallbox_diag_2n_aurora.athinput`

Keep these fixed unless the scan variable explicitly changes them:

```text
nghost = 4
refinement = adaptive
num_levels = 4
integrator = rk2              # changed only in RK scan
cfl_number = 0.2              # changed only in CFL scan
reconstruct = wenoz
rsolver = llf
use_analytic_background = true
evolve_gauge_residual = false
preserve_lapse_residual = false
diss = 0.3                    # changed only in diss scan
damp_kappa1 = 0.0             # changed only in damping scan
damp_kappa2 = 0.0
metric_diag_history = true
```

Use the existing submit helper:

`analysis/tde_star_profile/aurora/prepare_instability_debug_2n_qsub.sh`

Use the existing postprocessing pattern from:

`analysis/tde_star_profile/aurora/instability_debug_run_log.md`

## Required Scan Matrix

Create new input decks with explicit case names. Keep the baseline deck untouched.

Run these cases in this order, stopping early only if a case fails before wall-clock:

1. CFL reduction, same RK and diss:
   - `gnc_rk2_cfl010_diss030`
   - `integrator = rk2`
   - `cfl_number = 0.1`
   - `diss = 0.3`
   - `damp_kappa1 = 0.0`, `damp_kappa2 = 0.0`

2. CFL increase, same RK and diss:
   - `gnc_rk2_cfl030_diss030`
   - `integrator = rk2`
   - `cfl_number = 0.3`
   - `diss = 0.3`
   - `damp_kappa1 = 0.0`, `damp_kappa2 = 0.0`

3. RK3 at baseline CFL/diss:
   - `gnc_rk3_cfl020_diss030`
   - `integrator = rk3`
   - `cfl_number = 0.2`
   - `diss = 0.3`
   - `damp_kappa1 = 0.0`, `damp_kappa2 = 0.0`

4. RK4 at baseline CFL/diss:
   - `gnc_rk4_cfl020_diss030`
   - `integrator = rk4`
   - `cfl_number = 0.2`
   - `diss = 0.3`
   - `damp_kappa1 = 0.0`, `damp_kappa2 = 0.0`

5. No KO dissipation:
   - `gnc_rk2_cfl020_diss000`
   - `integrator = rk2`
   - `cfl_number = 0.2`
   - `diss = 0.0`
   - `damp_kappa1 = 0.0`, `damp_kappa2 = 0.0`

6. Stronger KO dissipation:
   - `gnc_rk2_cfl020_diss050`
   - `integrator = rk2`
   - `cfl_number = 0.2`
   - `diss = 0.5`
   - `damp_kappa1 = 0.0`, `damp_kappa2 = 0.0`

7. Light constraint damping:
   - `gnc_rk2_cfl020_diss030_kappa005`
   - `integrator = rk2`
   - `cfl_number = 0.2`
   - `diss = 0.3`
   - `damp_kappa1 = 0.05`, `damp_kappa2 = 0.0`

8. Moderate constraint damping:
   - `gnc_rk2_cfl020_diss030_kappa010`
   - `integrator = rk2`
   - `cfl_number = 0.2`
   - `diss = 0.3`
   - `damp_kappa1 = 0.10`, `damp_kappa2 = 0.0`

Use `tlim = 45.0`, PBS walltime `01:00:00`, and `ATHENA_WALLTIME=00:50:00` for all cases.

## Implementation Tasks

1. Inspect the dirty tree before editing.
2. Create the eight new decks under `inputs/tde/aurora/`.
3. Ensure every new deck has:
   - unique `<job>/basename`
   - `metric_diag_history = true`
   - `evolve_gauge_residual = false`
   - `preserve_lapse_residual = false`
4. Run local parse-only checks for every deck:
   ```bash
   /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks/src/athena -i <deck> -n
   ```
   If that executable is not runnable locally, use the existing local build executable that was used for prior parse checks.
5. Submit one case at a time using:
   ```bash
   CASE_NAME=<case> \
   INPUT_DECK=/home/hzhu/athenak_tde/inputs/tde/aurora/<deck> \
   JOB_NAME=<short_job_name> \
   analysis/tde_star_profile/aurora/prepare_instability_debug_2n_qsub.sh --submit
   ```
6. Monitor each job until it finishes or fails.
7. After each job, run:
   - metric-history summary/plot
   - radial density diagnostics
   - direct comparison against baseline job `8526059`

## Analysis Requirements

For each case, report:

- job id
- deck path
- run directory
- final time/cycle
- PBS exit status
- whether the run stopped cleanly on wall-clock or failed
- first nonfinite time if any
- final:
  - `bad-metric`
  - `detg-min`
  - `psi4-min`
  - `chi-max`
  - `Kdd-max`
  - `Add-max`
  - `Theta-max`
  - `Khat-max`
  - `rho-max`
  - `rho-mass`
- interpolated comparison at common time against baseline `8526059`
- growth-rate proxy for `1 - detg-min`, `1 - psi4-min`, and `Khat-max`
- radial density profile paths

Create a combined scan table and plot under:

`/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan/`

Required outputs:

- `gnc_parameter_scan_summary.csv`
- `gnc_parameter_scan_metric_trends.png`
- `gnc_parameter_scan_common_time_comparison.csv`
- `gnc_parameter_scan_notes.md`

## Classification Rules

Classify results as follows:

- If smaller CFL strongly suppresses growth while higher RK order does not, classify as time-step sensitivity.
- If RK3/RK4 suppress growth at the same CFL while RK2 does not, classify as time-integrator sensitivity.
- If stronger KO suppresses growth and zero KO accelerates growth, classify as high-frequency residual noise sensitivity.
- If constraint damping suppresses `Khat`, `Theta`, or determinant drift, classify as dampable constraint-mode growth.
- If none of the numerical scans materially changes the drift, reinforce the conclusion that the Gaussian-normal/frozen-flat-gauge setup itself is the dominant pathology.

Important: do not call any case "stable" unless it reaches at least the same physical time as the 2-node baseline and has materially smaller metric drift. Prefer "delayed/suppressed over this 2-node envelope" unless a longer run is performed.

## Deliverables

- New decks for the eight scan cases.
- Parse/smoke-test result summary.
- Aurora job ids and final statuses.
- Metric-history and radial-density output paths.
- Combined scan CSV and plot.
- A short conclusion ranking RK, CFL, KO dissipation, and constraint damping by impact on the Gaussian-normal instability.
- Recommended next run: either a longer 2-node/8-node best-parameter test or the original double-box failing case with the most stabilizing parameter choice.
