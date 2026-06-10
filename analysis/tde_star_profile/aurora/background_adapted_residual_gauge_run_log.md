# Background-Adapted Residual Gauge Run Log

## 2026-06-10 setup and local smoke tests

Baseline cleanup:

- Removed stale untracked core dumps only: `core.5350`, `core.54499`, `core.82917`.
- Committed and pushed the previous residual-gauge/debugging baseline before new implementation work:
  - commit: `b82cc106 Record residual gauge debugging diagnostics`
  - branch: `project/bg_subtract`
  - remote: `origin/project/bg_subtract`

Implementation snapshot after the baseline commit:

- Added input option `<z4c>/residual_gauge`.
  - default: `standard_subtract`
  - experimental: `background_adapted`
- Added residual gauge tunables:
  - `<z4c>/residual_lapse_f`
  - `<z4c>/residual_lapse_damping`
  - `<z4c>/residual_shift_damping`
- In `background_adapted` mode, the residual lapse RHS uses the background-shift advection of `alpha_res` plus a residual-only source:

```text
rhs(alpha_res) = lapse_advect L_beta_bg(alpha_res)
               - residual_lapse_f f(alpha_bg) alpha_bg Khat_res
               - residual_lapse_damping alpha_res
```

- This removes the direct `Khat_bg * alpha_res` source from the standard full-minus-background lapse subtraction.
- Added a gated pgen path `<problem>/pure_background = true` to initialize an analytic background with zero residual state for fixed-point checks.
- Replaced the history diagnostic column `aK-full` with `src-adapt`, keeping `src-full`, `src-bg`, `src-res`, `aK-bg`, and `Khat-res`.

Local build:

- Command: `cmake --build build_sym_debug --target athena --parallel 8`
- Result: passed.

Local smoke, TOV perturbation:

- Deck: `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_bgadapt_tiny_local.athinput`
- Executable: `build_sym_debug/src/athena`
- Run dir: `/tmp/athenak_bgadapt_smoke`
- Result: passed to `nlim = 1`, final time `1.0e-2`.
- History summary:
  - `last_bad-metric = 0`
  - `last_alpha-res = 1.1544015596776089e-06`
  - `last_Khat-res = 1.7228982046590960e-08`
  - `last_src-res = 3.6948556036975328e-07`
  - `last_src-adapt = 2.8724376457809850e-08`

Local smoke, pure-background fixed point:

- Deck: `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_purebg_tiny_local.athinput`
- Executable: `build_sym_debug/src/athena`
- Run dir: `/tmp/athenak_bgadapt_purebg_debug`
- Result: passed to `nlim = 1`, final time `1.0e-2`.
- History summary:
  - `last_bad-metric = 0`
  - `last_alpha-res = 0`
  - `last_beta-res = 0`
  - `last_B-res = 0`
  - `last_Gam-res = 0`
  - `last_src-full = 8.0780303355820060e-01`
  - `last_src-bg = 8.0780303355820060e-01`
  - `last_src-res = 0`
  - `last_src-adapt = 0`
  - `last_Khat-res = 0`
- Direct `Z4C_DEBUG post_rhs` max residual RHS reductions stayed at roundoff:
  - stage 1: `alpha_abs=0`, `beta_abs=0`, `B_abs=0`, `Gam_abs=3.885781e-16`, `Theta_abs=1.990936e-15`, `Khat_abs=1.387779e-16`, `g_abs=3.612562e-16`, `A_abs=1.086874e-15`
  - stage 2: `alpha_abs=0`, `beta_abs=0`, `B_abs=0`, `Gam_abs=1.226233e-16`, `Theta_abs=1.406870e-15`, `Khat_abs=5.551115e-17`, `g_abs=2.762547e-16`, `A_abs=7.912564e-16`

Note:

- The pure-background Z4c constraint history is not zero (`C-norm2 ~ 1.44`) because that history measures full-background discrete constraints on the coarse tiny grid. The fixed-point test above is the residual RHS/gauge-source test; it passes to roundoff.

Aurora GPU build:

- Timestamp: `2026-06-10T04:21:35Z`
- Command: `BUILD_PARALLELISM=8 analysis/tde_star_profile/aurora/configure_intel_gpu_z4c_tov_ks.sh`
- Result: passed.
- Executable: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks/src/athena`

## 2026-06-10 Aurora debug job: background-adapted lapse no-advection

- Case: `z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_aurora`
- Deck: `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_aurora.athinput`
- Job id: `8533757`
- Queue/project: `debug` / `MHDTidal`
- Nodes/ranks: 2 nodes, 12 ranks per node
- Athena walltime limit: `00:19:00`
- Purpose: direct comparison against the prior no-advection residual-lapse Schwarzschild debug case that failed around `t ~ 3.2`.
- Status:
  - `2026-06-10T04:22:54Z`: submitted.
  - `2026-06-10T04:32Z`: still queued. PBS comment: `Not Running: User has reached queue debug running job limit.`
  - `2026-06-10T04:43:38Z`: still queued for the same limit. Current debug slot is occupied by job `8533730`, elapsed `00:44` of requested `01:00`.
  - `2026-06-10T04:54Z`: still queued. PBS comment changed to `Not Running: Insufficient amount of resource: at_queue`.
  - `2026-06-10T04:58:02Z`: started running.
  - `2026-06-10T05:18:03Z`: finished with `Exit_status = 0`, `resources_used.walltime = 00:19:04`.
- Result:
  - Reached code time `t = 1.972499999999968`.
  - All history rows finite through the final row.
  - `bad-metric = 0`.
  - This is not decisive because the old standard-subtract no-advection control stayed finite until `t = 3.1` and first produced nonfinite history at `t = 3.2`.
  - Early-time behavior is comparable to the old failing control: at `t ~ 1.9`, `alpha-res ~ 4.25e-4`, `Gam-res ~ 3.62e-2`, and `C-norm2 ~ 0.89`.
- Analysis output:
  - CSV: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_aurora/residual_gauge_history_summary.csv`
  - Plots: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_aurora/residual_gauge_plots`
  - PBS stdout: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_aurora/pbs/bgadapt_noadv2n.o8533757`

## 2026-06-10 Aurora debug job: background-adapted lapse no-advection, longer walltime

- Case: `z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_1h`
- Deck: `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_aurora.athinput`
- Job id: `8533793`
- Queue/project: `debug` / `MHDTidal`
- Nodes/ranks: 2 nodes, 12 ranks per node
- PBS walltime: `01:00:00`
- Athena walltime limit: `00:55:00`
- Purpose: same setup as job `8533757`, but long enough to pass the old standard-subtract failure time near `t = 3.2`.
- Status:
  - `2026-06-10T05:21:37Z`: submitted.
  - `2026-06-10T05:31Z`: still queued. PBS comment: `Not Running: User has reached queue debug running job limit.` Current debug slot is occupied by job `8533784`, elapsed `00:10` of requested `01:00`.
  - `2026-06-10T06:07:10Z`: started running.
- Result:
  - The run reproduced the old failure.
  - Last finite history row: `t = 3.0999999999999450`.
  - First nonfinite history row: `t = 3.1999999999999420`.
  - The job was manually cancelled after the failure was visible in history, giving PBS `Exit_status = 271`.
  - Last finite `user.hst` values:
    - `alpha-res = 3.3760734527461989e-01`
    - `Gam-res = 2.0715707545824941e+01`
    - `Khat-res = 3.2191725806133050e+01`
    - `src-res = 7.5112128978652436e+01`
    - `src-adapt = 6.2777377984378383e+01`
    - `bad-metric = 0`
  - Last finite `z4c.user.hst` values:
    - `C-norm2 = 4.2292216243840102e+05`
    - `H-norm2 = 3.4586683082894149e+05`
    - `M-norm2 = 7.7052343708275002e+04`
    - `Theta-norm = 6.9453827714077909e-01`
- Analysis output:
  - CSV: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_1h/residual_gauge_history_summary.csv`
  - Plots: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_1h/residual_gauge_plots`
  - PBS stdout: `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/z4c_tov_ks_n3_schwarzschild_bgadapt_lapse_noadvect_hi2n_1h/pbs/bgadapt_noadv1h.o8533793`
- Interpretation:
  - The implemented background-adapted lapse source is not sufficient to remove the Schwarzschild residual-lapse instability.
  - The failure timing matches the old standard-subtract no-advection control: finite through `t = 3.1`, first bad row at `t = 3.2`.
  - The growth is already present in residual geometric/constraint fields before the nonfinite row, so the next probe should target whether `Khat_res`/constraint growth is generated by the Z4c geometric residual RHS independently of the precise lapse source factor.
