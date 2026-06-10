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
