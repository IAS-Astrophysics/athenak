# Residual-Gauge Schwarzschild Debug Run Log

Date: 2026-06-07

Goal prompt: `/home/hzhu/athenak_tde/residual_gauge_debug_goal.md`

## Objective

Use short Aurora `debug` queue, `select=2` jobs to isolate the early residual-gauge Z4c blow-up seen in the 64-node Schwarzschild AMR64 run.

The failed reference run was:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_headon_density_amr_64c_resgauge
```

It was finite through `t = 3.0`, bad by `t = 3.5`, and showed Z4c constraint growth before the lapse minimum collapsed.

## Code Changes For This Pass

The following changes are input-gated and preserve existing behavior by default:

- Added `<z4c>/evolve_lapse_residual`, defaulting to `<z4c>/evolve_gauge_residual`.
- Added `<z4c>/evolve_shift_residual`, defaulting to `<z4c>/evolve_gauge_residual`.
- Updated residual reconstruction, recasting, prescribed gauge residuals, pgen ADM-to-residual initialization, and residual Z4c RHS to honor the split lapse/shift controls.
- Kept `<z4c>/evolve_gauge_residual` as the compatibility master/default for existing decks.
- Reused the existing `<problem>/metric_diag_history = true` gated history and changed its final four slots to residual-gauge diagnostics:
  - `alpha-res`: max absolute residual lapse
  - `beta-res`: max absolute residual shift
  - `B-res`: max absolute residual auxiliary shift
  - `Gam-res`: max absolute residual conformal connection

Touched files:

```text
src/z4c/z4c.hpp
src/z4c/z4c.cpp
src/z4c/z4c_calcrhs.cpp
src/pgen/z4c_tov_ks.cpp
analysis/tde_star_profile/aurora/residual_gauge_history_summary.py
```

## Base Two-Node Deck

Created:

```text
inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_2n_aurora.athinput
```

This deck uses the small Schwarzschild AMR geometry with the star at `x = 20`, `tlim = 8`, dense history output `dt = 0.1`, residual gauge enabled, `diss = 0.5`, zero constraint damping, and metric/gauge history enabled.

Important settings:

```text
queue = debug
select = 2
RANKS_PER_NODE = 12
use_analytic_background = true
evolve_gauge_residual = true
evolve_lapse_residual = true
evolve_shift_residual = true
diss = 0.5
damp_kappa1 = 0.0
shift_Gamma = 1.0
shift_advect = 1.0
shift_eta = 2.0
metric_diag_history = true
```

## Verification

- Local build: `cmake --build build_sym_debug -j 4` passed.
- Local deck smoke:

  ```bash
  ./build_sym_debug/src/athena \
    -i inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_2n_aurora.athinput \
    -d /tmp/athena_rg_full_2n_smoke \
    time/nlim=0 time/tlim=0.0 output2/dt=100 output3/dt=100
  ```

  Result: initialized successfully and terminated on `nlim=0`. The one-rank local smoke produced 268 MeshBlocks and required a local `mesh_refinement/max_nmb_per_rank=512` override. The Aurora 24-rank deck uses `max_nmb_per_rank=64` because the initial debug load is only 11-12 MeshBlocks/rank.

- Aurora build:

  ```bash
  cmake --build /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks -j 8
  ```

  Result: completed successfully.

- Analysis helper:

  ```bash
  source ~/athenak_env
  python3 analysis/tde_star_profile/aurora/residual_gauge_history_summary.py \
    --run-dir /tmp/athena_rg_full_2n_smoke \
    --output /tmp/athena_rg_full_2n_smoke/summary.csv
  ```

  Result: passed and reported the residual-gauge history columns.

## Submitted Jobs

| Case | Job ID | Queue | Nodes | Deck | Status |
| --- | --- | --- | ---: | --- | --- |
| `resgauge_full_2n` | `8529730` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_2n_aurora.athinput` | failed during setup, `Exit_status=143`; rank allocation failed because the first deck used `max_nmb_per_rank=512` on GPU |
| `resgauge_full_2n` | `8529755` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_2n_aurora.athinput` | completed cleanly through `t = 8.0`; no non-finite history row; small/coarse deck did not reproduce the early residual-gauge blow-up |
| `resgauge_full_hi2n` | `8529771` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_hi2n_aurora.athinput` | failed during setup, `Exit_status=1`; ranks needed 147-148 MeshBlocks but deck had `max_nmb_per_rank=128` |
| `resgauge_full_hi2n` | `8529778` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_hi2n_aurora.athinput` | reproduced failure; finite through `t = 3.4`, first bad histories at `t = 3.5`, wall-clock guard stopped at `t = 4.855`; `Exit_status=0` |
| `resgauge_lapse_only_hi2n` | `8529810` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_lapse_only_hi2n_aurora.athinput` | failed earlier than full residual-gauge case; finite through `t = 3.1`, first bad histories at `t = 3.2`; PBS walltime killed after bad data, `Exit_status=-29` |
| `schwarzschild_frozengauge_hi2n` | `8529850` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_frozengauge_hi2n_aurora.athinput` | stable through wall-clock stop at `t = 4.9875`; all histories finite; `C-norm2 = 5.16e-5`; both residual lapse and residual shift disabled |
| `resgauge_lapse_noadvect_hi2n` | `8529879` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_lapse_noadvect_hi2n_aurora.athinput` | failed like lapse-only; finite through `t = 3.1`, first bad histories at `t = 3.2`; `lapse_advect=0` did not stabilize |
| `resgauge_lapse_sourceoff_hi2n` | `8533586` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_lapse_sourceoff_hi2n_aurora.athinput` | stable through wall-clock stop at `t = 4.91`; all histories finite; `C-norm2 = 1.13e-4`; disabling the non-advective lapse driver stabilized the lapse-only/no-advection failure |
| `resgauge_lapse_noadvect_diag_hi2n` | `8533675` | `debug` | 2 | `inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_lapse_noadvect_hi2n_aurora.athinput` | reproduced failure with lapse-source history diagnostics; finite through `t = 3.1`, first bad row at `t = 3.2`; `src-res` grew to `1.21e2` while `src-bg` stayed `8.08e-1` |

## Completed Result: `resgauge_full_2n`

Run directory:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_full_2n
```

Summary CSV:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_full_2n/history_summary.csv
```

Summary:

- Finished with `Exit_status = 0`.
- Reached `t = 8.0`.
- All checked histories stayed finite.
- Final `user.hst` diagnostics included `bad-metric = 0`, `alpha-res = 8.2072111408093562e-06`, `beta-res = 3.4156033527910079e-06`, `B-res = 0`, and `Gam-res = 4.7952924379570417e-05`.
- Final `z4c.user.hst` diagnostics included `C-norm2 = 9.6301832456478220e-02`, `H-norm2 = 7.4735447167720848e-02`, `M-norm2 = 1.9298433086704269e-02`, and `Theta-norm = 8.6271192516430281e-09`.

Interpretation:

The small/coarse 2-node deck is not a valid reproducer for the failed AMR64 residual-gauge case. The next baseline must preserve more of the failed run's star location and resolution.

## Active Job: `resgauge_full_hi2n`

Submitted:

```text
job id: 8529771
queue: debug
nodes: 2
case: resgauge_full_hi2n
deck: inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_hi2n_aurora.athinput
```

Purpose:

This is the higher-resolution 2-node reproducer, using the failed run's star position and finest resolution more closely while reducing the domain to fit regular `debug`.

First attempt result:

```text
job id: 8529771
Exit_status: 1
stdout: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_rg_hi2n.o8529771
```

Failure was before time integration:

```text
Root grid requires more MeshBlocks (nmb_thisrank=147/148) than specified by
<mesh_refinement>/max_nmb_per_rank=128
```

Action:

Raised the high-resolution deck's `max_nmb_per_rank` from `128` to `192`. This changes allocation capacity only; the physics, resolution, and refinement layout are unchanged.

Second attempt:

```text
job id: 8529778
max_nmb_per_rank: 192
```

Result:

```text
Exit_status: 0
walltime: 00:50:05
run dir: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_full_hi2n
stdout: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_rg_hi2n.o8529778
summary csv: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_full_hi2n/history_summary.csv
metric/gauge plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_full_hi2n/metric_gauge_trends.png
constraint plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_full_hi2n/z4c_constraint_trends.png
MHD integral plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_full_hi2n/mhd_integral_trends.png
```

History summary:

- `mhd.hst`, `user.hst`, and `z4c.user.hst` each have last fully finite row at `t = 3.4`.
- First bad row is `t = 3.5`.
- Last finite `user.hst` row at `t = 3.4` had:
  - `alpha-min = 1.5073280443039730e-01`
  - `detg-min = 2.6179289735760079e-02`
  - `Theta-max = 5.3275244312654863e+01`
  - `Khat-max = 8.3800179009872380e+01`
  - `bad-metric = 0`
  - `alpha-res = 8.2429056866299311e-01`
  - `beta-res = 5.8481088018221405e-01`
  - `B-res = 0`
  - `Gam-res = 1.2945850465826950e+02`
- Last finite `z4c.user.hst` row at `t = 3.4` had:
  - `C-norm2 = 1.3491340267898049e+06`
  - `H-norm2 = 1.1885612407410180e+06`
  - `M-norm2 = 1.6055044164682890e+05`
  - `Theta-norm = 3.4341633354433569e+00`

Interpretation:

The high-resolution 2-node reduced-domain deck is a valid reproducer for the failed AMR64 residual-gauge path. The failure timing matches the 64-node case closely: constraints grow rapidly and histories become bad at `t = 3.5`.

The stdout later reports primitive-solver `NANS_IN_CONS` with NaN magnetic fields near `(x,y,z) = (9.85, -1.75, -1.35)`, but that appears after the Z4c/gauge fields have already become non-finite. Treat the primitive failures as downstream evidence unless a split run shows otherwise.

## Active Discriminator: `resgauge_lapse_only_hi2n`

Deck:

```text
inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_lapse_only_hi2n_aurora.athinput
```

Changes relative to `resgauge_full_hi2n`:

```text
evolve_lapse_residual = true
evolve_shift_residual = false
shift_Gamma = 0.0
shift_advect = 0.0
shift_eta = 0.0
```

Submitted:

```text
job id: 8529810
queue: debug
nodes: 2
ATHENA_WALLTIME: 00:56:00
```

Purpose:

If this case stays finite past the full baseline's first bad time (`t = 3.5`), the residual shift/Gamma-driver path becomes the leading suspect. If it fails on the same schedule, inspect residual 1+log lapse and non-gauge Z4c residual RHS first.

Result:

```text
Exit_status: -29
reason: PBS walltime kill after bad data; histories were written
run dir: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_lapse_only_hi2n
stdout: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_rg_lapse2n.o8529810
summary csv: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_only_hi2n/history_summary.csv
metric/gauge plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_only_hi2n/metric_gauge_trends.png
constraint plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_only_hi2n/z4c_constraint_trends.png
MHD integral plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_only_hi2n/mhd_integral_trends.png
```

History summary:

- `mhd.hst`, `user.hst`, and `z4c.user.hst` each have last fully finite row at `t = 3.1`.
- First bad row is `t = 3.2`.
- Last finite `user.hst` row at `t = 3.1` had:
  - `alpha-min = 6.7783438940093022e-01`
  - `alpha-max = 1.3984980441035340e+00`
  - `detg-min = 5.2533232330304336e-01`
  - `Theta-max = 8.6815016941814793e+00`
  - `Khat-max = 3.3930303110811352e+01`
  - `bad-metric = 0`
  - `alpha-res = 4.2347467353945939e-01`
  - `beta-res = 0`
  - `B-res = 0`
  - `Gam-res = 2.1930928963033939e+01`
- Last finite `z4c.user.hst` row at `t = 3.1` had:
  - `C-norm2 = 1.2935627746608941e+05`
  - `H-norm2 = 9.2465078547211917e+04`
  - `M-norm2 = 3.6889987473937312e+04`
  - `Theta-norm = 3.3139775153009132e-01`

Interpretation:

Residual Gamma-driver shift is not the sole trigger: the run fails even with residual shift and auxiliary shift disabled. Because this lapse-only case fails earlier than the full residual-gauge case, the residual lapse path or non-gauge residual Z4c RHS must be checked next.

Next control:

Run the same high-resolution deck with both residual lapse and residual shift disabled. If that control is stable, the residual lapse path is strongly implicated. If it also fails, the issue is in the non-gauge residual Z4c/AMR/coupling path rather than residual gauge evolution alone.

## Active Control: `schwarzschild_frozengauge_hi2n`

Deck:

```text
inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_frozengauge_hi2n_aurora.athinput
```

Changes relative to `resgauge_full_hi2n`:

```text
evolve_gauge_residual = false
evolve_lapse_residual = false
evolve_shift_residual = false
shift_Gamma = 0.0
shift_advect = 0.0
shift_eta = 0.0
```

Submitted:

```text
job id: 8529850
queue: debug
nodes: 2
ATHENA_WALLTIME: 00:50:00
```

Purpose:

This checks whether the same high-resolution reduced-domain setup is stable when the gauge is prescribed by the analytic background. It is the direct control for the failing residual-lapse-only and full-residual-gauge cases.

Result:

```text
Exit_status: 0
run dir: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_frozengauge_hi2n
stdout: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_rg_frozen2n.o8529850
summary csv: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_frozengauge_hi2n/history_summary.csv
metric/gauge plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_frozengauge_hi2n/metric_gauge_trends.png
constraint plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_frozengauge_hi2n/z4c_constraint_trends.png
MHD integral plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_frozengauge_hi2n/mhd_integral_trends.png
```

History summary:

- All histories stayed finite through `t = 4.9875`.
- Final `user.hst` row had:
  - `alpha-min = 6.7783438940456497e-01`
  - `alpha-max = 9.8029381188633180e-01`
  - `detg-min = 1.0300085304991129e+00`
  - `Theta-max = 2.6636636034898090e-05`
  - `Khat-max = 5.9587050041447198e-01`
  - `bad-metric = 0`
  - `alpha-res = 0`
  - `beta-res = 0`
  - `B-res = 0`
  - `Gam-res = 1.0404385723955099e-01`
- Final `z4c.user.hst` row had:
  - `C-norm2 = 5.1588795104293211e-05`
  - `H-norm2 = 4.2973116265879672e-05`
  - `M-norm2 = 8.5921565014272639e-06`
  - `Theta-norm = 2.1165579300005482e-08`

Interpretation:

The non-gauge residual Z4c system on this high-resolution reduced-domain deck is stable when gauge residuals are frozen. Combined with the lapse-only failure at `t = 3.2`, the residual lapse evolution is the leading failing path. The next discriminator should split the residual lapse RHS by disabling lapse advection.

## Active Discriminator: `resgauge_lapse_noadvect_hi2n`

Deck:

```text
inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_lapse_noadvect_hi2n_aurora.athinput
```

Changes relative to `resgauge_lapse_only_hi2n`:

```text
lapse_advect = 0.0
```

Submitted:

```text
job id: 8529879
queue: debug
nodes: 2
ATHENA_WALLTIME: 00:50:00
```

Purpose:

This splits the residual lapse equation. If this stabilizes the run relative to `resgauge_lapse_only_hi2n`, the advective lapse term or its background subtraction is implicated. If it still fails near `t = 3.2`, the non-advective 1+log source term involving `alpha Khat` is the next target.

Result:

```text
Exit_status: 143
run dir: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_lapse_noadvect_hi2n
summary csv: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_hi2n/history_summary.csv
metric/gauge plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_hi2n/metric_gauge_trends.png
constraint plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_hi2n/z4c_constraint_trends.png
MHD integral plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_hi2n/mhd_integral_trends.png
```

History summary:

- `mhd.hst`, `user.hst`, and `z4c.user.hst` each have last fully finite row at `t = 3.1`.
- First bad row is `t = 3.2`.
- Last finite `user.hst` row at `t = 3.1` had:
  - `alpha-min = 6.7783438940385110e-01`
  - `alpha-max = 1.5513874499749030e+00`
  - `detg-min = 4.2222630257835481e-01`
  - `Theta-max = 1.5369527171257660e+01`
  - `Khat-max = 4.8937877034970519e+01`
  - `bad-metric = 0`
  - `alpha-res = 5.7633291670526987e-01`
  - `beta-res = 0`
  - `B-res = 0`
  - `Gam-res = 3.5174875651295899e+01`
- Last finite `z4c.user.hst` row at `t = 3.1` had:
  - `C-norm2 = 4.4779513457042928e+05`
  - `H-norm2 = 3.5403124909513799e+05`
  - `M-norm2 = 9.3760064319022436e+04`
  - `Theta-norm = 8.5653862404220293e-01`

Interpretation:

Lapse advection is not the root trigger. With residual shift disabled and `lapse_advect = 0`, the run still fails at the same code time as lapse-only. The next target is the non-advective residual 1+log source term, especially the subtraction of `f alpha Khat` between full and background states.

## Completed Discriminator: `resgauge_lapse_sourceoff_hi2n`

Deck:

```text
inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_lapse_sourceoff_hi2n_aurora.athinput
```

Changes relative to `resgauge_lapse_noadvect_hi2n`:

```text
lapse_oplog = 0.0
lapse_harmonic = 0.0
```

The initial residual lapse is still retained and `evolve_lapse_residual = true`, but the explicit non-advective lapse gauge driver RHS is disabled. With `lapse_advect = 0`, this removes the residual full-minus-background `-f alpha Khat` source while leaving the same high-resolution AMR/MHD/Z4c setup.

Verification:

```bash
./build_sym_debug/src/athena \
  -i inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_lapse_sourceoff_hi2n_aurora.athinput \
  -d /tmp/athena_rg_lapse_sourceoff_smoke \
  mesh/nx1=32 mesh/nx2=32 mesh/nx3=32 \
  meshblock/nx1=16 meshblock/nx2=16 meshblock/nx3=16 \
  mesh_refinement/refinement=none mesh_refinement/max_nmb_per_rank=16 \
  time/nlim=0 time/tlim=0.0 output1/dt=100 output2/dcycle=100000
```

Result: local initialization smoke passed.

Submitted:

```text
job id: 8533586
queue: debug
nodes: 2
ATHENA_WALLTIME: 00:50:00
```

Result:

```text
Exit_status: 0
run dir: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_lapse_sourceoff_hi2n
stdout: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_rg_lsrc2n.o8533586
summary csv: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_sourceoff_hi2n/history_summary.csv
metric/gauge plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_sourceoff_hi2n/metric_gauge_trends.png
constraint plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_sourceoff_hi2n/z4c_constraint_trends.png
MHD integral plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_sourceoff_hi2n/mhd_integral_trends.png
```

History summary:

- `mhd.hst`, `user.hst`, and `z4c.user.hst` all stayed finite through the wall-clock stop at `t = 4.91`.
- Final `user.hst` row had:
  - `alpha-min = 6.7783438940202756e-01`
  - `alpha-max = 9.8029373792253893e-01`
  - `chi-min = 7.7164177688327229e-01`
  - `detg-min = 1.0333096130021850e+00`
  - `Theta-max = 2.8057533270408391e-05`
  - `Khat-max = 5.9587050053025070e-01`
  - `bad-metric = 0`
  - `alpha-res = 9.7281915129032726e-06`
  - `beta-res = 0`
  - `B-res = 0`
  - `Gam-res = 2.7808403805411192e-01`
- Final `z4c.user.hst` row had:
  - `C-norm2 = 1.1294389773789130e-04`
  - `H-norm2 = 1.0377580742594120e-04`
  - `M-norm2 = 9.0843174384341014e-06`
  - `Theta-norm = 2.1462797236600310e-08`

Interpretation:

This is the strongest discriminator so far. The same high-resolution reduced-domain deck failed at `t = 3.2` with residual lapse active and `lapse_advect = 0`; it stays finite to `t = 4.91` when only the non-advective lapse driver source is disabled. The failure is therefore localized to the residual lapse driver source, not to Gamma-driver shift, lapse advection, generic AMR/MHD/Z4c evolution, or the presence of an initial residual lapse by itself.

The next code-level target is the full-minus-background implementation of the lapse driver source,

```text
rhs.alpha = -f_full alpha_full Khat_full - (-f_bg alpha_bg Khat_bg)
```

with the current 1+log settings `f = lapse_oplog * lapse_harmonicf + lapse_harmonic * alpha`. Add a gated per-cell diagnostic for the full, background, and residual components of this term, including max/argmax and local `alpha`, `Khat`, `Theta`, `chi`, `detg`, and density. Then test whether the source can be regularized or reformulated in terms of residual quantities without driving large constraint growth.

## Completed Diagnostic: `resgauge_lapse_noadvect_diag_hi2n`

Code changes:

- Kept the existing `metric_diag_history` gate.
- Replaced less useful metric-shape columns in the gated user history with residual-lapse source columns:
  - `src-full = max|-f_full alpha_full Khat_full|`
  - `src-bg = max|-f_bg alpha_bg Khat_bg|`
  - `src-res = max|src_full - src_bg|`
  - `aK-full = max|alpha_full Khat_full|`
  - `aK-bg = max|alpha_bg Khat_bg|`
  - `Khat-res = max|Khat_full - Khat_bg|`
- Kept the total user history count within the existing `NHISTORY_VARIABLES = 20` capacity. No global history/reduction capacity change is required.
- Updated `analysis/tde_star_profile/aurora/residual_gauge_history_summary.py` to summarize and plot the new source columns.

Verification:

- Local build: `cmake --build build_sym_debug -j 4` passed.
- Local one-step smoke passed and confirmed unique history labels: `src-full`, `src-bg`, `src-res`, `aK-full`, `aK-bg`, `Khat-res`.
- Aurora build:

  ```text
  /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks/src/athena
  timestamp: 2026-06-10 02:32:31 UTC
  ```

Submitted:

```text
job id: 8533675
queue: debug
nodes: 2
ATHENA_WALLTIME: 00:50:00
```

Result:

```text
Exit_status: 1
run dir: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_lapse_noadvect_diag_hi2n
stdout: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_rg_lsrcdiag.o8533675
summary csv: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_diag_hi2n/history_summary.csv
lapse-source plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_diag_hi2n/lapse_source_trends.png
metric/gauge plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_diag_hi2n/metric_gauge_trends.png
constraint plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_diag_hi2n/z4c_constraint_trends.png
MHD integral plot: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_lapse_noadvect_diag_hi2n/mhd_integral_trends.png
```

History summary:

- `mhd.hst`, `user.hst`, and `z4c.user.hst` were finite through `t = 3.1`.
- First bad row was `t = 3.2`.
- The run later aborted with a mesh-refinement allocation error after bad data drove many refinements:

  ```text
  Number of MeshBlocks in this rank on new tree = 224/225 exceeds max_nmb_per_rank = 192
  ```

  This is downstream of the lapse/Z4c blow-up and is not the primary failure.

Selected source-growth rows:

| time | `src-res` | `src-bg` | `Khat-res` | `Theta-max` | `alpha-res` | `C-norm2` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | `6.02e-4` | `8.08e-1` | `3.08e-4` | `0` | `9.73e-6` | `1.04e-1` |
| 1.0 | `4.91e-3` | `8.08e-1` | `2.52e-3` | `8.94e-4` | `1.67e-5` | not listed here |
| 2.0 | `2.10e-1` | `8.08e-1` | `1.08e-1` | `1.99e-2` | `1.22e-3` | not listed here |
| 2.5 | `1.75e0` | `8.08e-1` | `9.02e-1` | `2.41e-1` | `5.43e-3` | not listed here |
| 3.0 | `2.76e1` | `8.08e-1` | `1.37e1` | `4.45e0` | `1.03e-1` | not listed here |
| 3.1 | `1.21e2` | `8.08e-1` | `4.89e1` | `1.54e1` | `5.76e-1` | `4.48e5` |

Final finite diagnostics at `t = 3.1`:

- `alpha-min = 6.7783438940385110e-01`
- `alpha-max = 1.5513874499749030e+00`
- `detg-min = 4.2222630257835481e-01`
- `Theta-max = 1.5369527171257660e+01`
- `Khat-max = 4.8937877034970519e+01`
- `alpha-res = 5.7633291670526987e-01`
- `Gam-res = 3.5174875651295899e+01`
- `src-full = 1.2102647270369150e+02`
- `src-bg = 8.0780303355820160e-01`
- `src-res = 1.2102909046590740e+02`
- `aK-full = 6.0513236351845727e+01`
- `aK-bg = 4.0390151677910080e-01`
- `Khat-res = 4.8939219409949999e+01`
- `C-norm2 = 4.4779513457042928e+05`
- `H-norm2 = 3.5403124909513799e+05`
- `M-norm2 = 9.3760064319022436e+04`

Interpretation:

The residual lapse source instability is dominated by growth in the full/residual `Khat` term, not by a drifting analytic background source. The background contribution stays essentially fixed at `src-bg ~= 0.808`, while `src-res` grows by five orders of magnitude before the first bad row. This matches the source-off discriminator: removing the non-advective lapse driver prevents the lapse/Khat feedback loop and keeps constraints small through the same time interval.

Current classification:

- Residual Gamma-driver shift: not the root trigger.
- Lapse advection: not the root trigger.
- Generic AMR/MHD/Z4c evolution: not the root trigger on this deck; frozen gauge and source-off stay stable.
- Non-advective residual lapse driver, through the full-minus-background `alpha Khat` source: leading failure path.

Recommended next test:

Run a regular `debug`, 2-node, source-regularization discriminator before any long production run. The least invasive options are:

1. Keep residual lapse evolved but ramp `lapse_oplog` from `0` to `2` over a controlled time window, e.g. 20-50M, to avoid the early `Khat` feedback impulse.
2. Try light residual-consistent constraint damping, e.g. `damp_kappa1 = 0.05` or `0.1`, with the lapse source active, to test whether damping suppresses the `Khat` growth before it feeds the lapse.
3. If this is intended as a code fix rather than a production workaround, add a separate residual-gauge option that uses a regularized source based on residual quantities and verify that the background source is still analytically factored out.

## Planned Discriminator Overrides

Use the same base deck unless a result shows the small AMR deck does not reproduce the early failure. Submit one at a time.

| Case | Purpose | Overrides |
| --- | --- | --- |
| `resgauge_full_2n` | Reproduce the small residual-gauge failure. | none |
| `resgauge_lapse_only_2n` | Test residual lapse with residual shift/B zeroed. | `z4c/evolve_shift_residual=false z4c/shift_Gamma=0.0 z4c/shift_eta=0.0 z4c/shift_advect=0.0` |
| `resgauge_shift_only_2n` | Test residual Gamma-driver shift while lapse is fixed/preserved. | `z4c/evolve_lapse_residual=false z4c/preserve_lapse_residual=true z4c/evolve_shift_residual=true` |
| `resgauge_no_gauge_advect_2n` | Distinguish advection/Lie gauge terms from driver source terms. | `z4c/lapse_advect=0.0 z4c/shift_advect=0.0` |
| `resgauge_light_damping_2n` | Check whether residual-consistent damping delays growth. | `z4c/damp_kappa1=0.1` |
| `resgauge_static_refine_2n` | Check whether AMR motion/prolongation participates. | `mesh_refinement/refinement=static problem/amr_rho_slope_refine=false` |

## Analysis Checklist

For each run:

1. Locate `.hst` files under `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/<case>`.
2. Determine last finite row and first non-finite row.
3. Inspect `alpha-min`, `alpha-res`, `beta-res`, `B-res`, `Gam-res`, `chi-min`, `detg-min`, `Khat-max`, `Theta-max`, and `bad-metric`.
4. Compare common-time histories against the failed 64-node residual-gauge run and the frozen-gauge baseline.
5. Classify according to the decision rules in `residual_gauge_debug_goal.md`.
