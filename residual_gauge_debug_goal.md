# Goal-Mode Prompt: Debug Residual-Gauge Z4c Instability With 2-Node Aurora Debug Jobs

Repository:

```text
/home/hzhu/athenak_tde
```

Project storage and charge project:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation
MHDTidal
```

Use only the regular Aurora `debug` queue with `select=2` for new discriminator jobs. Do not use 64-node `debug-scaling` jobs until the early-time residual-gauge instability is isolated with small jobs.

Before Python analysis, run:

```bash
source ~/athenak_env
```

## Objective

Track down the early NaN/constraint blow-up introduced by enabling background-subtracted residual gauge evolution in the Schwarzschild head-on AMR64 run.

The immediate goal is to determine whether the failure is caused by:

- residual Gamma-driver shift evolution,
- residual lapse evolution,
- gauge advection choices,
- constraint-damping or gauge coupling,
- AMR prolongation/restriction or ghost-zone communication of residual gauge fields,
- or a bug in the full-minus-background residual gauge RHS.

Use short 2-node debug-queue jobs that reach at least `t = 5` and preferably `t = 8`, because the bad 64-node residual-gauge run was already invalid between `t = 3.0` and `t = 3.5`.

## Critical Constraints

- Do not revert unrelated dirty-tree changes.
- Do not change `NGHOST` behavior.
- Keep diagnostics input- or environment-gated.
- Do not submit 64-node production or `debug-scaling` jobs during this debug pass.
- Do not use KO `diss > 0.5` unless the user explicitly asks.
- Treat the analytic background as constraint-free.
- If inspecting or changing constraint damping, make sure damping is applied in the background-subtracted residual system and does not inject analytic-background truncation error into the residual fields.
- Keep exact run records: case name, input deck, job ID, queue, node count, executable, build path, run directory, post directory, and key metric results.
- If a run produces NaNs, do not use its restart files except for diagnostic inspection.

## Current Evidence

Failed reference run:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_headon_density_amr_64c_resgauge
```

Frozen-gauge comparison run:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_headon_density_amr_64c
```

Failed residual-gauge deck:

```text
inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_headon_density_amr_64c_resgauge_aurora.athinput
```

Important settings in the failed residual-gauge run:

```text
use_analytic_background = true
evolve_gauge_residual = true
diss = 0.5
damp_kappa1 = 0
damp_kappa2 = 0
shift_Gamma = 1.0
shift_eta = 2.0
```

Known failure timing:

- `mhd.hst`: last fully finite row at `t = 3.0`; first bad row at `t = 3.5`.
- `z4c.user.hst`: last fully finite row at `t = 3.0`; first bad row at `t = 3.5`.
- `user.hst`: `alpha_min` normal through `t = 3.0`, negative at `t = 3.5`, and `-inf` by `t = 4.0`.

Constraint growth before lapse collapse:

```text
t=1.0   C ~ 3.6e-03
t=1.5   C ~ 6.0e-02
t=2.0   C ~ 8.0e+00
t=2.5   C ~ 4.3e+02
t=3.0   C ~ 6.0e+04
t=3.5   NaN
```

The frozen-gauge comparison stayed finite much longer and had constraints around `1e-4` at the same early times. The MHD conserved integrals remain close until after the Z4c constraints have already grown. This points first to the Z4c/gauge path, not to the fluid update, late physical-boundary reflection, or black-hole encounter.

## Current Repo State To Use

Relevant split residual-gauge controls have already been added in this working tree:

```text
<z4c>/evolve_lapse_residual
<z4c>/evolve_shift_residual
<z4c>/preserve_lapse_residual
```

These controls default to compatibility behavior unless the input deck overrides them.

Touched implementation files include:

```text
src/z4c/z4c.hpp
src/z4c/z4c.cpp
src/z4c/z4c_calcrhs.cpp
src/pgen/z4c_tov_ks.cpp
analysis/tde_star_profile/aurora/residual_gauge_history_summary.py
```

Existing run log:

```text
analysis/tde_star_profile/aurora/residual_gauge_debug_run_log.md
```

Keep updating this log as jobs are submitted and analyzed.

## Build And Helper Paths

Aurora build path:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks
```

Athena executable:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks/src/athena
```

Build command:

```bash
cmake --build /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks -j 8
```

Submission helper:

```text
analysis/tde_star_profile/aurora/prepare_instability_debug_2n_qsub.sh
```

PBS wrapper used by the helper:

```text
analysis/tde_star_profile/aurora/submit_aurora_case.pbs
```

Default storage locations:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit
```

## Existing Two-Node Result

The first small 2-node deck was:

```text
inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_2n_aurora.athinput
```

It uses a smaller Schwarzschild AMR setup with the star at `x = 20`, `tlim = 8`, dense history output, residual gauge enabled, `diss = 0.5`, zero constraint damping, and metric/gauge history enabled.

Submitted jobs:

```text
8529730  resgauge_full_2n  setup failure due GPU memory over-allocation from max_nmb_per_rank=512
8529755  resgauge_full_2n  completed cleanly through t=8 after max_nmb_per_rank=64
```

Successful run directory:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/resgauge_full_2n
```

Summary CSV:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/resgauge_full_2n/history_summary.csv
```

Result: the small deck did not reproduce the early failure by `t = 8`. It was finite, with no first bad history row. Therefore the next priority is a higher-resolution 2-node deck that is closer to the failed AMR64 run while still fitting in regular `debug`.

## Next Immediate Job

Use this high-resolution 2-node reproduction deck first:

```text
inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_hi2n_aurora.athinput
```

This deck keeps the failed run's fine resolution and star setup more closely, but uses a reduced domain so it can run on 2 nodes:

```text
star center: x = 40
boost: vx = -0.1
base dx: 0.1
finest dx: 0.0125
num_levels: 4
tlim: 8
diss: 0.5
damp_kappa1: 0
evolve_gauge_residual: true
evolve_lapse_residual: true
evolve_shift_residual: true
```

Submit it with:

```bash
CASE_NAME=resgauge_full_hi2n \
INPUT_DECK=/home/hzhu/athenak_tde/inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_resgauge_full_hi2n_aurora.athinput \
JOB_NAME=z4c_rg_hi2n \
QUEUE=debug \
SELECT_RESOURCE=2 \
PBS_WALLTIME=01:00:00 \
ATHENA_WALLTIME=00:50:00 \
POSTPROCESS=0 \
analysis/tde_star_profile/aurora/prepare_instability_debug_2n_qsub.sh --submit
```

If this deck fails from GPU memory allocation, reduce only what is necessary to fit: `max_nmb_per_rank`, refinement extents, or output count. Do not change the physics settings unless necessary to get a runnable reproduction.

## Two-Node Debug Matrix

Submit one job at a time unless queue pressure is very low. Always analyze the previous case before submitting the next one.

Use the high-resolution 2-node deck as the base if it reproduces the failure. If it does not reproduce the failure, adjust the small reproduction deck first until the early blow-up appears on 2 nodes.

Run these discriminators after a failing 2-node baseline exists:

1. `resgauge_lapse_only_2n`
   - Purpose: test residual lapse evolution while residual shift and auxiliary shift are disabled.
   - Overrides:

     ```text
     z4c/evolve_shift_residual=false
     z4c/shift_Gamma=0.0
     z4c/shift_eta=0.0
     z4c/shift_advect=0.0
     ```

2. `resgauge_shift_only_2n`
   - Purpose: test residual Gamma-driver shift while lapse is held fixed or preserved.
   - Overrides:

     ```text
     z4c/evolve_lapse_residual=false
     z4c/preserve_lapse_residual=true
     z4c/evolve_shift_residual=true
     ```

3. `resgauge_no_gauge_advect_2n`
   - Purpose: distinguish driver source terms from advective gauge coupling.
   - Overrides:

     ```text
     z4c/lapse_advect=0.0
     z4c/shift_advect=0.0
     ```

4. `resgauge_light_damping_2n`
   - Purpose: determine whether residual-consistent damping delays or suppresses growth.
   - Overrides:

     ```text
     z4c/damp_kappa1=0.05
     ```

     Then try `0.1` only if the `0.05` result is informative but not decisive. Confirm damping remains background-subtracted.

5. `resgauge_static_refine_2n`
   - Purpose: test whether AMR motion or refinement operations participate.
   - Overrides:

     ```text
     mesh_refinement/refinement=static
     problem/amr_rho_slope_refine=false
     ```

6. `resgauge_boundary_or_refine_probe_2n`
   - Purpose: locate first bad values relative to refinement boundaries and physical boundaries.
   - Add or enable a gated diagnostic that reports the location of the worst cell for `alpha`, `chi`, `detg`, constraints, and residual gauge fields.

## Required Diagnostics

For each run, track at least:

- `min(alpha_full)` and `max(alpha_full)`,
- `max|alpha_res|`,
- `max|beta_res|`,
- `max|vB_res|`,
- `max|vGam_res|` or the relevant conformal-connection residual,
- `max|Khat_res|`,
- `max|Theta_res|`,
- `min(chi)`,
- `min(detg)` or the equivalent determinant validity metric,
- bad-metric cell count,
- Z4c constraint norms,
- MHD conserved integrals and density extrema.

If possible, add location diagnostics for the first or worst bad value:

```text
field name
cycle
time
block id or logical block coordinates
i,j,k
x,y,z
distance to nearest refinement boundary
distance to physical boundary
local rho
local alpha, chi, detg, Khat, Theta
local alpha_res, beta_res, B_res, vGam_res
```

Keep diagnostics cheap and gated.

## Monitoring Cadence

To conserve tokens, poll active jobs every 10 minutes unless the job is expected to finish sooner.

Use a loop like:

```bash
while qstat -u "$USER" | grep -q '<JOBID>'; do
  date -u
  qstat -f <JOBID> | egrep 'Job_Name|job_state|queue =|exec_host|resources_used.walltime|Exit_status|comment' || true
  sleep 600
done
date -u
qstat -xf <JOBID> | egrep 'Job_Name|job_state|queue =|resources_used.walltime|Exit_status|comment' || true
```

After completion, inspect stdout and histories:

```bash
tail -n 100 /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/<stdout-file>
find /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/<case> -maxdepth 1 -type f -name '*.hst' -print
```

Use the summary helper:

```bash
source ~/athenak_env
python3 analysis/tde_star_profile/aurora/residual_gauge_history_summary.py \
  --run-dir /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/<case> \
  --output /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/<case>/history_summary.csv
```

## Analysis Requirements

For each run:

1. Determine the last finite history row and first non-finite history row.
2. Record the first time `alpha_min` becomes nonpositive.
3. Record the first time `chi`, determinant, or bad-metric count becomes invalid.
4. Compare constraint norms against the failed 64-node residual-gauge run at common times.
5. Compare fluid conserved integrals against the frozen-gauge run to determine whether MHD changes before or after the gauge/constraint blow-up.
6. If the run fails, determine whether first bad values are near the star center, the black hole, a refinement boundary, a physical boundary, or spread globally.
7. Write CSVs and plots under:

   ```text
   /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/<case>
   ```

Trend plots should include:

- constraint norm vs time,
- `alpha_min` vs time,
- `max|alpha_res|` vs time,
- `max|beta_res|` vs time,
- `max|vB_res|` vs time,
- `max|vGam_res|` vs time,
- `Khat_max` and `Theta_max` vs time,
- `chi_min` and determinant minimum vs time.

## Decision Rules

Use these classifications:

- If the full residual-gauge 2-node case fails but `resgauge_lapse_only_2n` is stable, suspect residual shift or shift coupling first.
- If `resgauge_lapse_only_2n` fails, inspect residual 1+log lapse RHS and background subtraction.
- If `resgauge_shift_only_2n` fails, inspect Gamma-driver source terms, residual `vGam` construction, shift advection, and beta/B ghost communication.
- If disabling gauge advection stabilizes the run, inspect advective gauge terms and full-minus-background subtraction in those terms.
- If damping only delays failure without changing qualitative growth, damping is not the root fix.
- If fixed refinement stabilizes the run while AMR fails, inspect prolongation/restriction and ghost-zone consistency for residual gauge variables.
- If every residual-gauge split fails but frozen gauge is stable, inspect mathematical consistency between the residual gauge RHS, residual Z4c variables, algebraic projection, and recasting.
- If the higher-resolution 2-node case does not fail by `t = 8`, the reproducer is still inadequate. Move the 2-node deck closer to the failed AMR64 configuration before drawing physics conclusions.

## Deliverables

At the end of the goal-mode run, provide:

1. A concise diagnosis of the most likely failing path.
2. The exact run matrix and job IDs.
3. Input decks and code changes made.
4. Local and Aurora build or smoke-test status.
5. For each run, the first bad time or confirmation that it stayed finite through the target time.
6. CSV and plot paths for history trends.
7. A recommendation for the next code fix or production-safe setting.

Do not submit a new long 64-node run until the 2-node debug matrix identifies a stable residual-gauge configuration or a specific code fix has passed the short tests.
