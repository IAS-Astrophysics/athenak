# Goal Prompt: Fix Zero-Feedback Symmetry Breaking

Use this prompt as the complete Goal Mode objective.

Primary goal: Fix the symmetry breaking in the AthenaK TOV/Z4c
`schwarzschild_zero_feedback_smr_dense` discriminator first. The working
hypothesis is that fixing this case will also fix the full coupled solver, but
that must be verified, not assumed.

The goal is not to broadly improve diagnostics. The goal is to identify and fix
the first code path that makes the zero-feedback run lose reflection symmetry,
then rerun the full coupled case to test whether the same fix resolves it.

Repository:

```text
/home/hzhu/athenak_tde
```

## Current Evidence

The dense discriminator matrix is complete and recorded in:

```text
analysis/tde_star_profile/aurora/sector_isolation_run_record.md
```

Important completed cases:

- `minkowski_static_uniform_dense`, job `8522794`:
  uniform static Minkowski baseline; MHD `dens`, `velx`, `vely`, and `velz`
  exactly symmetric in usable masks.
- `minkowski_static_smr_dense`, job `8522837`:
  SMR static Minkowski baseline; density exactly symmetric in all masks.
  High-density velocity differences are tiny, max Linf abs around `1e-11`;
  `z4c_Theta` has a refinement-only baseline around Linf abs `1.44e-5`.
- `schwarzschild_infall_smr_dense`, job `8522851`:
  full coupled infall reproduces coherent MHD asymmetry. At cycle 20/time
  `0.25`, high-density `rho > 1e-8` density asymmetry is:
  Linf abs `3.40e-8`, local-relative Linf `1.97e-3`,
  peak-relative Linf `3.55e-4`, L2 peak-relative `8.14e-5`.
- `schwarzschild_zero_feedback_smr_dense`, job `8522863`:
  matter feedback to Z4c disabled, but the MHD break remains nearly identical:
  density Linf abs `3.42e-8`, local-relative Linf `1.97e-3`,
  peak-relative Linf `3.57e-4`, L2 peak-relative `8.15e-5`.
  Z4c/ADM mirror differences stay small: `z4c_Theta` Linf abs `1.46e-11`,
  `z4c_Gamy/Gamz` about `2.91e-11`/`5.82e-11`, checked ADM fields zero.
- `schwarzschild_fixed_mhd_tmunu_smr_dense`, job `8522867`:
  frozen MHD with retained initial matter source stays symmetric.
- `schwarzschild_fixed_mhd_refresh_tmunu_smr_dense`, job `8522883`:
  frozen MHD with refreshed fixed-fluid `Tmunu` also stays symmetric.

Current classification:

- The necessary path is not matter feedback into Z4c.
- The break is isolated to fluid evolution running in the coupled Z4c/ADM
  context.
- Most likely classes are:
  - ADM-to-GRMHD data path,
  - dynamic x3 reconstruction / Riemann inputs,
  - fluid ghost or boundary/refinement communication feeding x3 fluxes,
  - coupled task ordering or stale/unsynchronized ADM/primitive data consumed
    by `DynGRMHD_CalcFluxes`.

The first target is the zero-feedback case because it breaks without matter
feedback and should be the smallest decisive coupled reproducer.

## Hard Constraints

- Do not change `NGHOST` behavior. The `NGHOST=3` choice is intentional.
- Do not revert unrelated dirty-tree changes.
- Keep diagnostics input- or env-gated.
- Use `source ~/athenak_env` before Python analysis.
- Do not claim “roundoff” based only on local relative error; always report:
  absolute error, local relative error, peak-relative error, and masked
  high-density metrics.
- Use local serial smoke tests only for build/input parsing and diagnostic
  sanity checks.
- Use Aurora GPU+MPI for decisive results.
- Submit Aurora jobs one at a time.
- Preserve current working evidence; update
  `analysis/tde_star_profile/aurora/sector_isolation_run_record.md` as new
  evidence is produced.
- Commit coherent checkpoints: diagnostics, candidate fixes, run-record updates.

## Goal Mode Operating Loop

Work in short, evidence-driven iterations:

1. Inspect the current repo state and the latest run record.
2. Build or run only the smallest diagnostic needed to answer the next
   localization question.
3. Submit at most one Aurora job at a time.
4. While an Aurora job is queued or running, conserve tokens:
   - sleep or use the monitor script at roughly 10 minute intervals,
   - do not repeatedly inspect unchanged queue state,
   - when a job completes, inspect logs and postprocessed metrics before
     deciding the next action.
5. Update the run record after each decisive result.
6. Commit coherent checkpoints separately:
   - prompt/docs,
   - diagnostic instrumentation,
   - candidate solver fix,
   - run-record updates.

If the current worktree contains partial diagnostic edits, inspect them before
continuing. Either finish and commit them if they are useful, or remove only
those generated scratch edits after confirming they are unrelated. Do not revert
user changes or unrelated dirty files.

Prefer a narrow diagnosis over a speculative fix. The first successful
iteration should answer where the zero-feedback path first diverges:

- primitive state before x3 reconstruction,
- ghost/refinement communication,
- ADM/metric values consumed by GRMHD,
- reconstructed x3 left/right states,
- Riemann solve / flux computation,
- task ordering or stale data.

## Existing Useful Infrastructure

Aurora executable:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks/src/athena
```

Important scripts:

```text
analysis/tde_star_profile/aurora/submit_aurora_roadmap.sh
analysis/tde_star_profile/aurora/submit_aurora_case.pbs
analysis/tde_star_profile/aurora/monitor_symmetry_job.sh
analysis/tde_star_profile/aurora/z4c_bg_validation_metrics.py
```

Metric output roots:

```text
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/<case>
/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/<case>
```

Main decks:

```text
inputs/tde/aurora/z4c_tov_ks_schwarzschild_infall_zero_feedback_smr_dense_aurora.athinput
inputs/tde/aurora/z4c_tov_ks_schwarzschild_infall_sym_smr_dense_aurora.athinput
inputs/tde/aurora/z4c_tov_ks_minkowski_static_sym_smr_dense_aurora.athinput
```

Key source files:

```text
src/pgen/z4c_tov_ks.cpp
src/dyn_grmhd/dyn_grmhd.cpp
src/dyn_grmhd/dyn_grmhd.hpp
src/dyn_grmhd/dyn_grmhd_fluxes.cpp
src/mhd/mhd_tasks.cpp
src/mhd/mhd_update.cpp
src/bvals/flux_correct_cc.cpp
src/coordinates/adm.cpp
src/z4c/z4c_adm.cpp
src/z4c/z4c_calcrhs.cpp
```

Existing toggles / diagnostics to inspect before adding new ones:

- `zero_tmunu_feedback`
- `refresh_tmunu_when_fixed`
- `ATHENA_SYM_DEBUG`
- `ATHENA_FLUX_DEBUG`
- `SymmetryFluxDebugProbe`
- `SymmetryRKDebugProbe`
- flux correction debug probes

## Required Workflow

### 1. Start Clean

Inspect the repo state first:

```bash
git status --short
git log --oneline -12
```

If there are dirty changes:

- Identify whether they are relevant to this symmetry work.
- Do not revert unrelated user changes.
- Commit relevant intended changes or remove generated scratch artifacts.
- Only proceed once the repo state is understood.

### 2. Reproduce and Quantify Current Zero-Feedback Failure

Do not assume prior numbers are enough. Confirm current metrics exist for
`schwarzschild_zero_feedback_smr_dense`; if not, rerun it:

```bash
CASE_FILTER=schwarzschild_zero_feedback_smr_dense \
  analysis/tde_star_profile/aurora/submit_aurora_roadmap.sh
```

Monitor with:

```bash
analysis/tde_star_profile/aurora/monitor_symmetry_job.sh JOB_ID \
  schwarzschild_zero_feedback_smr_dense
```

Summarize:

- `xy` y-reflection and `xz` z-reflection separately.
- MHD `dens`, `velx`, `vely`, `velz`.
- Masks `rho > 1e-8`, `1e-10`, `1e-12`, and `all`.
- Absolute L2/Linf, local-relative L2/Linf, peak-relative L2/Linf.
- Growth versus time, not just final output.

Baseline comparison:

- Compare against `minkowski_static_smr_dense`.
- Treat the zero-feedback density growth as the bug to fix.

### 3. Localize Before Fixing

The next decisive probe should be immediately before x3 reconstruction/Riemann
input inside `DynGRMHD_CalcFluxes`.

Add the smallest input/env-gated diagnostics needed to compare parity pairs at
the same physical mirror locations where the slice metrics show the final
break, around:

```text
x ~= 20.03125
y/z ~= +/-0.03125 for density max
```

Start with the existing `ATHENA_FLUX_DEBUG` target machinery if possible.

Probe these quantities before x3 reconstruction / Riemann solve:

- Primitive cell states:
  - `dens`
  - pressure/internal energy if available
  - `velx`, `vely`, `velz` with correct parity
  - magnetic fields if active / available
- Face/reconstructed x3 inputs:
  - left/right primitive states at the face
  - left/right B fields
  - x3 face mass flux inputs
- ADM / metric fields consumed by GRMHD at those exact target points:
  - lapse
  - shift
  - spatial metric
  - extrinsic curvature
  - determinant / derived quantities used by flux calculation
- Ghost-zone values near the target mirror pair.
- Values before and after boundary communication if the first probe suggests
  ghost/refinement data are implicated.

For every printed pair, report:

- coordinates,
- meshblock/rank/level,
- side of mirror pair,
- parity-adjusted difference,
- absolute difference,
- local relative difference,
- peak-relative difference when meaningful.

Keep output volume limited:

- Require an env var or input flag.
- Restrict to one or a few target pairs.
- Print only selected cycles/stages unless debugging requires more.

### 4. Isolate the First Asymmetric Data Path

Work backward from `dyngrflux_x3`:

1. Are primitive cell-centered states symmetric before reconstruction?
2. Are ghost cells symmetric before reconstruction?
3. Are ADM/metric fields symmetric at the actual GRMHD consumption points?
4. Are reconstructed left/right x3 states symmetric after parity adjustment?
5. Does the Riemann solve receive symmetric inputs but produce asymmetric flux?
6. Does asymmetry appear before or after boundary communication / prolongation /
   restriction?
7. Does the same issue occur in the full coupled infall, or only zero-feedback?

Do not jump to broad refactors. Add the narrowest diagnostic necessary, run the
smallest meaningful case, inspect, and then decide the next edit.

Use this decision tree:

- If cell-centered primitive states are already asymmetric before x3
  reconstruction, inspect the preceding MHD update, boundary refresh,
  prolongation/restriction, and ghost exchange.
- If primitive states are symmetric but ghost-zone values are asymmetric,
  inspect mesh refinement and boundary communication for MHD primitive or
  conserved variables.
- If primitives and ghosts are symmetric but ADM/metric values at the exact
  GRMHD consumption points are asymmetric, inspect Z4c-to-ADM conversion,
  face metric construction, data synchronization, and task ordering.
- If primitives and ADM are symmetric but reconstructed x3 left/right states
  are asymmetric, inspect x3 reconstruction stencils, indexing, and face-state
  pairing across the mirror pair.
- If reconstructed states and ADM are symmetric but x3 fluxes are asymmetric,
  inspect the Riemann solver, floor application, determinant/normalization, and
  any branch that could differ between mirror partners.
- If the zero-feedback case is fixed, rerun the full coupled infall before
  claiming the full solver is fixed.

### 5. Candidate Fix Policy

Only implement a fix after identifying the first asymmetric data boundary.

Acceptable candidate fix classes include:

- Correct parity handling for vector/tensor components in a coupled boundary,
  prolongation, restriction, or metric data path.
- Correct stale or unsynchronized ADM data used by GRMHD flux calculation.
- Correct ordering between Z4c ADM production and GRMHD flux consumption.
- Correct coordinate/face-center selection for x3 metric/ADM states.
- Correct x3 reconstruction input indexing or face-state pairing across
  mirrored cells.

Avoid:

- Changing `NGHOST`.
- Masking the metric by symmetrizing output-only data.
- Adding global parity-enforcement as a substitute for fixing the broken path.
- Fixing only the metric script while the solver remains asymmetric.
- Treating atmosphere-relative noise as the bug unless the high-density masks
  also show the same effect.

### 6. Verification Gates

After any candidate fix:

1. Build locally and run input parsing smoke tests.
2. Rebuild the Aurora GPU executable if source changes affect it.
3. Rerun `schwarzschild_zero_feedback_smr_dense` first.
4. The fix is successful only if high-density density asymmetry in zero-feedback
   no longer grows above the SMR Minkowski baseline. Specifically compare final
   and time-history metrics against the old broken values:
   - old density high-density Linf abs `~3.42e-8`,
   - old Linf peak-relative `~3.57e-4`,
   - old Linf local-relative `~1.97e-3`.
5. Then rerun `schwarzschild_infall_smr_dense`.
6. The full solver is considered fixed only if the same high-density density
   and velocity asymmetry no longer grows above baseline.
7. Also rerun or spot-check `minkowski_static_smr_dense` if the fix touches
   shared MHD boundary/reconstruction/ADM consumption code.

Use the metrics script:

```bash
source ~/athenak_env
python3 analysis/tde_star_profile/aurora/z4c_bg_validation_metrics.py \
  --run-dir RUN_DIR \
  --case CASE_NAME \
  --output-root /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post
```

Use the monitor script for queued jobs:

```bash
analysis/tde_star_profile/aurora/monitor_symmetry_job.sh JOB_ID CASE_NAME
```

If manual polling is needed to conserve token usage, use a loop with a long
sleep interval, for example:

```bash
while qstat JOB_ID >/dev/null 2>&1; do
  date
  qstat JOB_ID
  sleep 600
done
```

Then inspect the run directory, logs, metrics CSV/JSON, and generated plots.

### 7. Deliverables

Produce and commit:

1. A short diagnosis note in
   `analysis/tde_star_profile/aurora/sector_isolation_run_record.md`.
2. Any minimal diagnostic code/input changes used to identify the source.
3. The actual solver fix, if found.
4. Local smoke-test results.
5. Aurora run IDs and metric/plot paths for:
   - fixed `schwarzschild_zero_feedback_smr_dense`,
   - follow-up `schwarzschild_infall_smr_dense`,
   - any baseline rerun needed by the touched code path.
6. A final classification update:
   - ADM-to-GRMHD path,
   - x3 reconstruction/Riemann path,
   - boundary/refinement/ghost communication,
   - task ordering/stale data,
   - or another concrete source proven by diagnostics.

## Success Criteria

The goal is complete only when current evidence proves:

- The zero-feedback symmetry break is fixed or the exact first asymmetric data
  path has been isolated strongly enough that a specific code fix is obvious.
- If a code fix is implemented, `schwarzschild_zero_feedback_smr_dense` passes
  the high-density symmetry criteria against the SMR Minkowski baseline.
- The full coupled infall case is rerun after the zero-feedback fix and its
  result is documented.
- All new diagnostics remain gated.
- The worktree is clean.
- The run record contains the current evidence, job IDs, metric paths, and
  classification.
