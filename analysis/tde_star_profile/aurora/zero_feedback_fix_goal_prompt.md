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

## Latest Critical Result

The zero-feedback break has now been localized past fluxes, RK, C2P, and the
coordinate-source formula itself.

Latest decisive run:

```text
case: schwarzschild_zero_feedback_coordsrc_c0_n1
job: 8523105
run dir: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/schwarzschild_zero_feedback_coordsrc_c0_n1
post dir: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/schwarzschild_zero_feedback_coordsrc_c0_n1
stdout: /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_zero_fb_coordsrc.o8523105
```

It reproduced the same one-cycle high-density density break:

```text
time = 0.0125
rho > 1e-8 density Linf abs = 2.473825588822365e-10
rho > 1e-8 density Linf local-rel = 6.328741409524629e-6
rho > 1e-8 density Linf peak-rel = 2.6777820876067915e-6
max pair: x=20.03125, mirror_a=-0.03125, mirror_b=0.03125
```

At cycle 0/stage 1, `COORDSRCDBG` showed that the high-density mirror pair
entered `DynGRMHDPS::AddCoordTermsEOS` with parity-symmetric conserved inputs,
but the below-side source received zero lapse and shift:

```text
above: alpha=0.95353031364429219
above: beta=(0.09077972002248523, 0.00014162202811620161, 0.00014162202811620161)
above: src_momx=-2.0970968379518207e-09
above: src_tau=2.039226465360487e-10

below: alpha=0
below: beta=(0, 0, 0)
below: src_momx=0
below: src_tau=0
```

The below-side cached metric determinant and metric derivatives are nonzero and
parity-consistent, so this is not a simple all-ADM-cache failure. It points
specifically at how `adm.alpha` and `adm.beta_u` are stored or exposed to GRMHD
when Z4c is present.

The strongest current hypothesis is:

```text
ADM::adm.alpha and ADM::adm.beta_u alias pz4c->u0 when Z4c is present.
During the coupled RK task graph, pz4c->u0 can contain residual/stage storage
or can be modified by Z4c tasks while GRMHD source terms read it. The rest of
ADM, including g_ij, K_ij, psi4, is cached in ADM::u_adm and stays valid.
GRMHD coordinate sources therefore sometimes consume zero/stale lapse and shift
on a rank/block-dependent subset of the mesh.
```

The first candidate fix should make GRMHD consume a stable full-ADM cache for
lapse and shift as well as metric/K/psi4. Do not assume this fixes the full
coupled solver until the full coupled case is rerun.

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
- The first breaking task is dynamic GRMHD coordinate source terms, not x3
  reconstruction/Riemann, flux correction, RK update, C2P, or matter feedback.
- The immediate data defect is zero/stale lapse and shift entering
  `DynGRMHDPS::AddCoordTermsEOS` on one side of the mirror pair.
- Current best class: coupled ADM-to-GRMHD data path, specifically unstable
  Z4c-backed lapse/shift aliasing or task-order exposure of Z4c `u0`.

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
2. Make the smallest edit needed to test the stable ADM lapse/shift-cache
   hypothesis.
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

Prefer a narrow fix over a broad refactor. The first successful iteration
should prove whether stable ADM-owned lapse/shift storage removes the
cycle-0/stage-1 coordinate-source break in the zero-feedback case.

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

### 3. Confirm the Code-Level Hazard

Before editing, inspect the exact current code path and confirm the hazard is
still present:

```bash
rg -n "adm\\.alpha|adm\\.beta_u|u_adm|pz4c->u0|Z4cToADM|MHD_AddSrc|Z4c_Z4c2ADM" \
  src/coordinates/adm.cpp \
  src/z4c/z4c_adm.cpp \
  src/dyn_grmhd/dyn_grmhd.cpp \
  src/mhd/mhd_tasks.cpp \
  src/z4c/z4c_tasks.cpp
```

Expected current behavior:

- In `src/coordinates/adm.cpp`, when `pmy_pack->pz4c != nullptr`,
  `ADM::u_adm` is allocated with `nadm - 4`.
- In that same branch, `adm.alpha` and `adm.beta_u` are shallow slices of
  `pz4c->u0`.
- `adm.psi4`, `adm.g_dd`, and `adm.vK_dd` are shallow slices of `ADM::u_adm`.
- `src/z4c/z4c_adm.cpp` reconstructs full Z4c state and writes through
  `Z4cToADMImpl(pmbp, pz4c.full, pmbp->padm->adm, pz4c.opt)`.
- Because `padm->adm.alpha` and `padm->adm.beta_u` alias `pz4c->u0`,
  `Z4cToADMImpl` writes full lapse/shift back into Z4c solution storage rather
  than into the stable ADM cache.
- `MHD_AddSrc` depends on `MHD_ExplRK`, not on a fresh stable
  `Z4c_Z4c2ADM` cache. This makes `AddCoordTermsEOS` vulnerable to whatever
  value is currently exposed through `pz4c->u0`.

If this code shape has changed, stop and reassess before applying the fix.

### 4. Implement the Smallest Candidate Fix

Preferred first fix:

```text
Make ADM::u_adm always allocate all ADM fields, including lapse and shift, and
make adm.alpha / adm.beta_u always shallow-slice ADM::u_adm.
```

Rationale:

- GRMHD should consume one stable ADM cache for all ADM quantities.
- `Z4cToADMImpl` already writes alpha, beta, psi4, metric, and K through the
  abstract `adm_state`; after the fix, all of those writes land in `ADM::u_adm`.
- This matches how `g_ij`, `K_ij`, and `psi4` already behave.
- It avoids depending on Z4c `u0` being in a full-state form at every point in
  the coupled task graph.

Expected edit in `src/coordinates/adm.cpp`:

- Remove the special `pmy_pack->pz4c != nullptr` storage branch for alpha/beta.
- Allocate `u_adm` with `nadm` regardless of whether Z4c exists.
- Initialize `adm.alpha` from `u_adm, I_ADM_ALPHA`.
- Initialize `adm.beta_u` from `u_adm, I_ADM_BETAX, I_ADM_BETAZ`.
- Leave `adm.psi4`, `adm.g_dd`, and `adm.vK_dd` as slices of `u_adm`.
- Add one concise comment explaining that the ADM cache owns lapse/shift even
  for Z4c so GRMHD does not read Z4c residual/RK stage storage.

Do not make broader changes initially:

- Do not change `NGHOST`.
- Do not add parity-enforcement or symmetrization.
- Do not rewrite task ordering unless the stable-ADM-cache fix fails.
- Do not remove the existing env-gated diagnostics until the fix is validated.

After editing, inspect all uses of `I_ADM_ALPHA`, `I_ADM_BETAX`,
`I_ADM_BETAY`, `I_ADM_BETAZ`, and `u_adm` to make sure the larger allocation
does not require output or restart metadata adjustments:

```bash
rg -n "I_ADM_ALPHA|I_ADM_BETA|adm_alpha|adm_bet|u_adm" src analysis inputs
```

### 5. If the Preferred Fix Fails

If the one-cycle zero-feedback job still shows zero lapse/shift on one side,
do not start broad refactoring. Use the existing `COORDSRCDBG` evidence to
choose the next narrow probe:

- If `ADM::u_adm` has full lapse/shift before `MHD_AddSrc` but
  `AddCoordTermsEOS` sees zeros, inspect shallow-slice initialization and
  view lifetime.
- If `ADM::u_adm` itself has zeros before `MHD_AddSrc`, inspect
  `Z4c::Z4cToADM`, `Z4c::ReconstructFullState`, and task ordering around
  `Z4c_Z4c2ADM`.
- If the one-cycle zero-feedback case passes but the longer zero-feedback case
  fails later, rerun `COORDSRCDBG` at the first failing cycle/stage and compare
  the first nonzero metric time.
- If zero-feedback passes but full coupled still fails, rerun the same
  cycle-0/stage-1 probes on the full coupled deck before touching matter-source
  code.

### 6. Verification Gates

After any candidate fix:

1. Build locally and run input parsing smoke tests.
2. Rebuild the Aurora GPU executable if source changes affect it.
3. Rerun a one-cycle zero-feedback coordinate-source diagnostic first, using
   the same target and env vars as job `8523105`.
4. The one-cycle gate passes only if:
   - both high-density mirror partners see nonzero, parity-correct lapse/shift,
   - coordinate source increments are parity-correct and nonzero on both sides,
   - the one-cycle high-density density break drops from the old Linf abs
     `2.473825588822365e-10` to the SMR/Minkowski baseline or roundoff scale.
5. Then rerun `schwarzschild_zero_feedback_smr_dense`.
6. The fix is successful only if high-density density asymmetry in zero-feedback
   no longer grows above the SMR Minkowski baseline. Specifically compare final
   and time-history metrics against the old broken values:
   - old density high-density Linf abs `~3.42e-8`,
   - old Linf peak-relative `~3.57e-4`,
   - old Linf local-relative `~1.97e-3`.
7. Then rerun `schwarzschild_infall_smr_dense`.
8. The full solver is considered fixed only if the same high-density density
   and velocity asymmetry no longer grows above baseline.
9. Also rerun or spot-check `minkowski_static_smr_dense` if the fix touches
   shared MHD boundary/reconstruction/ADM consumption code.

One-cycle fixed-job template:

```bash
qsub -N z4c_zero_fb_coordsrc_fix \
  -v "CASE_NAME=schwarzschild_zero_feedback_coordsrc_fix_c0_n1,INPUT_DECK=/home/hzhu/athenak_tde/inputs/tde/aurora/z4c_tov_ks_schwarzschild_infall_zero_feedback_smr_dense_aurora.athinput,ATHENA_WALLTIME=00:09:00,ATHENA_EXTRA_ARGS=time/nlim=1 time/tlim=0.025,ATHENA_COORDSRC_DEBUG=1,ATHENA_COORDSRC_DEBUG_CYCLE=0,ATHENA_COORDSRC_DEBUG_STAGE=1,ATHENA_C2P_DEBUG=1,ATHENA_C2P_DEBUG_CYCLE=0,ATHENA_SYM_X_TARGET=20.03125,ATHENA_SYM_Z_TARGET=0.0" \
  analysis/tde_star_profile/aurora/submit_aurora_case.pbs
```

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
   - fixed one-cycle zero-feedback coordinate-source diagnostic,
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
