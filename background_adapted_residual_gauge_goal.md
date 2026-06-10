# Goal Mode Prompt: Background-Adapted Residual Gauge Debugging

Repository: `/home/hzhu/athenak_tde`

Objective: implement and explore a background-adapted residual lapse/shift gauge for the background-subtracted Z4c system, using small 2-node Aurora debug-queue jobs charged to `MHDTidal`. The goal is to remove the Schwarzschild Kerr-Schild residual lapse instability while keeping the analytic black-hole background an exact fixed point of the residual system.

## Context

The previous residual gauge debugging localized the Schwarzschild long-run failure to the residual lapse source:

- Source-off control stayed finite.
- Residual lapse source-on run failed early.
- The background source stayed finite, but `Khat_res` and then `src_res` grew rapidly.
- This points to lower-order background cross terms in the residual 1+log source, especially terms schematically like `alpha_bg * Khat_res` and `Khat_bg * alpha_res`.

The working hypothesis is that the current residual gauge is too close to a direct full-gauge subtraction. The full fields may have the expected Z4c characteristic structure, but the residual gauge variables are evolving under the operator linearized around a nontrivial Schwarzschild Kerr-Schild background. That leaves strong background-coupled residual source terms. We want a gauge driver that is explicitly background-adapted:

- The analytic background lapse/shift must remain an exact fixed point.
- Gauge sources must vanish when residual fields vanish.
- The driver should respond primarily to residual geometric quantities such as `Khat_res`, `Gam_res`, and possibly `Theta_res`.
- The principal gauge behavior should remain close to standard 1+log/Gamma-driver where possible.
- Dangerous lower-order products involving background curvature/extrinsic curvature and residual gauge fields should be removed or controlled.

## Constraints

- Do not change `NGHOST` behavior.
- Do not revert unrelated dirty-tree changes.
- Keep all new diagnostics and experimental gauge modes input- or environment-gated.
- Use `source ~/athenak_env` before Python analysis.
- Charge Aurora jobs to project `MHDTidal`.
- Use the Aurora `debug` queue with 2 nodes for these experiments, not the 64-node `debug-scaling` queue.
- Use short walltimes initially, for example `-t 00:20:00` or `-t 00:30:00`, unless there is a clear reason to extend.
- Submit one or at most a small number of jobs at a time; avoid queue-limit churn.
- KO dissipation should be at most `0.5` unless the user explicitly approves otherwise.
- Be careful with stability claims: record exact failure time, last finite time, first bad metric/history row, and whether constraints or gauge fields blow up first.
- Preserve the existing run logs and append new results rather than overwriting prior records.

## Relevant Files

Inspect these first:

- `src/pgen/z4c_tov_ks.cpp`
- `src/z4c/z4c.cpp`
- `src/z4c/z4c_calcrhs.cpp`
- `src/z4c/z4c_adm.cpp`
- `src/coordinates/adm.cpp`
- `analysis/tde_star_profile/aurora/residual_gauge_debug_run_log.md`
- `analysis/tde_star_profile/aurora/residual_gauge_history_summary.py`
- Existing Aurora PBS helper scripts under `analysis/tde_star_profile/aurora/`

Also search for all current residual gauge options, including:

- `evolve_lapse_residual`
- `preserve_lapse_residual`
- `evolve_shift_residual`
- `damp_kappa1`
- lapse source diagnostics
- shift/Gamma-driver source terms
- background update/reconstruction/recast hooks

## First Step: Repo Hygiene, Commit, and Push

Before implementing anything new or submitting any jobs, first restore a deliberate tracked-file baseline:

1. Run `git status --short`.
2. Inspect every modified tracked file with `git diff` and determine whether each change is relevant to the previous symmetry/residual-gauge debugging.
3. For tracked-file changes, either:
   - stage and commit the relevant changes needed to preserve the current debugging state, or
   - clean/revert only changes that are confirmed to be disposable and not user work.
4. Inspect untracked files and classify them as useful source/deck/log artifacts, generated outputs, or disposable crash/temp files.
5. Do not delete or revert unrelated user/Cursor changes. If unrelated tracked changes are present, leave them untouched and explicitly document that they were not included in the cleanup commit.
6. Clean up only files that are clearly disposable, such as stale core files or failed temporary artifacts. Do not remove run logs, scripts, inputs, or analysis files unless they are demonstrably obsolete and unrelated.
7. Stage only the relevant tracked-file changes and useful new files needed to preserve the current debugging state.
8. Commit the staged cleanup/debugging baseline before starting the background-adapted gauge implementation. Use a clear commit message, for example:

```text
Record residual gauge debugging diagnostics
```

9. Push the commit to the configured remote branch. If the branch has no upstream, set the upstream explicitly after confirming the target remote/branch from `git remote -v` and `git branch --show-current`.
10. After pushing, run `git status --short` again and document any remaining untracked or unrelated dirty-tree files in the new run log.
11. Only proceed with implementing the background-adapted residual gauge after the relevant tracked-file state has been committed or intentionally cleaned, and the resulting baseline commit has been pushed.

## Mathematical Target

Implement an experimental background-adapted residual gauge mode. It should be opt-in through the input file, for example:

```ini
<z4c>
residual_gauge = background_adapted
```

or another name matching local conventions.

For lapse, avoid evolving the raw direct-subtraction source

```text
S_alpha_res = S_alpha[full] - S_alpha[bg]
```

if that expands into strong lower-order background couplings. Instead implement a residual driver whose RHS vanishes on the background, for example one of these experimental forms:

```text
alpha = alpha_bg * exp(a)
D_bg a = -2 f_alpha Khat_res - eta_alpha a
```

or, if using `alpha_res` directly:

```text
D_bg alpha_res = -2 alpha_bg f_alpha Khat_res - eta_alpha alpha_res
```

where:

- `a = log(alpha / alpha_bg)` if practical,
- `Khat_res = Khat_full - Khat_bg`,
- `D_bg = partial_t - beta_bg^i partial_i` or the code's sign-convention equivalent,
- `f_alpha` should default to `1`,
- `eta_alpha` should default to `0` initially, then be scanned only if needed.

The exact implementation should follow the local Z4c gauge convention. Do not force the above formula blindly if the code uses different signs or variables. The key fixed-point requirement is:

```text
all residual fields = 0  =>  all residual gauge RHS terms = 0
```

For shift, implement or stub a similarly gated background-adapted Gamma-driver mode. A minimal first pass may keep residual shift disabled/prescribed while testing lapse. If implementing shift:

```text
D_bg beta_res^i = c_beta B_res^i - eta_beta beta_res^i
D_bg B_res^i = c_B D_bg Gamma_res^i - eta_B B_res^i
```

or the closest form consistent with the code. Again, the background must be an exact fixed point.

## Required Fixed-Point Tests

Before running expensive jobs, add or use diagnostics that verify:

1. Pure Schwarzschild Kerr-Schild background with no star gives zero residual RHS to roundoff.
2. `alpha_res = 0`, `beta_res = 0`, `B_res = 0`, `Khat_res = 0`, `Gam_res = 0` stays zero.
3. The diagnostic history reports the background-adapted source terms separately from the old full-minus-bg source if possible.
4. The maximum absolute residual gauge RHS on the pure background is reported.

If the background is not a fixed point, stop and fix that before submitting Aurora jobs.

## Implementation Plan

1. Read the current Z4c gauge RHS code and identify exactly where the lapse and shift sources are computed.
2. Add a small enum/string option for the experimental gauge mode.
3. Implement the lapse-only background-adapted mode first.
4. Keep the old residual gauge mode available for A/B comparison.
5. Add diagnostics:
   - `alpha-res`
   - `Khat-res`
   - old direct-subtraction lapse source, if available
   - new background-adapted lapse source
   - constraint norm or existing `C-norm2`
   - `Theta-max`
   - `bad-metric`
   - `alpha-min`, `alpha-max`, `chi-min`, `detg-min`
6. Build locally and run a serial or small local smoke test only to confirm parsing and startup.
7. Build the Aurora executable.
8. Run 2-node debug jobs comparing the old and new gauge behavior.

## Debug-Queue Experiment Matrix

Use the same resolution and physics settings as the previous minimal reproducer where possible. Prefer short runs to `t ~ 5M` or `10M` first, since the prior residual lapse failure appeared by `t ~ 3.2M`.

Minimum matrix:

1. **Old residual gauge control**
   - Schwarzschild Kerr-Schild
   - self-gravity on
   - residual lapse source as previously failing
   - 2 nodes, debug queue
   - confirm it reproduces early growth/failure

2. **Background-adapted lapse only**
   - same deck except new gauge mode
   - residual shift disabled/prescribed if needed
   - same dissipation and CFL as the control

3. **Background-adapted lapse with modest damping**
   - same as case 2
   - add `eta_alpha` or equivalent small damping, e.g. `0.05`, `0.1`
   - only if case 2 still shows growth

4. **Background-adapted lapse with constraint damping**
   - only after verifying constraint damping is itself background-subtracted and does not feed pure-background truncation error into residual fields
   - compare `damp_kappa1 = 0` and `0.1`

5. **Longer 2-node debug confirmation**
   - if a case is stable to `10M`, extend to `30M` or the maximum practical debug duration
   - compare against the old failing time around `30M` where relevant

Do not jump to 64-node production until the 2-node debug matrix shows a clear improvement.

## Analysis Requirements

For each run record:

- case name
- job id
- deck path
- executable path
- git commit/hash or dirty-tree summary
- node count
- queue
- walltime
- mesh/refinement mode
- gauge mode
- CFL
- RK integrator
- spatial order/reconstruction
- KO dissipation
- constraint damping parameters
- final time reached
- first bad time, if any
- first bad field/location, if available
- metric/history file paths
- plot paths

Use or extend:

- `analysis/tde_star_profile/aurora/residual_gauge_history_summary.py`

Required plots/metrics:

- gauge source vs time
- `alpha_res`, `Khat_res`, `B_res`, `Gam_res` vs time
- constraints vs time
- `alpha-min/max`, `chi-min`, `detg-min`, `bad-metric` vs time
- density maximum and star position if fluid is active

If a run fails, locate whether the first bad values appear:

- near mesh/block boundaries,
- near refinement boundaries,
- near physical boundaries,
- near the star,
- near the black-hole excision/inner strong-field region, if applicable.

## Monitoring

After submitting a job:

1. Check status with `qstat`/Aurora equivalent.
2. Sleep for about 10 minutes between polling cycles to preserve token usage.
3. When a job finishes or fails, immediately run the history summary script.
4. Append results to a new log:

```text
analysis/tde_star_profile/aurora/background_adapted_residual_gauge_run_log.md
```

5. If the first run fails due to a deck/build/scheduler mistake, fix the mistake and resubmit once.
6. If the first run fails due to physics/numerics, do not blindly resubmit. Analyze the failure first.

## Decision Criteria

Classify the outcome:

- **Success:** background-adapted lapse keeps `alpha_res`, `Khat_res`, and constraints bounded well past the old failure time.
- **Partial success:** failure is delayed but the same source growth remains.
- **No improvement:** new source still drives `Khat_res`/constraint blow-up at roughly the same time.
- **Different failure:** the instability moves to shift, constraints, metric determinant, AMR boundary, or fluid coupling.

Recommended next action depends on classification:

- If lapse-only background-adapted mode succeeds, test adding residual shift evolution.
- If lapse-only mode still fails through `Khat_res`, inspect Z4c `Khat` equation and constraint damping terms for background cross-couplings.
- If constraints blow up before lapse source, focus on background-subtracted constraint damping and residual Z4c RHS consistency.
- If failure localizes to mesh/refinement boundaries, inspect residual ghost-zone fill and background reconstruction/recast consistency across AMR/SMR boundaries.

## Deliverables

1. Code changes implementing the gated background-adapted residual gauge mode.
2. Input decks for the 2-node debug tests.
3. Local smoke-test output.
4. Aurora debug-queue job ids and run directories.
5. Updated run log with plots/metrics.
6. A short conclusion comparing old residual gauge vs background-adapted gauge.
7. Recommendation on whether this is ready for a longer 64-node Schwarzschild production test.
