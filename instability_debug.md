# Goal-Mode Prompt: Residual Z4c TOV Instability Debugging

Goal: Continue debugging the residual Z4c self-gravity instability for the static TOV star, using small 2-node Aurora debug-queue jobs instead of 64-node debug-scaling jobs.

Repository: `/home/hzhu/athenak_tde`

Primary objective:
Determine why the residual Z4c self-gravity run for a static TOV star on a Minkowski analytic background develops an interior metric blow-up around `t ~ 33`, and implement the smallest decisive diagnostic or fix.

## Current Evidence

- The original full-box self-gravity run and the double-box run both fail around `t ~ 33`.
- Half-box run reached only `t = 26.93` before clean wall-clock exit, but showed the same growing Z4c norm trend.
- Double-box run `8525983` reproduced the failure:
  - Good through `t = 32.5`.
  - At `t = 33.0`, Z4c and MHD histories become NaN.
  - First NaN reports are near the stellar center, not near the outer boundary:
    approximately `(+-0.00625, +-0.00625, -0.01875)`.
  - Metric is already nonphysical there: `detg < 0`, `psi4 < 0`, huge `K_dd`.
- Radial density diagnostics show the star is not gradually dispersing:
  - Profile is retained through `t = 30`.
  - The `t = 35` dump is atmosphere floor after metric blow-up.

Relevant output paths:

- Half-box radial diagnostics:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_halfbox_amr_64c_n8_radial_density/`
- Double-box radial diagnostics:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_selfgrav_starfloor_doublebox_amr_64c_radial_density/`
- Double-box run directory:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/minkowski_static_selfgrav_starfloor_doublebox_amr_64c`
- Double-box submit log:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/submit/z4c_sg_floor64_double.o8525983`

## Leading Suspects

1. Fixed flat gauge:
   - Current decks use `<z4c>/evolve_gauge_residual = false`.
   - For a Minkowski background this forces the full evolved gauge to `alpha = 1`, `beta = 0`.
   - The initial TOV lapse residual is discarded, so this is not really a static TOV gauge evolution.

2. No Z4c constraint damping:
   - Current decks use `damp_kappa1 = 0`, `damp_kappa2 = 0`.
   - KO dissipation is on, but Z4c norm grows from small values to order unity before failure.

3. Background truncation error may not be fully factored out:
   - The residual evolution should compute `RHS(full state with matter) - RHS(background state without matter)` using identical finite-difference operators and gauge/damping terms.
   - If any damping, geometry, or projection term acts on the full state without subtracting the background counterpart, background truncation error can feed directly into the residual fields.

4. AMR/residual/projection interaction:
   - Residual fields are prolongated/restricted, then full fields are reconstructed and algebraic constraints are enforced, then fields are recast to residuals.
   - This nonlinear projection may amplify coarse/fine interface noise.

5. No early metric-validity diagnostic:
   - `ComputeGeometryData` inverts using `1/detg`; once `detg <= 0`, the run is already unrecoverable.

## Important Files

- `src/pgen/z4c_tov_ks.cpp`
- `src/z4c/z4c.cpp`
- `src/z4c/z4c_calcrhs.cpp`
- `src/z4c/z4c_tasks.cpp`
- `src/z4c/z4c_adm.cpp`
- Existing decks under `inputs/tde/aurora/`
- Existing wrappers under `analysis/tde_star_profile/aurora/`

## Constraints

- Do not change NGHOST behavior.
- Do not revert unrelated dirty-tree changes.
- Inspect the dirty tree before editing; preserve user/Cursor changes unless clearly obsolete and relevant.
- Keep diagnostics input- or environment-gated.
- Use `source ~/athenak_env` before Python analysis.
- Use project `MHDTidal` and storage under `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation`.
- For new Aurora tests, use the regular `debug` queue with 2 nodes, not 64 nodes on `debug-scaling`.
- Use short runs first:
  - 2 nodes
  - `-q debug`
  - walltime around `01:00:00`
  - internal clean-exit time around `00:50:00`
- Avoid submitting many jobs at once.
- Use local/serial smoke tests only for build/input parsing.

## Required Code Audit: Background Truncation-Error Subtraction

The residual Z4c evolution must consistently evolve:

```text
RHS_residual = RHS(full state with matter) - RHS(background state without matter)
```

using the same finite-difference operators, same gauge terms, and same damping terms, so analytic-background truncation error is removed from the residual fields.

Audit `src/z4c/z4c_calcrhs.cpp` and related helpers to verify the following.

1. Constraint damping is background-subtracted.
   - Check all `kappa1/kappa2` terms.
   - Damping terms for `Theta`, `Khat`, and `Gamma^i` should be of the form `damping(full) - damping(background)`, not just `damping(full residual)`.
   - If any damping term acts directly on full fields without subtracting the background contribution, fix it before running damping tests.

2. Gauge terms are background-subtracted.
   - If `evolve_gauge_residual = true`, verify lapse, shift, and `B^i` RHS terms are computed as full minus background.
   - If `evolve_gauge_residual = false`, verify the frozen gauge is intentionally prescribed and does not leave stale gauge residuals in ghost zones.

3. Geometry/Ricci/derivative terms are background-subtracted.
   - Check every term involving finite differences of `chi`, `gtilde_ij`, `Atilde_ij`, `Khat`, `Theta`, `Gamma^i`, lapse, and shift.
   - Confirm the code uses the same finite-difference stencil on `full` and `background`, then subtracts the results.
   - Search for any residual RHS term that uses only `full` or only `u0` where it should be `full - background`.

4. KO dissipation is appropriate for residual evolution.
   - Current expected behavior is usually KO on the residual field, not on the analytic background.
   - Verify this is intentional and document it.

5. Algebraic projection does not reintroduce background truncation error.
   - Check the sequence:
     `residual u0 -> reconstruct full -> enforce algebraic constraints on full -> recast residual`.
   - Determine whether enforcing constraints on the full field can inject background discretization/projection error into the residual.
   - If so, test enforcing projection on both full and background consistently before recasting.

Deliverable for this audit:

| Term group | Current implementation | Background truncation factored out? | Code location | Fix needed? |
| --- | --- | --- | --- | --- |
| Constraint damping | TBD | TBD | TBD | TBD |
| Gauge terms | TBD | TBD | TBD | TBD |
| Geometry/Ricci terms | TBD | TBD | TBD | TBD |
| KO dissipation | TBD | TBD | TBD | TBD |
| Algebraic projection | TBD | TBD | TBD | TBD |

## Requested Debugging Plan

1. Add or enable a lightweight Z4c metric-validity diagnostic.

   Track:
   - min/max `chi`
   - min `det(gtilde)`
   - min ADM `psi4`
   - min ADM `detg`
   - max `|K_dd|`
   - max `|A_dd|`
   - max `|Theta|`
   - max `|Khat|`
   - locations of all extrema

   Requirements:
   - Make output gated by an input flag or environment variable.
   - The diagnostic should identify the first sign of blow-up before MHD primitive recovery reports NaNs.
   - Prefer history-style compact output over verbose per-cell dumps unless a failure threshold is crossed.

2. Create small 2-node debug-queue decks/wrappers for decisive discriminator cases.

   Baseline reproduction, smaller:
   - Same static TOV residual self-gravity physics.
   - Use 2 nodes on the regular `debug` queue.
   - Prefer a smaller box/resolution that still reproduces the instability quickly enough, if possible.

   Prescribed TOV lapse residual:
   - Freeze shift if desired, but retain the initial TOV lapse residual instead of forcing `alpha = 1`.
   - Purpose: test whether the flat gauge causes the instability.

   Constraint damping scan:
   - Only run this after confirming constraint damping is background-subtracted.
   - Try modest nonzero `damp_kappa1`, for example `0.02`, `0.05`, `0.1`, with sensible `kappa2`.
   - Purpose: test whether the growing Z4c mode is dampable.

   Fixed refinement/no-derefine test:
   - Use fixed SMR or AMR with derefinement disabled near the star.
   - Purpose: separate AMR/prolongation noise from formulation/gauge instability.

3. Run local/serial smoke tests only for build/input parsing.

4. Submit 2-node Aurora debug jobs one at a time.

   Record for every run:
   - case name
   - job id
   - deck
   - executable
   - queue
   - node count
   - rank count
   - walltime
   - run directory
   - post-processing directory

5. Monitor each job until completion or failure.

   For each run, identify:
   - first bad field
   - first bad time/cycle
   - first bad location
   - whether the first bad location is interior, AMR interface, or boundary
   - whether the MHD failure is downstream of an already invalid metric

6. After each run, run radial density diagnostics.

   Summarize:
   - whether the star profile is retained
   - whether Z4c grows
   - first failing field/location/time
   - whether failure occurs before, near, or after `t ~ 33`

## Classification Target

- If retaining the TOV lapse residual stabilizes or delays failure, classify as gauge pathology.
- If damping stabilizes or delays failure, classify as constraint-mode instability.
- If fixed refinement stabilizes or delays failure, classify as AMR/prolongation/projection interaction.
- If background-subtraction audit finds a missing subtraction, fix that first and rerun the smallest reproduction.
- If none help, inspect residual RHS consistency more deeply and consider formulation-level ill-posedness or missing source terms.

## Deliverables

- Brief written debugging log with run table.
- Minimal code/input changes needed for diagnostics and tests.
- 2-node debug queue job ids and run directories.
- Metric-validity diagnostic outputs.
- Radial density plot paths.
- Background-subtraction audit table.
- Ranked conclusion and next recommended fix.

