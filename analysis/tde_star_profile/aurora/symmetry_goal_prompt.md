# AthenaK TOV/Z4c Symmetry Goal Prompt

Goal: Continue the AthenaK TOV/Z4c symmetry discriminator campaign from
`/home/hzhu/athenak_tde` with low-token monitoring.

## Current State

- Worktree should be clean.
- Relevant commits:
  - `118936b1 chore: add symmetry job monitor`
  - `4b6f7e79 docs: record first symmetry baseline submission`
  - `d4639483 feat: add TOV symmetry discriminator suite`
  - `c072051f chore: preserve TOV symmetry diagnostics baseline`
- Aurora GPU executable is built:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-z4c_tov_ks/src/athena`
- First baseline `minkowski_static_uniform_dense`, job `8522794`, completed.
- Its postprocessed metrics are in:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/minkowski_static_uniform_dense`
- Uniform baseline result so far: MHD `dens`, `velx`, `vely`, and `velz`
  are exactly symmetric for `all`, `rho > 1e-12`, and `rho > 1e-10`;
  `rho > 1e-8` has no valid central-window pairs.

## Monitoring Rule

Use the monitor script to conserve tokens:

```bash
analysis/tde_star_profile/aurora/monitor_symmetry_job.sh JOB_ID CASE_NAME
```

It sleeps 10 minutes between checks by default. Do not poll more frequently
unless a job is expected to have just completed.

Report only state changes, completion, failures, metric paths, and compact
metric summaries.

## Next Execution Order

1. Submit `minkowski_static_smr_dense`:

   ```bash
   CASE_FILTER=minkowski_static_smr_dense \
     analysis/tde_star_profile/aurora/submit_aurora_roadmap.sh
   ```

2. Monitor that job:

   ```bash
   analysis/tde_star_profile/aurora/monitor_symmetry_job.sh JOB_ID \
     minkowski_static_smr_dense
   ```

3. When complete, summarize symmetry metrics compactly and update:

   ```text
   analysis/tde_star_profile/aurora/sector_isolation_run_record.md
   ```

4. Commit the run-record update.

5. Then submit one job at a time:

   - `schwarzschild_infall_smr_dense`
   - `schwarzschild_zero_feedback_smr_dense`
   - `schwarzschild_fixed_mhd_tmunu_smr_dense`
   - `schwarzschild_fixed_mhd_refresh_tmunu_smr_dense`

## Analysis Requirements

- Source `~/athenak_env` before Python analysis.
- Do not overinterpret raw local-relative errors at atmosphere density.
- Report absolute, local-relative, and peak-relative L2/Linf.
- Prefer high-density masks when nonempty; if `rho > 1e-8` is empty, say so
  and use `rho > 1e-10` / `rho > 1e-12`.
- Classify a break only when absolute and peak-relative metrics grow coherently
  above the Minkowski static baseline in high-density material or relevant
  ADM/Z4c fields.
- Keep diagnostics/input changes env- or input-gated.
- Do not change NGHOST behavior.
- Do not revert unrelated dirty-tree changes.
