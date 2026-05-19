# Orbit Test Cluster Transfer Prompt

You are taking over an AthenaK Z4c circular-orbit TOV-star test on a freer cluster.

## Repo State

- AthenaK repo root should be this checkout, currently at commit `66ed6f73`.
- Read and use:
  - `codex_cluster_handoff_prompt.md`
  - `src/pgen/z4c_tov_ks.cpp`
  - `inputs/dyngr/z4c_tov_ks_circular_orbit_hydro.athinput`
  - `scripts/submit_ks_circular_orbit_hydro_mpi.sh`
  - `analysis/plot_ks_orbit_hydro_movie.py`

## Goal

Run the non-magnetized TOV star circular-orbit test around a Schwarzschild Kerr-Schild black hole. Use the CUDA+MPI build, not the no-MPI build. Submit only one debug/test-queue segment, about 1 hour, save restart files, and do not auto-resubmit until diagnostics are inspected.

## Physics And Numerics

- Input: `inputs/dyngr/z4c_tov_ks_circular_orbit_hydro.athinput`
- Problem generator: `pgen_name = z4c_tov_ks`
- Background: `bh_mass = 1.0`, `bh_spin = 0.0`, analytic Kerr-Schild background, background-subtracted Z4c.
- Orbit: `star_orbit = circular_geodesic`, `star_orbit_radius_factor = 2.0`.
- Expected pgen output: orbit radius about `225.3 M`, omega about `2.96e-4`, tangential boost about `0.067 c`.
- Non-magnetized first test: `b_norm = 0.0`.
- Numerics: `nghost = 4`, `reconstruct = wenoz`, `rsolver = hlle`, `z4c/diss = 0.5`, no constraint damping.
- AMR: `num_levels = 6`, finest `dx = 0.125 M`, about 18.4 cells across the star diameter. Keep `amr_star_refine = true` and density-gradient AMR enabled.
- Outputs: history every `1 M`, `xy_mhd` bin slices every `25 M`, restart files every `25 M`.

## Build Guidance

Use or adapt the CUDA+MPI build. If the build directory exists:

```bash
cmake --build build_cuda_mpi_z4c_tov_ks -j 8
```

If you need to configure from scratch, inspect the existing AthenaK build conventions on the cluster and configure CUDA + MPI + Z4c support consistently. Do not switch to the no-MPI executable.

## Scheduler Guidance

Adapt `scripts/submit_ks_circular_orbit_hydro_mpi.sh` to the new cluster's Slurm directives, modules, and GPU binding. The old script requested:

- 1 node
- 4 MPI tasks
- 4 GPUs
- 12 CPUs/task
- 1 hour walltime
- AthenaK internal walltime: `-t 00:55:00`

Keep AthenaK's internal `-t` a few minutes shorter than the allocation so it can write clean restart files.

Run command shape:

```bash
srun ... build_cuda_mpi_z4c_tov_ks/src/athena \
  -i inputs/dyngr/z4c_tov_ks_circular_orbit_hydro.athinput \
  -d /path/to/run_dir \
  -t 00:55:00 2>&1 | tee -a /path/to/run_dir/run.log
```

## After The Segment

1. Check `run.log` for orbit radius, omega, boost, AMR block count, allocation errors, primitive-solve errors, NaNs, or fatal messages.
2. Check history and user-history files for `rho-max` and `alpha-min`.
3. Verify restart files exist.
4. Generate diagnostics and movie:

```bash
python3 analysis/plot_ks_orbit_hydro_movie.py /path/to/run_dir --zoom-width 10 --fps 4
```

Do not over-shrink the domain to make it fit. A tiny local patch around the star was tried and reached `t=5`, but produced primitive-solve NaNs at the artificial patch boundary, so that is not a valid stability test. Prefer the full 4-GPU MPI setup or a carefully scaled equivalent.

## Report Back

Include:

- build command and executable used
- scheduler script changes
- job id and run directory
- whether restart files were written
- final simulated time reached
- central density behavior from `rho-max`
- whether the star remains coherent in the star-centered movie
