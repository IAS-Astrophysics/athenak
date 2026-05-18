# Codex Handoff Prompt For Moving This Workflow

You are taking over an AthenaK workflow for Z4c + GRMHD TOV-star tests. The repository root is expected to be an AthenaK checkout. Start by reading these files:

- `src/pgen/z4c_tov_ks.cpp`
- `src/eos/primitive-solver/piecewise_polytrope.cpp`
- `src/eos/primitive-solver/piecewise_polytrope.hpp`
- `tov_stability.md`
- `inputs/dyngr/z4c_tov_ks_circular_orbit_hydro.athinput`
- `scripts/submit_ks_circular_orbit_hydro_mpi.sh`
- `analysis/plot_ks_orbit_hydro_movie.py`

## Current Physics Goal

We are testing whether a TOV star remains well resolved and shape-preserving when evolved with background-subtracted Z4c on analytic backgrounds:

1. Minkowski background, `bh_mass = 0`, `coord/minkowski = true`, fixed full gauge `alpha = 1`, `beta^i = 0`, for isolated-star stability.
2. Schwarzschild Kerr-Schild background, `bh_mass = 1`, `bh_spin = 0`, `use_analytic_background = true`, with a non-magnetized TOV star placed on a circular geodesic at `2*r_t`.

The latest user request is to run only one debug/test-queue segment for the circular-orbit hydro case, save restart files, and avoid automatic resubmission until results are inspected.

## Core Code Structure

`src/pgen/z4c_tov_ks.cpp` is the central problem generator.

Important features now in that file:

- `bh_mass = 0` selects a Minkowski background. This requires `<coord>/minkowski = true`; the analytic background then gives full lapse `1` and full shift `0`.
- `bh_mass = 1` selects the existing unit-mass Kerr-Schild helper. `bh_spin` is supported for the background, but the new circular geodesic orbit currently requires Schwarzschild: `bh_spin = 0`.
- `use_analytic_background = true` means the evolved Z4c variables are background-subtracted. The residual lapse, shift, and B fields are pinned to zero; the full gauge comes from the analytic background.
- `star_orbit = circular_geodesic` computes a Schwarzschild circular orbit from geodesic relations:
  - default radius is `star_orbit_radius_factor * tidal_radius`;
  - tidal radius is `R_star * (M_BH/M_star)^(1/3)`;
  - default factor is `2.0`;
  - `Omega = sqrt(M/r^3)`, `u^t = 1/sqrt(1 - 3M/r)`, and the initial tangential boost is derived from `r*u^phi`.
- `RefinementCondition` adds density-gradient AMR plus a star-tracking refinement guard. This is important for boosted and orbiting stars.
- `TOVKerrSchildHistory` writes a user history containing `rho-max` and `alpha-min`.
- `InitializeDipoleMagneticField` initializes a dipole magnetic field only if `b_norm != 0`. For the current first orbit test, keep `b_norm = 0.0`.

The piecewise-polytrope changes make EOS initialization stricter and accept `pwp_gamma_thermal`; they also initialize `eps_pieces[0]`.

## Tests Already Done

See `tov_stability.md` for details. Condensed results:

- Early no-dissipation/no-damping isolated-star GPU tests failed quickly.
- A damping sweep to `t = 20` found that KO dissipation is the main stabilizer.
- Recommended isolated-star settings from those tests:
  - `nghost = 4`
  - `reconstruct = wenoz`
  - `rsolver = hlle`
  - `diss = 0.5`
  - `damp_kappa1 = 0.0`
  - `damp_kappa2 = 0.0`
- Constraint damping with `damp_kappa1 = 0.02` did not materially improve the central-density behavior relative to `diss = 0.5` alone.
- A 200M dissipation-only test showed:
  - unboosted `L=5`, `dx=0.125` remained stable to `t=200`;
  - boosted `v=0.2` tests were dominated by the star leaving the box;
  - for `v=0.2` and `t_end=200`, the half-width should satisfy at least `L > v*t_end + R`, i.e. `L > 41.2M`, before adding safety margin.
- Resolution guidance for boosted stars:
  - minimum tested: `dx=0.125`, about 18.4 cells across the unboosted stellar diameter;
  - preferred: about 24 cells across diameter;
  - conservative: about 32 cells across diameter.

## Current Circular-Orbit Setup

Input:

- `inputs/dyngr/z4c_tov_ks_circular_orbit_hydro.athinput`

Script:

- `scripts/submit_ks_circular_orbit_hydro_mpi.sh`

Current setup:

- Non-magnetized star: `b_norm = 0.0`.
- TOV model matches the isolated-star stability tests:
  - `rhoc = 1.0e-6`
  - `kappa = 5.0e-3`
  - `gamma = 5/3`
  - `npoints = 200000`
  - `dr = 2.5e-4`
  - `rho_cut = 1.0e-16`
- Schwarzschild Kerr-Schild background:
  - `bh_mass = 1.0`
  - `bh_spin = 0.0`
  - `<coord>/a = 0.0`
  - `use_analytic_background = true`
- Orbit:
  - `star_orbit = circular_geodesic`
  - `star_orbit_radius_factor = 2.0`
  - expected radius is about `225.3M` for this TOV model.
- Numerics:
  - `nghost = 4`
  - `wenoz`
  - `hlle`
  - `diss = 0.5`
  - no constraint damping
- AMR:
  - root spacing is `dx = 4M`;
  - `num_levels = 6`, so finest `dx = 0.125M`;
  - this gives about 18.4 cells across the stellar diameter;
  - density-gradient AMR plus `amr_star_refine = true` keeps the star on the finest level.
- Outputs:
  - history every `1M`;
  - `xy_mhd` slices every `25M`;
  - restart files every `25M`.

The latest submitted job on the original cluster was `8425077`, using QoS `gpu-test`, 4 MPI ranks, 4 GPUs, 1 hour walltime, and AthenaK internal walltime `00:55:00`. The job was pending when this prompt was written.

## Build/Run Guidance On A New Cluster

Use the CUDA+MPI build, not the no-MPI build. The old path was:

```bash
cmake --build build_cuda_mpi_z4c_tov_ks -j 8
```

The old submit script assumes Slurm, A100 GPUs, and these options:

```bash
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --constraint=nomig&gpu80
#SBATCH --time=01:00:00
```

On the new cluster, adjust the scheduler directives, module environment, and `srun` GPU binding. Keep the executable path pointing to the CUDA+MPI build. Keep `-t 00:55:00` or similar so AthenaK writes a clean restart before the allocation ends.

Suggested first command after adapting the script:

```bash
sbatch scripts/submit_ks_circular_orbit_hydro_mpi.sh
```

After the segment completes, inspect:

- `run.log` for the printed orbit radius, omega, boost, and any AMR/allocation errors.
- `*.z4c.user.hst` for `rho-max` and `alpha-min`.
- restart files in the run directory.
- `bin/*.xy_mhd.*.bin` slices.

Generate the movie:

```bash
python3 analysis/plot_ks_orbit_hydro_movie.py /path/to/run_dir --zoom-width 10 --fps 4
```

The movie script makes a 2 by 3 panel layout:

- columns: density, pressure, and `|u^i|`;
- top row: global frame;
- bottom row: star-centered frame;
- overlays: expected geodesic, density-peak trajectory, and current offset;
- all panels use `aspect='equal'`.

## What To Decide Next

1. If the first one-hour segment reaches the AthenaK walltime and writes restart files, inspect the central density and star-centered movie before resubmitting.
2. If the star remains coherent, resubmit from the latest restart for another segment.
3. If the density peak walks off the geodesic or the shape distorts, first verify AMR is covering the whole star; then test one more refinement level or a larger star-refinement guard radius.
4. Keep the first magnetic-field tests separate. The current orbit test is intentionally non-magnetized.
