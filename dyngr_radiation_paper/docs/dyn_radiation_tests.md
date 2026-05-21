# Dynamical Radiation Test Matrix

The paper figures are generated from committed inputs and scripts in
`dyngr_radiation_paper/`.  Unless noted otherwise, run commands from the
repository root.

## Core Figures

| Test | Command | Output |
| --- | --- | --- |
| Flat beams | `python dyngr_radiation_paper/scripts/plot_comparisons.py` after running the three beam inputs | `figures/beam_comparison.png`, `figures/beam_2d_fields.png`, `figures/beam_2d_residuals.png` |
| Crossing beams | `python dyngr_radiation_paper/scripts/run_crossing_beams.py --run-dir /tmp/dynrad_crossing_beams` | `figures/crossing_beams_comparison.png`, `figures/crossing_beams_convergence.png` |
| Kerr photon-orbit beam | `python dyngr_radiation_paper/scripts/run_kerr_orbit_beam.py --run-dir /tmp/dynrad_kerr_orbit_beam` | `figures/kerr_orbit_beam_comparison.png` |
| Equilibration | `python dyngr_radiation_paper/scripts/run_equilibration.py --run-dir /tmp/dynrad_equilibration` | `figures/equilibration_comparison.png` |
| Linear waves | `python dyngr_radiation_paper/scripts/run_linear_wave_convergence.py --run-dir /tmp/dynrad_linear_wave` | `figures/linear_wave_convergence.png` |
| ADM formal checks | `python dyngr_radiation_paper/scripts/run_adm_formal_tests.py --run-dir /tmp/dynrad_adm_formal` | `figures/adm_formal_tests.png` |

## Dynbbh High-Resolution Beam

Build a CPU MPI dynbbh executable, then run:

```sh
cmake -S . -B build-mpi-dynbbh \
  -D CMAKE_BUILD_TYPE=Release \
  -D Athena_ENABLE_MPI=ON \
  -D PROBLEM=dynbbh \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_OPENMP=OFF \
  -D Kokkos_ENABLE_SYCL=OFF
cmake --build build-mpi-dynbbh -j6
python dyngr_radiation_paper/scripts/run_dynbbh_beam_figures.py \
  --run-dir /tmp/dynbbh_beam_highres_z4_mpi_t8 \
  --basename paper_dynbbh_beam_hi72z4_t8 \
  --athena build-mpi-dynbbh/src/athena --mpi-ranks 4 \
  --tlim 8.0 --nlim 260 --snapshot-dt 1.0 --track-dt 0.25 \
  --nx1 72 --nx2 72 --nx3 4 --mb-nx1 36 --mb-nx2 36 --mb-nx3 4 \
  --nlevel 4 --ppc 0.004 --ntrack 16 --beam-spread 20.0
```

`--mpi-ranks > 1` requires an executable configured with
`Athena_ENABLE_MPI=ON`; otherwise `mpirun` starts multiple independent serial
jobs that can corrupt shared outputs such as tracked-particle files.  The
figure script checks the executable configuration before launching MPI runs.
The SYCL Level Zero dynbbh device-loss investigation is recorded in
[`level_zero_device_lost_dynbbh.md`](level_zero_device_lost_dynbbh.md).

This produces `figures/dynbbh_beam_summary.png`,
`figures/dynbbh_beam_particles_lineout.png`, and
`figures/dynbbh_beam_timeseries.png`, plus the individual snapshot frames under
`figures/dynbbh_beam_timeseries/<basename>/`.

The current validated run completed on four MPI ranks to `t=8.0` with
`72 x 72 x 4` cells, `nlevel=4`, `dt=5.555556e-02`, 145 cycles, and empty
stderr.  The final lineout diagnostic reported
`max Rtt = 3.827080e+00` and integral `4.336137e+00`.

## ADM Formal Checks

The analytic-ADM checks are run without Z4c:

```sh
python dyngr_radiation_paper/scripts/run_adm_formal_tests.py --run-dir /tmp/dynrad_adm_formal
```

They cover:

- FLRW redshift with `a(t)=1+0.2t`, checking `E ~ a^-4` and
  `sqrt(gamma) E ~ a^-1`. The latest final relative errors at `t=0.5` were
  `1.18618e-4` for both quantities.
- Static lapse-gradient source with `alpha=1+0.1 sin(2 pi x)`, checking the
  sign and shape of `-F^i partial_i alpha`. The latest run had correlation
  `0.999307` with the first-order prediction and relative RMS mismatch
  `4.68375e-2`.
- Momentum-source closure, checking the angular Hamiltonian-force sum against
  the Valencia momentum source. The latest absolute residual fell from
  `1.80196e-5` at `Nx=32` to `1.14669e-6` at `Nx=128`, consistent with the
  second-order ADM metric-gradient cache.
