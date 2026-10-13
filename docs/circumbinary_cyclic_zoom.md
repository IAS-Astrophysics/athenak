# Circular binary cyclic zoom prototype

This branch connects Minghao Guo's cyclic zoom machinery to the circumbinary
branch's prescribed ADM metric and Valencia GRMHD solver. It provides a small,
magnetized ambient reservoir test and repeatable local regression checks.
These are infrastructure tests, not converged parsec binary accretion models.

For the shift-aware finite-radius Michel inflow, rotating single-hole validation,
and matched Aurora controls, see [Bondi boundary validation](bondi_boundary_validation.md).
Select `problem/initial_condition=bondi_inflow`; the original ambient input below
retains its constant reservoir boundary for infrastructure regression tests.

## Source history

The local branch `codex/circumbinary-cyclic-zoom` starts at fetched
`origin/main` (`c5a0d7f9155a70149931bf0be5a4ffb673f2532a`, cyclic zoom #713)
and merges the newest published circumbinary branch, `origin/circumbinary_stable`
(`ccd38a7858d43f5e76854dfb5fd8801983ae14aa`). The branches have extensive framework
divergence. Merge conflicts retain the circumbinary framework/solver and port
main's cyclic zoom addition onto it. This is not a reconciliation of every
unrelated main framework change. Kokkos uses the circumbinary branch's gitlink.
The original checkout is untouched.

## Implemented behavior

Set `problem/corotating=true`, `problem/initial_condition=bondi_reservoir`, and
`adm/dynamic=false`. With `G=M_total=c=1`, the fixed circular orbit has
`Omega=sep^(-3/2)`. The phase-zero boosted superposed Kerr-Schild four-metric is
pulled back by `v_grid=Omega cross r`, including its time-space components.
The resulting spatial metric and lapse equal their inertial phase-zero values,
and `beta_rot=beta_inertial+v_grid`. Spatial automatic differentiation includes
the rotation; coordinate time derivatives vanish. The ADM data are recomputed
after mesh changes, rather than at every Runge-Kutta stage. This remains an
approximate prescribed binary spacetime, without radiation reaction.

Valencia CFL now uses the conservative coordinate light-cone bound
`|beta^i|+alpha sqrt(gamma^ii)` at the two interpolated faces of each active cell.
A rotating grid can extend beyond its light cylinder; its coordinate shift is
then larger than one and the outer timestep bound matters.

`amr_criterion*/ncenters=2` enables the union of two growing spherical masks.
Set both centers to the fixed hole positions, using `x1c,x2c,x3c` and
`x1c2,x2c2,x3c2`. For the equal-mass examples they are at `x=+sep/2,-sep/2`.
Complete coarsening families are saved; the return sweep restores their saved
logical topology. This avoids refinement spreading into unrelated siblings.
The Valencia mask restores primitive pressure and Eulerian `W v^i`, then uses
the existing EOS to convert only the restored cells with their actual ADM
metric and current CT magnetic field. The binary's physical puncture excision
is retained: set `f_exc=0`; artificial central Kerr excision is unsupported.
The original EMF storage/correction machinery is retained and uses the
Valencia Riemann solver when refreshing fluxes.

The reservoir starts with uniform rest density `rho_inf`, sound speed `cs_inf`,
and zero Eulerian `W v^i`. It is not an exact steady Bondi solution. Accretion
must develop dynamically. This implementation defines `r_bondi=2/cs_inf^2`;
`cs_inf`, if supplied, overrides the sound speed inferred from `r_bondi`.
Pressure is `rho_inf cs_inf^2/[gamma (1-cs_inf^2/(gamma-1))]`.
`b_inf` is the uniform **densitized** vertical CT field, not a spatially uniform
orthonormal magnetic field. User boundaries replenish reservoir gas and magnetic
ghost faces only outside the domain; normal faces on the physical boundary
remain CT-evolved. Internal AMR ghost cells retain exchanged data.
The reservoir initializer requires `user` on all six outer boundary faces.

Keep `ncycle_check=refinement_interval=1`. Start fresh runs at zoom zone zero.
`trun_fac_z r_z^trun_pow`, capped by `trun_max`, specifies a physical residence
time at each level. Zero intervals are supported for tests and trigger at most
one transition per completed cycle. Production dwell times need independent
physical calibration. Zoom restart records have a version marker and last-event
cycle; old upstream zoom restart records are incompatible. Both scheduled
checkpoints before AMR and final checkpoints after AMR are supported.

The pre-coarsening EMF is evaluated from the current state after synchronizing
and prolonging ghost data on both uninterrupted and restarted runs. Solver
scratch EMFs and cached primitives at AMR corners are not used as checkpoint
state. Boundaries are refreshed after the final CT face repair. Derived output
buffers also grow when the mesh grows after a coarse restart.

## Bounded Mac tests

The supplied smoke input uses `4^3` active cells per block, four ghost cells,
nine physical levels, 960 initial blocks, `sep=64`, and `r_bondi/sep=32`.
Small blocks exercise hierarchy and redistribution; their ghost-cell overhead
is large and they are unsuitable for performance or convergence claims.

```sh
git submodule update --init --recursive
cmake -S . -B build-mac-zoom-opt -DPROBLEM=dynbbh \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAthena_ENABLE_MPI=OFF \
  -DAthena_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_DEBUG=ON -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
  -DKokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK=ON
python3 scripts/tests/run_bounded.py --seconds 600 --memory-gb 20 \
  --report build-mac-zoom-opt/build-resources.json -- \
  cmake --build build-mac-zoom-opt -j 2
python3 scripts/tests/run_bounded.py --seconds 900 --memory-gb 20 \
  --report build-mac-zoom-opt/test-resources.json -- \
  python3 scripts/tests/test_binary_zoom.py \
  --exe build-mac-zoom-opt/src/athena --out build-mac-zoom-opt/integration
```

Python requires NumPy; reading binary outputs does not require h5py. The wrapper
monitors the aggregate resident memory of the command and its descendants every
100 ms, kills them at the deadline or budget, disables core dumps, and writes
a resource report. On macOS this is sampled RSS monitoring, not an OS-enforced
address-space cap. Configured small meshes provide headroom below 30 GB.
Run builds/tests sequentially. The suite writes its numerical measurements to
`integration/results.json`; generated output and build directories are ignored.

Checks include metric pullback and time independence, automatic versus finite
derivatives, evolution beyond the light cylinder, a complete nine-level zoom
cycle, restart equivalence before/after AMR and during the second outward
sweep, and a complete eighteen-level cycle. The latter has `sep=8192`, an outer
half-width of `1048576`, and `r_bondi/sep=128`. It spans a factor `2^17` in cell
size and is intentionally under-resolved near the horizons.

The short restart case uses a strict pointwise comparison. Longer fluid restart
comparisons require a relative component maximum-norm difference below `1e-5`
and record the measured residuals. Time, cycle, logical topology, and ADM data
must match exactly; divergence differences must be below `1e-12`. Reconstructing
ghost states and recovering primitives does not currently guarantee bitwise
fluid reproducibility. The tests compare float32 visualization fields and do
not establish double-precision checkpoint equivalence for arbitrary inputs.

Unsupported inputs fail explicitly: dynamical spacetime/radiation with zoom,
self-gravity, tilted or evolving spins, trajectory tables in the rotating
setup, moving excision, invalid hierarchy/intervals, and rotating torus ICs.
Use exactly one AMR criterion for Valencia cyclic zoom.
The tests do not validate MPI, GPUs, a steady Bondi accretion rate, or long-time
conservation/convergence of the cyclic approximation.

## Recorded local results (2026-10-12)

Apple M5 Pro, 48 GiB RAM, Apple Clang 21, Kokkos Serial with bounds checks:

| Test | Wall time | Mesh restored | Maximum absolute div B |
| --- | ---: | ---: | ---: |
| Nine-level full cycle, 16 steps | 61.70 s | 960 blocks | 5.64e-18 |
| Eighteen-level full cycle, 34 steps | 245.41 s | 1,968 blocks | 5.73e-19 |

The complete suite took 487.68 s with a sampled aggregate peak RSS of
12,119,080,960 bytes (12.12 decimal GB), under its 20 GB guard. All states had
positive density and pressure. Before-AMR, after-AMR, and second-sweep restarts
passed; the largest recorded relative fluid field difference was `4.77e-6`.
Metric, cycle, time and mesh topology comparisons were exact.

The nine-level timestep grew from 0.02368 to 4.3146 total-mass time units
(a factor 182); the eighteen-level timestep grew from 0.19892 to 11060.84
(a factor 55,605). End-to-end speedup depends on physical dwell times and
convergence. These schedules deliberately trigger zooms every step and do not
measure the cost of a relaxed Bondi flow. The deep run advances only about
40,681 total-mass time units.

Machine-readable measurements are saved in
[`circumbinary_cyclic_zoom_validation.json`](circumbinary_cyclic_zoom_validation.json).
Raw final-suite logs, visualization fields, and checkpoints remain in
`build-mac-zoom-opt/integration`; exploratory failed-run outputs were removed.

## Before a physical parsec-scale calculation

Choose total mass and separation before converting to code units:
`r_g=4.785e-6 pc (M_total/1e8 Msun)` and
`t_g=492.55 s (M_total/1e8 Msun)`. Thus the deep example can represent
`sep=1 pc` at approximately `M_total=2.55e9 Msun`, with a nominal Bondi radius
of `128 pc`. It does not include the host potential that may dominate there.
Choose the outer box to contain the intended reservoir, resolve each horizon
on the finest mesh, and enable the horizon resolution checks. A gas density
normalization is physically arbitrary only while gas self-gravity and
radiative processes are neglected.

Cooling, gas self-gravity, fragmentation/star formation, stellar gravity,
radiative feedback, and angular momentum of the supplied gas require separate
modeling. This prototype does not answer whether cold gas at large radii
fragments. Neither the short test durations nor their aggressive zoom schedule
establish the cost or validity of a relaxed astrophysical solution.
