# Hole-centered Bondi cyclic zoom

The physical input pair is in `inputs/dyngr/bondi_hierarchy/`. Regenerate it with
`scripts/tests/make_bondi_hierarchy.py --out inputs/dyngr/bondi_hierarchy`.
The gas reservoir, outer Michel boundary, total mass, orbit/frame angular velocity,
box and cavity prescription match. The hierarchy follows the actual horizon(s):
one mass-one hole or two mass-half holes. Each finest cell has width m_hole/16.
The binary no longer has a finest-level patch at the empty center of mass.
These are small-separation development cases: a=32 M, r_B=161.23 M, box ±256 M,
8³ active cells per MeshBlock. They are not parsec-scale production simulations.

## Residence and geometry

`amr_criterion1/bondi_hydro_schedule=true` selects Eq. 17 of
[Guo et al., Cyclic Zoom](https://arxiv.org/html/2504.16802v2#S3.SS1):
100 t_g at zone zero, zero residence at zones one and two, and ten dynamical
times elsewhere. This is the unmagnetized hydrodynamic prescription. A viscous
or magnetized disk needs its own appropriate relaxation criterion.

The binary adaptation is a modeling choice, not a result validated in that paper.
At r<a, the mask is the union of spheres at the two holes and its radius means
distance to a hole. In G=M_total=c=1 units, r_i=0.5×2^i and
 t_dyn=sqrt(r_i³/0.5). At r>=a the mask becomes a COM sphere and t_dyn=sqrt(r_i³).
At the switch the new sphere encloses both previous spheres for equal masses.
Saved topology restores the original two-center hierarchy on the return sweep.
The preset currently rejects unequal masses rather than silently assigning them
incorrect common length units. Both local clocks coincide for equal masses;
future unequal-mass support should take the longer required dwell and the most
restrictive (shortest) CFL step, with independently mass-scaled local radii.

| Binary zone | Radius / M_total | Center(s) | Residence / M_total |
|---|---:|---|---:|
| 0 | 0.5 | two holes | 50 |
| 1 | 1 | two holes | 0 |
| 2 | 2 | two holes | 0 |
| 3 | 4 | two holes | 113.137 |
| 4 | 8 | two holes | 320 |
| 5 | 16 | two holes | 905.097 |
| 6 | 32 | COM | 1810.193 |
| 7 | 64 | COM | 5120 |
| 8 | 128 | COM | 14481.547 |

A full cycle has one visit to each endpoint and two to intermediate stages:
31068.40 M = 27.3158 binary orbital periods, of which 50 M advances the finest
horizon grid (0.16094% of the global time). The single-hole cycle is 30174.48 M,
with 100 M on its finest grid. These clocks must be reported separately.
Skipped zones still require an AMR transition and currently one numerical step;
no zero-time event can repeat indefinitely. Timesteps can slightly overshoot a
dwell, so the table gives requested residence, not exact step-aligned duration.

`residence_scale=1` is the physical default. Very small values are only for
bounded mesh/restart tests and do not demonstrate relaxation. `trun_max`, if
explicitly reduced, still caps residence; leave its default for Eq. 17.

## Initial cavity and steady feeding

The current seed starts tapering at r_cut=a (32 M in these inputs). For
q=min(r/r_cut,1), S(q)=q³(10−15q+6q²) gives
rho=rho_floor+S(q)(rho_Michel−rho_floor). It approaches the floor continuously
at r=0, with no finite flat floor core. S has zero first and second derivatives
at both endpoints. Pressure follows K rho^gamma subject to its floor; radial
inflow is tapered by the same S. Outside r_cut the original fields are unchanged.
The single-hole control uses the same taper and reservoir.

`bondi_cavity_core_fraction=0` is now the default. A positive value retains the
optional finite floor core for reproducing older experiments. This affects only
fresh initial data, never checkpoints or the outer feeding boundary. The movies
of job 8925171 show the previous r_cut=a/2, core_fraction=0.5 seed and must not be
interpreted as results from the revised taper.

Do not assume the initial transient is harmless without checking repeated-cycle
accretion profiles and sensitivity to the cavity seed. The prescription removes
the artificial central cusp; it does not establish a relaxed binary flow.

## Performance changes

The stationary ADM background was already computed only on initialization and
mesh changes. Boundary primitives now have a problem-owned cache keyed to the
same metric/layout invalidation; conserved boundary variables are still rebuilt.
`problem/bondi_cache_boundary=false` retains the direct reference calculation.

The profile points primarily to reconstruction/flux kernels, not metric or
boundary evaluation. `mhd/dyn_flux_team_size=0` retains Kokkos AUTO; explicit
GPU team sizes are benchmarked for the short rows in small MeshBlocks. The CPU
accepts only AUTO or one. The generic output loader now packs selected blocks
per variable and performs one device-to-host transfer per variable rather than
allocating and copying each block individually, including sliced selections.

The Aurora benchmark runs one node in `debug`, never `debug-scaling`, using
rank-specific GPU affinity and per-rank mesh/storage capacities. It compares
one rank, six whole-GPU ranks, and twelve tile ranks. Aggregate tracked Kokkos
allocation limits total 16 GB and aggregate RSS limits total 12 GB, leaving
headroom below the requested 30 GB. Reports retain actual peaks and deadlines.
The pinned compiler/runtime recipe is documented in
[the SANE environment audit](aurora_profile/sane_build_environment.md).
Kernel profiling synchronizes launches and is used to locate costs; speedups
must use unprofiled timing, excluding initialization/JIT and output intervals.

## Correctness fixes found during this work

The floor-core stress test exposed a `ResetFloor` policy bug: when density and
 temperature were simultaneously low, non-aggressive density flooring returned
before enforcing temperature. Both floors are now applied independently; a
small C++ regression exercises density-only, temperature-only and combined cases.
The evolution EOS uses a temperature floor, so evolved pressure need not remain
above the separate initial-condition pressure floor.

The merge/split restart test exposed stale coarse ghost corners at physical
boundaries. The analytic boundary now restricts its fine conserved ghost state
into the corresponding coarse boundary slots before prolongation. Without this,
restart and continuous solutions differed after returning to finer levels.
With the fix, all saved float32 fluid, metric and magnetic fields in the test
are bitwise identical through a mid-cycle restart, with and without boundary
caching. Both outward and inward transitions restore the exact original topology.

Aurora's MPI advertises MPI_TAG_UB=262143 (18 bits), whereas the old tag encoding
assumed 20 bits. Startup now derives the local-block-ID bit count from the
reported limit, reserves six buffer bits, and checks new/restart capacities.
This selects 12 local-ID bits on Aurora; implementations supporting the old
20-bit layout retain 14. A standalone encoding regression covers the MPI
minimum, Aurora's limit and larger limits.

Per-block collective checkpoint writes stalled in the six-rank Aurora test.
Disjoint checkpoint records now use independent MPI-IO writes, preserving the
file layout and metadata synchronization. Validation results and measured
performance are recorded separately in the profiling report.

## Longer low-resolution pilot

Job 8925171 used physical residence times on one node and saved 32 finite,
positive snapshots through t=3100.56 M. It stopped at t=3199.735 M when the final
outward transition changed no blocks. The low-resolution input now uses a wider refinement-overlap region
(`f_ref=4`), which passed a complete eight-level outward/inward sweep with exact
restored topology and finite positive saved fields. The guard remains enabled.
This changes the mesh coarsening region, not the residence-time formula. The
original failure is retained in its receipt; no long-time relaxation claim is made. The density/temperature
movie uses saved states only, without interpolating simulation states.
