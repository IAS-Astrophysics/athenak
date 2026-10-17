# Magnetized binary Bondi pilot

This branch implements and tests a low-resolution, magnetized uniform reservoir
around an equal-mass circular binary. The physical cyclic-zoom run is ongoing
on Aurora in debug-scaling. Job8938200 reached9782.468493 M cleanly after
completing the first physical roundtrip and restoring the finest topology at
9683.038 M. The returned finest snapshot matches the original stationary ADM
geometry exactly. Job8938971 subsequently stopped cleanly at9903.615850 M, still on the
finest grid. Its fields and all twelve resource reports passed validation.
Job8939546 is the accepted continuation toward the unchanged20000 M target.

Seven completed physical jobs have passed every saved fluid/divergence pair and
all twelve rank/resource checks. Their accumulated solver walltime is about
6.4 node-hours. The latest finest exposure is only about470.58 M: the initial
250 M dwell plus220.58 M after the first return. Global9904 M must not be
interpreted as9904 M of inner-flow evolution or as demonstrated equilibrium.

| Physical job | Final time (M, native checkpoint where recorded) | Conservative peak GB | Scaled divB maximum |
| --- | ---: | ---: | ---: |
| 8937401 | 117.945320 | 23.56 | 4.74e-13 |
| 8937572 | 238.431940 | 23.45 | 5.81e-13 |
| 8937793 | 1601.324871 | 22.04 | 5.12e-13 |
| 8938001 | 7579.573404 | 22.63 | 1.86e-12 |
| 8938136 | 9518.668019 | 22.81 | 2.87e-12 |
| 8938200 | 9782.468493 | 23.53 | 2.32e-12 |
| 8938971 | 9903.615850 | 23.47 | 3.29e-12 |

The scaled divergence values are the cell-centered diagnostic defined below,
not a proof of exact global magnetic-flux or mass conservation. Every job used
one node, twelve tile ranks and the55-minute clean solver cutoff. The exact
continuation receipts and physical-JOB reports retain restart provenance.

## Configuration and meaning

- Total binary mass1, equal component masses0.5, separation32 M, spins0,
  orbital/frame frequency32^(-3/2). The orbit and metric are prescribed.
- Uniform initial rho_inf=1, thermal rB=2GM/cs_inf^2=500 M, beta_inf=1,
  gamma=5/3, vertical field, and2% initial pressure noise. Density is a
  scale-free normalization, not a gas self-gravity model. No cavity is needed
  to remove a central Michel density cusp because this seed is uniform.
- Thermal sound speed and pressure obey the relativistic gamma-law relation.
  beta_inf fixes the asymptotic field normalization B_inf=sqrt(2P_inf/beta_inf).
  Local physical beta varies with curvature and evolves dynamically.
- Box faces are at2048 M (4.096 rB); root32^3, MeshBlocks8^3 with4 ghost cells,
  thirteen physical refinement levels, eight cyclic levels,1408 initial blocks.
  Minimum dx is0.03125 M near the punctures; the nominal r=1 M horizon sphere
  has dx=0.0625 M. This is a pilot, not a spatially converged reproduction.
- The mask-radius ladder starts at0.5 M and doubles through64 M. A24x48
  angular audit found r_mask/dx=16 at all sampled boundary points in all8
  stages, including the two merged COM stages. This is not an apparent-horizon
  finder. Exact audit metadata are in gpu-roundtrip-8937389/.

The MHD timing preset follows the structure of Eq20 in
[Guo et al.](https://arxiv.org/html/2504.16802v2#S3.SS3):250 M on the finest
stage (500 individual-hole gravitational times), zero prescribed dwell on
stages1 and2, then min(50 local dynamical times,0.15 global Bondi times).
The local mass is0.5 below separation32 M and1 after the masks merge into
one COM mask. The shared reservoir cap is1185.85 M. residence_scale=1 in all
physical jobs; accelerated values were used only in explicit topology tests.
Global elapsed time must not be equated with fine-zone exposure.

## Inertial-rest data and outer boundary

The initial seed and exterior ghost cells use the same coordinate-velocity
conversion. Nonrotating coordinate rest becomes v_coordinate=(Omega*y,-Omega*x,0).
With the actual binary ADM fields,

```
v_Eulerian = (v_coordinate + beta_shift) / alpha
W = (1 - gamma_ij v_Eulerian^i v_Eulerian^j)^(-1/2)
u_primitive = W v_Eulerian
```

Rest is imposed where timelike and within the configured Lorentz ceiling10.
Spacelike initial rest states near the holes are limited to v_Eulerian^2=0.9;
the EOS also limits a thin high-W region. These inner seed exceptions are
explicitly tested and counted. A spacelike boundary rest state is rejected.
The actual outer reservoir is far from these limits.

All six faces maintain uniform exterior-ghost density, pressure and vertical
densitized field Btilde=sqrt(gamma)*B_Eulerian. Noise is initial only. Physical
normal magnetic faces remain CT-evolved; only exterior faces are prescribed.
The mass flux comes from the interior/exterior Riemann problem and is not set
to an analytic Bondi rate. This is a fixed reservoir, which can reflect subsonic
signals. Boundary-distance sensitivity remains a production requirement.

## Verified results

- Local32^3/64^3 inertial/rotating single-hole controls reached2 M. Fine-grid
  differences were0.0411% density,0.513% pressure and0.0202% vertical field,
  all smaller than at32^3. Initial and outer-ghost inertial velocities recover
  zero within2e-7 wherever representable. Deeper magnetic ghosts retain the
  prescribed field. A deliberately impossible boundary is rejected.
- The tight a=4 M binary control checks the combined metric static limit,
  rather than using isolated-hole spheres. Tests distinguish spacelike seed
  limits and the numerical Lorentz ceiling. No solver change was required.
- GPU test8937389 sampled all8 stages and restored the exact finest topology
  with positive finite fluid and max scaled divB3.43e-15. This was accelerated
  topology validation, not a physically relaxed evolution.
- Physical jobs8937401 and8937572 stopped normally at the55-minute cutoff.
  All12 rank checks passed. Conservative RSS-plus-tracked peak sums were
  23.56 and23.45 GB. The stationary metric was evaluated once per rank during
  each of these fixed-topology jobs. Later mesh changes require reevaluation.
- The physical300.022 M coarse snapshot has1072 active blocks and scaled
  divB5.05e-13. Its density and pressure are finite and positive.

Diagnostics retain raw divB and an exterior-excision value, and gate on
`dx*abs(divBtilde)/max(norm(Btilde),0.001*ambient) < 1e-9`. This cell-centered
proxy is not an invariant face-flux norm. Large fields inside puncture excision
can increase raw absolute roundoff; their amplitudes are not interpreted as
physical magnetic pressure. Every saved fluid/divB pair is checked for matching
time, cycle and topology. Normal termination, all rank reports, one host and
all12 tile affinities are also required.

## Build, execution and continuation

The isolated source-magnetic/build-magnetic pair uses the validated SANE
compiler/Kokkos configuration. Build/runtime MPICH versions, flags, source
manifest and executable hashes are in aurora-build/. Of337 source/header/CMake
files,336 match the branch byte-for-byte; the only source difference is an
archived stale MPI-tag comment. Build CMake, script and compiler launcher match.
An older runtime copy omitted explicit EnableImplicitScaling=0; this is recorded
and synchronized for job8937793 onward. Runtime controls are now saved at launch.
Existing SANE production files and other agents' jobs were not modified.

Each PBS job requests debug-scaling, one node and one hour, launches12 explicit
tiles and uses Athena `-t 00:55:00`. The aggregate guard budgets are14 GB tracked
allocations plus14.4 GB RSS, total28.4 GB below30 GB. Postprocessing uses one
BLAS/OMP/MKL/NumExpr thread and a4 GB guard on the login host; the solver retains
OMP8. An initial OpenBLAS thread-limit failure was diagnosed and fixed without
rerunning the completed solver job.

`validate_magnetic_pilot.py` checks each completed job. Restarted jobs use the
native source checkpoint as their initial time/cycle reference. The guarded
`submit_magnetic_continuation.py` requires successful PBS completion, matching
final checkpoint/fluid state, unchanged build hashes, an empty debug-scaling
queue, and a unique receipt written before qsub. Unknown receipts are never
retried automatically. Output counters are preserved; forced-final dumps can
create gaps in the saved cadence, so actual timestamps are always used.

`watch_magnetic_pilot.py --job 8938200 --continue-jobs` follows accepted receipts
under an exclusive lock to the20000 M target. Failures stop it for review.
The remote root is /lus/flare/projects/MHDTidal/hzhu/cyclic_zoom_profile_20261013.
A magnetic-monitor.stop file there prevents further submissions without
cancelling a running job; magnetic-monitor-state.json identifies the controller.
Only one monitor may run. No debug or debug-scaling jobs belonging to other
agents are cancelled or changed.

## Artifacts and limits

- `latest_finest_xy.png/json`: equatorial density, temperature proxy p/rho,
  and physical fluid-frame beta at9903.62 M; exact ADM topology matched.
- `latest_finest.png/json`: the latest downloaded actual finest saved state,
  currently9782.47 M, with density, temperature and fluid-frame beta.
- `physical-JOB/`: exact logs, provenance and rank/resource validation packets.
- `frames-long/`: completed longer frame controls and their passing audit.
- `gpu-roundtrip-8937389/`: successful full GPU topology test and resolution audit.
- `failed-8937358/`, `failed-8937379/`: diagnosed initialization memory-guard stops.
- `DEVELOPMENT_LOG.md`: chronological details, including superseded status notes.

The renderer requires a matching stationary ADM topology and converts Btilde
and Wv to physical fluid-frame b^2 before computing beta=2p/b^2. It masks
excision and supports xy/xz slices. It never substitutes frozen finer data for
an active coarse snapshot. A completed20000 M pilot would still be a short,
low-resolution experiment; statistical/spatial convergence, global conservative
mass balance and boundary-distance independence would remain unproven.

## Movie and boundary-layer assessment

The first-return movie and six-face diagnostics are in
[movie-first-return/README.md](movie-first-return/README.md). A mild polar
boundary adjustment is visible: adjacent face-mean density/pressure departures
reach0.75%/1.24%, with individual-cell departures1.61%/2.66%. No large boundary
instability is apparent, but reflection and inner-flow contamination remain
unexcluded without matched box-size/characteristic-boundary controls.

## Equatorial density/temperature/beta movie and excision audit

[xy-movie/README.md](xy-movie/README.md) documents the t=0..9903.62 M
equatorial movie, including physical beta on every active topology. Actual saved
headers confirm smooth_excision=0: this pilot uses hard puncture primitive resets
at0.8 M, with magnetic damping off. The user's smooth drain implementation is
present but was left disabled in the input. No current evolution was changed.
