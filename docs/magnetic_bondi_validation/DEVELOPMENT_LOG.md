# Development and run history

This preserves chronological diagnostics, including superseded status notes.
Use README.md for the current workflow and interpretation.

# Magnetized binary Bondi pilot

The requested pilot uses the uniform-background setup of Guo et al.2025,
Section III.3, rather than the analytic Michel inflow used in the previous
hydro experiment: https://arxiv.org/html/2504.16802v2#S3.SS3.
Input: inputs/dyngr/bondi_hierarchy/binary_magnetized_lowres.athinput.

Total mass is1, q=1, separation32 M, both spins0, gamma=5/3, rho_inf=1,
rB=2GM/cs_inf^2=500 M, beta_inf=1, with an initially vertical field and2%
random pressure perturbations. No cavity taper is used. Beta is normalized
asymptotically via B_inf=sqrt(2P_inf/beta_inf); local physical beta varies with
the curved metric. The constant field stored by Valencia CT is densitized
B_tilde^z=sqrt(gamma)B_Eulerian^z, so its initial discrete divergence vanishes.
The pressure uses the relativistic gamma-law sound-speed relation.

## Inertial rest and boundaries

The same helper sets initial and boundary velocity. For uniform gas its
inertial coordinate velocity is zero, rotating coordinate velocity is
(Omega*y,-Omega*x,0), and stored primitives are

```
v_Eulerian = (v_coordinate + beta_shift) / alpha
W = (1 - gamma_ij v_Eulerian^i v_Eulerian^j)^(-1/2)
u_primitive = W v_Eulerian
```

Initial rest is exact where timelike and within the configured Lorentz ceiling
(gamma_max=10); the EOS applies that ceiling in a thin high-W region. Where inertial-coordinate rest is
spacelike near the holes, it is impossible to impose; only those initial
cells are limited to v_Eulerian^2=0.9, with physical puncture excision applied
by the solver. Boundary data must be timelike and are never silently limited.
The pre-existing Michel velocity limiter is unchanged. A negative single-hole
control with its boundary inside the static limit confirms explicit rejection.

All six outer boundaries prescribe uniform density, pressure, inertial rest,
and vertical field in exterior ghosts. Random perturbations are initial only.
Normal magnetic faces on the physical boundary remain CT-evolved; only exterior
ghost faces are prescribed. Tests verify the evolving field in deeper exterior
ghosts; the first ghost cell averages in the evolved physical normal face. The actual flux comes from the Riemann problem
with the interior state; it is not explicitly fixed to a Bondi accretion rate.
This is a fixed reservoir, not a nonreflecting characteristic boundary, and
subsonic disturbances can reflect. Boundary-distance sensitivity remains a
production requirement. The box faces are at2048 M (4.096 rB).

## Mesh and cyclic timing

The coarse root grid is32^3 with8^3 active cells per MeshBlock. Thirteen
physical refinement levels have a minimum dx=0.03125 M=m_hole/16 near the
punctures. The nominal single-hole horizon sphere r=1 M has dx=0.0625 M
(m_hole/8); this minimum-spacing statement is not uniform horizon resolution.
Only eight inner levels participate in cyclic zoom. Thus r_z/dx=16 at the
corresponding zoom levels, while the widest mask radius is64 M, inside rB.
The complete initial mesh has1408 blocks. This remains a coarse outer-grid
pilot, not a spatially converged reproduction of the paper.

bondi_mhd_schedule selects Eq20: finest dwell500 local t_g=250 M; stages1–2
have zero prescribed dwell; other stages use min(50 t_dyn,0.15 tB). The clock
uses each hole's mass and hole-centered radius below a, then total mass and
COM radius at r>=a. tB=2GM/cs_inf^3 uses the shared ambient conditions.
Magnetic EMF corrections and CT remain enabled. residence_scale=1 for the
physical pilot; accelerated values are only for bounded topology tests.

## Completed local checks

- Initial and exterior ghost inertial velocities recover zero within2e-7
  from float32 output, for single-hole inertial/rotating cases and a binary.
  Omega=0.2 with box8 M exercises a boundary outside the coordinate light
  cylinder. The radius and indices include exterior ghost-cell coordinates.
- The finer32^3 frame control evolved to1 M; inertial/rotating differences
  are0.0513% density,1.31% pressure and0.0352% vertical field. All improve
  from16^3. The coarse pressure difference is6.97%, so these are convergence
  controls rather than claims of exact equality at finite resolution.
- A five-level magnetic hierarchy completed two accelerated outward/inward
  cycles in18 steps, with finite positive fields and max|divB|=2.15e-16.
  Peak RSS was3.44 GB, elapsed45 s. The test is intentionally underresolved
  and verifies topology/CT, not physical relaxation.
- Restart from its middle checkpoint reproduces final float32 fields within
  6e-8 absolute (magnetic fields within1.5e-8); divB stays2.14e-16. It is
  not bitwise identical for every magnetic/velocity field. Saved comparisons
  use rtol5e-6 and atol2e-8, and identical time, cycle and mesh geometry.
- The full1408-block pilot initial state has zero discrete divB, positive
  finite primitives, and peak RSS6.02 GB. No evolution is claimed by that
  initialization check. Validation JSON and resource receipts accompany this file.

Reproduce the frame tests with scripts/tests/test_magnetic_reservoir.py under
run_bounded.py. The accelerated input is saved as roundtrip.athinput; use
mhd/dyn_flux_team_size=0 on CPU and compare the restart at cycle9 with cycle18.

## Aurora execution status

The isolated source-magnetic/build-magnetic pair under
/lus/flare/projects/MHDTidal/hzhu/cyclic_zoom_profile_20261013 completed the
final forced problem-generator rebuild with the validated SANE environment.
The final build took 332 seconds with peak RSS 5.00 GB. Source and executable
hashes are checked again by the batch script at launch. Build receipts are
in aurora-build/. Existing hydro/SANE production files are not overwritten.

The first GPU test, job 8937358, stopped before fluid initialization: the
Kokkos guard rejected 1.431 GB/rank against its 1.333 GB/rank budget. PBS exit
was 143 after MPI terminated the remaining ranks. Evidence is archived in
failed-8937358/. There are no fluid snapshots or relaxation results from it.

The mesh contains 1408 blocks, distributed as 117–118 per rank; capacity220
reserved unnecessary array space. Capacity160 retains the identical mesh,
physical resolution, field, dwell prescription and memory guards. Load
balancing checks the new per-rank block count against capacity on every AMR
change. The accelerated full-hierarchy test will verify this choice dynamically.
Replacement job 8937379 was accepted in debug-scaling; its predecessor/reason
are recorded in magnetic-smoke-cap160.receipt. It uses one node and twelve
explicit GPU tiles, 18 steps and a five-minute solver cutoff. The isolated
rank wrapper was synchronized with the current validated wrapper before this
queued job started, removing an inherited hardcoded420-second timeout and
restoring per-rank logs. Its hash is in magnetic-smoke-cap160-runtime.sha256.
The test must pass
scripts/tests/validate_magnetic_pilot.py --roundtrip before physical use.
Job 8937379 initialized the full GPU mesh with finite positive fluid and
zero discrete divB, but its first timestep exceeded the 1 GB rank RSS guard
(1.008 GB observed). Initial validation and failure evidence are archived in
failed-8937379/. Its allocation guard did not trigger.

The next test, job8937389 (magnetic-smoke-cap128.receipt), uses capacity128.
The exact saved-parent restoration cannot enlarge the original1408-block
tree, whose balanced requirement is118 blocks/rank. This retains10 spare
slots and does not lower physical resolution. The per-node budgets are now
14 GB combined tracked allocations plus14.4 GB aggregate RSS (1.2 GB/rank),
total28.4 GB. The wrapper's optional allocation-budget setting preserves its
16 GB default for unrelated workflows. The revised full-hierarchy test8937389 completed18 steps normally and passed
all12 rank checks: finite positive fluid, nonzero evolving field, all8 zoom
stages and a return to the exact original finest topology. Maximum discrete
divB was3.72e-15. The sum of peak rank RSS was11.46 GB; tracked allocation
peaks summed to12.34 GB, giving a conservative23.80 GB total below30 GB.
Elapsed rank runtime was about36 seconds. The test's global time2.30 M is
accelerated topology validation, not physical relaxation. Exact receipts and
all rank logs/reports are in gpu-roundtrip-8937389/.

Physical pilot8937401 was accepted after this validation, from fresh initial
data with residence_scale1, no cycle limit, target20000 M and the55-minute
clean solver cutoff. magnetic-physical.receipt records it. It started on one node at23:48 UTC. The runtime log confirms the250 M
first dwell, all12 explicit tiles, and forward evolution with dt=0.007260854 M.
Its final result remains pending; neither target completion nor inflow
equilibrium is claimed.

scripts/plot_magnetic_bondi_snapshot.py renders density, p/rho and physical
fluid-frame beta from matching stationary ADM and fluid snapshots. It requires
identical saved topology, converts Btilde to Eulerian B with sqrt(gamma), and
uses b^2=(B_i B^i+(B_i u^i)^2)/(1+u_i u^i). Excision cells are masked. Initial
rendering was inspected locally; an evolved physical-run figure remains pending.

scripts/tests/aurora_bondi_magnetic.pbs uses one-hour allocations. The physical
pilot uses -t00:55:00 and residence_scale=1. The tracked allocation budgets
sum to14 GB and rank RSS budgets sum to14.4 GB, below the30 GB ceiling.
Check qselect -u hzhu -q debug-scaling before submission and never duplicate
an accepted receipt or alter other jobs. The physical target is 20000 M;
startup or an accelerated test does not establish a relaxed magnetized flow.

## Continuations and completion checks

validate_magnetic_pilot.py checks every saved fluid/divB state, all twelve
rank exit/resource reports, one host, tile affinities, the physical55-minute
cutoff and normal solver termination. Physical runs may stop on the wall clock
or target time, never an accelerated cycle limit. Metric evaluation counts are
recorded to audit reuse of the stationary geometry.

submit_magnetic_continuation.py --predecessor JOB runs only after PBS reports
a successful completed job and its validation.json exists. It decodes the
actual final checkpoint's time/cycle/block count and physical MHD parameters,
compares them with the final fluid state, and preserves output counters with
CZ_RESTART. It checks debug-scaling is empty and verifies build hashes, then
writes a unique predecessor receipt before qsub. Existing or unknown receipts
are never automatically retried. The helper refuses live predecessors (tested
against8937401), accelerated residence times, failures, and targets above20000 M.
The current double-precision header decoder was checked against both local
full-pilot initialization checkpoints. No running solver/input is modified.

## Longer frame control and early physical diagnostic

The32^3/64^3 controls reached2 M with finite positive fields and exact exterior
reservoir data. Fine-grid inertial/rotating differences are0.0411% density,
0.513% pressure and0.0202% vertical field, improving from0.151%,2.50% and0.0969%.
Initial-rest checks now use the actual metric light cone and configured Lorentz
ceiling, rather than isolated-hole distance heuristics. The very tight a=4 M
binary has a combined static-limit region reaching farther than isolated-hole
spheres. Spacelike initial cells are checked against the intended v^2=0.9
limiter; high-W cells subject to gamma_max=10 are explicitly counted. The
outer ghost velocity error remains below2e-7. The initial300-second fine-case
budget was insufficient because rotation reduces the CFL step;600 seconds
was used thereafter. Completed datasets were audited only after every requested
parameter matched their saved headers. frames-long/ includes normal solver
logs for all six completed cases, the final passing audit, and the computation
resource receipt (3.79 GB peak, whose earlier heuristic-rest assertion failed).
No fluid solver changes were needed for this test-harness correction.

At t=50.0055 M, the physical run is finite and positive and has developed
magnetized near-hole inflow. Density/temperature/fluid-frame-beta views are in
physical-8937401/early_flow.png (xy) and early_poloidal.png (xz). These are early
transients, not relaxed flows. The largest densitized fields are inside
puncture excision: coordinate norm2131 there versus7.45 outside. Raw discrete
divB is5.38e-10 globally and4.62e-12 outside excision. A field-scaled cellwise
proxy dx*abs(divBtilde)/max(norm(Btilde),0.001*ambient) is2.14e-13. The validator
retains both absolute measures and now gates on the dimensionless proxy<1e-9;
an absolute-only cutoff is inappropriate when interior field amplitudes grow.
The proxy uses cell-centered fields and is not an invariant face-flux norm.

A24x48 angular audit found r_mask/dx=16 at every sampled mask-boundary point
in all8 GPU zoom stages, including merged COM stages6 and7. Resolution audit
metadata are archived beside the successful GPU test. This does not establish
spatial convergence or resolve a true binary apparent horizon.

## First physical clean stop

Job8937401 stopped normally on the55-minute solver wall clock after16244
steps at t=117.9453202306 M. All12 rank exits were0. Every saved fluid/divB
state passed validation. The conservative sum of peak RSS and tracked
allocation peaks is23.56 GB (11.23+12.33), below the28.4 GB guard sum and30 GB
ceiling. Raw divB reaches2.67e-9 inside excision and1.26e-11 outside; the
field-scaled maximum is4.73e-13. The large interior fields (coordinate
Btilde norm5167) are excluded from physical interpretation; the exterior
maximum is14.0. Each rank evaluated the stationary metric once over all16244
steps. All exact logs/reports are in physical-8937401/.

The first250 M finest dwell remains incomplete, so no physical zoom cycle or
inflow equilibrium has been established. latest_finest.png shows the actual
117.945 M xz snapshot. After checking the final checkpoint's native header,
hash, time/cycle/block count and physics, continuation8937572 was accepted
from magnetic_bondi_binary.00001.rst toward20000 M. Its unique predecessor
receipt is magnetic-continuation-8937401.receipt. It retains one node/all12
tiles, physical dwell times, output counters and the55-minute clean cutoff.
This continuation started at00:48 UTC. Its restored log retains zone0 and
next_time=250 M, so the dwell clock was not reset; its final result is pending.

## Guarded chain monitor

watch_magnetic_pilot.py --job8937572 --continue-jobs follows only accepted
predecessor receipts up to the20000 M target. It holds an exclusive monitor
lock, observes PBS directly, validates every completed successful job under a
4 GB postprocessing guard, and calls the tested submission helper only when
debug-scaling is empty. Failures, unknown receipts or diagnostic errors stop
the monitor for review; they never trigger a blind resubmission. The monitor
runs lightweight orchestration on the login host, not fluid evolution. A
magnetic-monitor.stop file in the remote pilot root prevents further submissions
without cancelling a running job. magnetic-monitor-state.json records its PID
and current job. Queue/observation waits use50-second intervals.

Restart validation uses the native source checkpoint as its initial time/cycle
reference, so a legitimate short final continuation with only one final fluid
dump can still be audited. Fresh runs and accelerated topology tests still
require initial and evolved snapshots. Saved output counters are preserved.
AthenaK's forced-final dump advances its output clock even between periodic
dumps, so continuation output times can have gaps; no uniform saved cadence
or interpolation across those gaps is assumed.

The full337-file src C++/header/CMake manifest is saved in aurora-build/.
336 files match the branch byte-for-byte. The only mismatch is a stale MPI-tag
comment in bvals.hpp; source-comment-diff.patch records the exact benign
difference. No solver/EOS differences were found. Build-recipe hashes are
also saved; the SANE dependency/compiler-launcher configuration remains
intentionally recorded separately from the solver source.

The runtime environment copy had one stale difference: it lacked the branch's
explicit EnableImplicitScaling=0. Tile affinity masks were already explicit
and correct. environment-original.sh preserves the exact original recipe;
the current branch environment has now been synchronized for subsequent jobs.
The ongoing8937572 job retains its already-sourced environment. Future batch
launches record EnableImplicitScaling, device hierarchy, GPU-aware MPI and OMP
thread count explicitly in runtime-controls.txt, alongside script hashes.
The build CMake file, build script and compiler launcher match the branch.

## Second physical clean stop and postprocessing thread guard

Job8937572 stopped normally at t=238.4319395305 M, cycle32838. All12 ranks
exited0; the conservative peak sum is23.45 GB. Every saved state is finite
and positive; max field-scaled divB is5.80e-13 and max exterior absolute divB
is3.33e-11. This remains the initial250 M finest dwell, without a physical
zoom transition yet. The native restart/cycle/output counters were preserved.

The monitor correctly stopped before submission when the first postprocessing
attempt failed during NumPy import: OpenBLAS tried64 threads and pthread
creation failed on the shared login host. This was not a solver failure.
validation-openblas-failure.json retains the resource evidence. Postprocessing
now explicitly uses one BLAS/OMP/MKL/NumExpr thread; the bounded retry passed
in1.88 seconds at117 MB RSS. Solver OMP remains8 as in the validated runtime.

After that successful validation and an empty debug-scaling queue,
continuation8937793 was accepted from magnetic_bondi_binary.00002.rst.
magnetic-continuation-8937572.receipt records the exact checkpoint hash and
native time. The fixed monitor follows8937793; the original monitor terminated
on the postprocessing error and did not duplicate a job. Subsequent launches
use the synchronized explicit EnableImplicitScaling=0 runtime setting.

The first physical zoom transitions occurred after the250 M finest dwell in
job8937793. The saved t=300.022 M state has1072 active blocks, maximum physical
level9 (zone3), finite positive fluid, and max field-scaled divB5.05e-13.
This is real physical-duration evolution, distinct from the accelerated smoke
test. first-coarse-validation.json records the check; runtime-controls.txt
confirms EnableImplicitScaling=0, composite hierarchy, GPU-aware MPI and OMP8.
No physical roundtrip has completed yet at this receipt.

## First physical magnetic roundtrip (2026-10-17)

Jobs8937793,8938001,8938136,8938200 passed all saved-field and twelve-rank
resource validations. The first finest return is logged at9683.038 M in
physical-8938200/rank-0.log; latest saved time9782.47 M restores the exact
initial ADM topology. The latest_finest figure was rerendered and inspected.
The SSH observation handle disappeared after8938200; authoritative PBS state
was F/exit0 and no live monitor process remained. Restarting the exclusive
monitor on that accepted predecessor revalidated it and produced the unique
accepted continuation8938971. No simulation was rerun or duplicated.
Target remains20000 M; no equilibrium or spatial-convergence claim.
