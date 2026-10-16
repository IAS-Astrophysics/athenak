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

Initial rest is exact wherever timelike. Where inertial-coordinate rest is
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
physical refinement levels resolve the holes at dx=0.03125 M=m_hole/16.
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
clean solver cutoff. magnetic-physical.receipt records it. Its result remains
pending; neither target completion nor inflow equilibrium is claimed.

scripts/tests/aurora_bondi_magnetic.pbs uses one-hour allocations. The physical
pilot uses -t00:55:00 and residence_scale=1. The tracked allocation budgets
sum to14 GB and rank RSS budgets sum to14.4 GB, below the30 GB ceiling.
Check qselect -u hzhu -q debug-scaling before submission and never duplicate
an accepted receipt or alter other jobs. The physical target is 20000 M;
startup or an accelerated test does not establish a relaxed magnetized flow.
