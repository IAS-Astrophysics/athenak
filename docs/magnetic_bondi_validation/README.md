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
The pre-existing Michel velocity limiter is unchanged.

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

An isolated source-magnetic/build-magnetic pair under
/lus/flare/projects/MHDTidal/hzhu/cyclic_zoom_profile_20261013 is being built
with the validated SANE compiler/Kokkos configuration. Existing hydro and SANE
production checkouts/executables are not overwritten. The latest source must
be synchronized and an incremental build completed after the current full
build, then a bounded magnetic GPU roundtrip validated before physical use.

scripts/tests/aurora_bondi_magnetic.pbs uses debug-scaling (newly authorized),
one node, twelve explicit GPU tiles, one-hour allocation and -t00:55:00.
Existing tracked-allocation16 GB plus aggregate rank RSS12 GB guards stay
below30 GB. Check qselect -u hzhu -q debug-scaling before submitting; do not
alter other jobs. No GPU submission has yet been made at this receipt.
The pilot physical target is20000 M; successful startup or a short smoke test
must not be reported as a relaxed magnetized binary flow.
