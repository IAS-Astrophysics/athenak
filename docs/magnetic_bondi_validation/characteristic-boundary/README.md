# Incoming-characteristic reservoir boundary: formulation and staged tests

An experimental opt-in implementation now exists in the full GRMHD solver;
see [IMPLEMENTATION.md](IMPLEMENTATION.md) for its design, current tests and
remaining validation. It is **not approved for production use**. The original
linear prototype and failed-job record below remain historical provenance.
The completed smooth-excision experiment is unchanged. Corrected capacity
control job 8941753 completed; all four fixed-boundary controls passed their
field/resource audits. A separate characteristic build is in progress.

## Boundary prescription

The ambient reservoir is specified in the inertial frame: rho_inf, p_inf and
vertical magnetic field, with inertial coordinate velocity zero. Transform this
state using the local lapse, shift and spatial metric; do not impose zero
rotating-coordinate velocity. For a rotating Cartesian face, the correct wave
classification is relative to that face, including its normal grid velocity.

At a face let U be the conservative MHD state, with the face-normal magnetic
field treated as the CT constraint rather than an independent incoming wave.
Linearize the normal coordinate flux at a positive physical reference state:

    A_n = dF_n/dU = R Lambda L,  L R = I.
    P_in  = R diag(lambda_a < -epsilon) L
    P_out = I - P_in
    U_boundary = U_ref + P_out (U_inside-U_ref)
                       + P_in  (U_reservoir-U_ref).

Normals are outward here; eigenvalues are coordinate speeds relative to the
boundary. Zero-speed modes are retained from the interior. In a local
orthonormal frame the normal speeds must be mapped back with lapse, shift and
normal metric factors; classification by inertial radial velocity, Eulerian
velocity alone, or the sign of a Cartesian primitive velocity is insufficient.
If every characteristic enters, prescribe the reservoir. If every characteristic
leaves, extrapolate the interior. A physical subsonic ambient boundary usually
has both incoming and outgoing modes.

The projector is a local, linearized construction. For finite perturbations,
reference-state selection, positivity, EOS consistency and reflection must be
validated. It is not an exact nonlinear transparent boundary and cannot remove
coordinate CFL constraints. A positivity fallback must be counted and reported.

For a production implementation, construct boundary **face fluxes and edge
EMFs** consistently. Keep the physical face-normal B from CT, derive tangential
magnetic data from the characteristic state, and use consistent upwind EMFs and
a divergence-compatible ghost extension. Independently replacing all face B
components would violate this prescription. Multi-face corners require an
unsplit policy; sequentially overwriting corner states by each face is not a
validated solution. Extrapolate perturbations relative to the position-dependent
ambient state, rather than copying a rotating-frame velocity to a different
position. These integration items are not implemented by the Python prototype.

## Tests implemented and their limits

`scripts/tests/test_characteristic_reservoir.py` numerically differentiates the
special-relativistic ideal-MHD conservative state and flux to obtain the full
seven-wave Jacobian about inertial-rest gas. The local flat rotating-face model
includes constant normal and tangential shift; normal B is constant (1D CT).
It checks normal, tangential and oblique magnetic fields at seven normal shifts,
including +/-11.3137085, exceeding the production face signal speed. It checks
real eigensystems, decomposition residuals, projector idempotence, outgoing-mode
retention, incoming-mode removal and preservation of the exact ambient state.
Eight finite-volume RK2/upwind pulse tests cover two resolutions and four shifts.
They are an internally consistent linear-wave verification, not an independent
nonlinear reflection benchmark or a comparison against the old boundary.

Local result: all 21 projector cases and eight pulse tests passed. The pulse
scheme evolves characteristic amplitudes, algebraically equivalent to the
constant-coefficient coupled system. Uniform-state preservation here is exact
algebraic preservation; it does not imply that the production curved-space
solver preserves it to machine precision.

A supplemental local analytic/quadrature check demonstrates why a coordinate
diode is unsuitable. In a cube of halfwidth L rotating at Omega, each of four
vertical faces loses rho Omega L^3 if inward flux is clipped while outward flux
is retained. The box mass is 8 rho L^3, hence the **initial** fractional loss is
Omega/2 = 0.002762135864 per M for a=32. This is not a prediction of exponential
long-term depletion. The submitted prototype predates this supplemental local
check; its exact source hash is recorded in submission.json.

## Aurora controls

`scripts/tests/aurora_characteristic_boundary.pbs` requests capacity, select=1,
walltime=01:00:00. The unchanged validated smooth executable runs four fixed-
reservoir controls: inertial and corotating frames at 32^3 and 64^3, box +/-2048,
production Omega=32^-1.5, unperturbed beta=1 reservoir, t=64 M, no refinement.
All twelve GPU tiles use the existing SANE environment/binding and aggregate
28.4 GB guards. Each solver retains -t 00:55:00 and has a shorter 600-second
watchdog. There is no automatic resubmission. This short, coarse experiment
isolates exterior behavior and does not resolve horizons or test inner accretion.

The isolated characteristic prototype also runs on the allocated node. The GPU
controls still use the existing fixed boundary: their results must not be labeled
as results from a characteristic AthenaK implementation. Completion requires
checking every case's normal time-limit exit, all twelve resource and memory
reports, positive finite fields, divergence and frame-correct ghost state.
Job receipt and results reside at:

    /lus/flare/projects/MHDTidal/hzhu/cyclic_zoom_profile_20261013/
      boundary-characteristic-20261018/

Before adoption, the next required tests are a nonlinear boundary-face/CT
implementation, fast/slow/Alfven pulse reflection measurements at matched
inertial and rotating physical boundaries, grid refinement, then an AMR control.
A fixed rotating cube and a fixed inertial cube have different moving-face
physics; compare transformed local face fluxes or common interior regions,
not untransformed boundary primitive components.

## Capacity outcome: failed before GPU evolution

Job 8941632 finished with PBS exit status 143. The isolated characteristic
prototype passed on Aurora (21 projector cases, eight pulse cases), but the
first GPU control aborted during parameter parsing:

    Parameter 'smooth_excision' in block 'coord' on command line not found

This was a test-harness defect: AthenaK CLI overrides require pre-existing input
keys. It is not evidence of a boundary instability. No fluid outputs or GPU
allocation reports were produced; none of the four controls completed. Twelve
rank resource files were preserved, including empty files left by MPI teardown;
empty files do not establish rank resource usage. The solver error appears on
multiple ranks and completed resource reports report returncode 1. The launcher
then terminated remaining ranks. See failure-summary.json and aurora-failure/
for the exact submitted scripts, source hashes, scheduler record and rank logs.

The corrected harness now prepares smooth_excision and output1/output2
ghost_zones explicitly in a separate control input before CLI overrides.
`prepare_characteristic_control.py` preserves the submitted original input.
A local CPU initialization-only check passed (cycle 0, no evolution); its flux
team size was set to 0 for CPU compatibility, whereas the GPU script retains 64.
Any future allocation must supply a new absolute CZ_BOUNDARY_TRIAL directory;
the corrected script rejects reuse of a started trial directory.
No replacement job was submitted. The monitor is paused on this diagnosed
failure. Nonlinear boundary integration, GPU evolution and reflection validation
remain outstanding; the linear prototype passing does not replace them.
