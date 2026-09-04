# Dynamic binary black-hole problem

The analytic binary uses `problem/q = M2/M1`, total mass one,
`M1 = 1/(1+q)`, and `M2 = q/(1+q)`.  `a1` and `a2` are dimensionless spin
magnitudes (`chi`), while `th_a*` and `ph_a*` are angles in degrees.  The metric
constructs each physical Kerr vector as `a_i = M_i chi_i`.

An optional smooth analytic-orbit spin ramp is enabled with
`problem/spin_ramp = true`.  `spin_ramp_start_time` and
`spin_ramp_timescale` select a cubic smoothstep from zero to the requested
dimensionless spins.  The ramp affects the prescribed background only; it is
not a self-consistent spin evolution.

## Trajectory tables

Set `problem/use_traj_table = true` and provide `problem/traj_file`.  Each
non-comment row contains 21 whitespace-separated finite numbers:

```text
t m1 m2 x1 y1 z1 x2 y2 z2 chix1 chiy1 chiz1 chix2 chiy2 chiz2 vx1 vy1 vz1 vx2 vy2 vz2
```

Times must increase strictly, masses must be positive, and each spin vector
must satisfy `|chi| <= 1`.  The table must cover the simulation's current time
through `time/tlim`.  Positions use cubic Hermite interpolation with the
supplied velocities; the interpolated velocities and accelerations are the
exact first and second derivatives of that polynomial.  Masses and
dimensionless spins use linear interpolation with matching time derivatives.
At an interior table knot, derivatives are taken from the segment to the
right; the final endpoint uses the last segment.  Endpoint and evaluated
interpolated velocities must remain subluminal.  For smooth AD/FD agreement,
tables should avoid jumps in acceleration or in the mass/spin slopes at knots.
A minimal five-row elliptical-orbit example with tilted spins is provided in
[`tst/inputs/dynbbh_elliptical_spin.traj`](../tst/inputs/dynbbh_elliptical_spin.traj).

The old `adjust_mass1` and `adjust_mass2` inputs are rejected.  Table masses are
the physical Kerr-Schild masses and are never silently rescaled.

## Moving excision and sink controls

Set `coord/excision_scheme = puncture` to center two excision masks on the
instantaneous binary trajectory.  Their Kerr-shaped distances include each
hole's physical spin vector and boost.  Positions, velocities, spins, and
masses are refreshed at the correct explicit Runge--Kutta stage time; the mask
is then rebuilt before that stage's primitive recovery.  This is also done
during initialization, and `coord/excise_shrink_start_time` is an absolute
simulation time so a restart does not restart the radius transition.

`coord/excise_1_rad` and `coord/excise_2_rad` set explicit radii.  A
non-positive radius selects the instantaneous Kerr horizon radius
`M(1+sqrt(1-|chi|^2))`.  `excise_to_horizon` always uses that radius,
`excise_cap_to_horizon` caps a fixed requested radius, and
`excise_horizon_fraction` (default 1, range `(0,1]`) scales the horizon target
of the automatic, capped, direct and shrinking radius modes. This permits an
interior target such as `0.8*rH` without changing the current mass/spin tracking.
See [multi-stage disk zoom](dynbbh_zoom.md) for a portable restart workflow.
`excise_shrink_to_horizon` transitions from the requested radius to the
horizon with a cubic smoothstep over `excise_shrink_timescale`.  The last two
time controls are `excise_shrink_start_time` and
`excise_shrink_timescale`.

`coord/require_resolved_horizon = true` makes an under-resolved horizon a
fatal setup error.  Otherwise the code warns.  An optional historical fallback,
`problem/unresolved_sink = true`, exponentially relaxes conserved density,
momentum, and energy toward configurable floors only while a hole has fewer
than `sink_resolved_cells_across_horizon` cells across its horizon.  Its
radius is at least `sink_cells_per_radius` local cells (and
`sink_radius`, when positive), its transition width is `sink_width` or an
automatic local value when negative, and its physical drain time is
`sink_timescale`.  The fallback supplements rather than disables geometric
excision.  It currently requires MHD.

`coord/smooth_excision = true` replaces the hard primitive reset by a compact
smooth projection toward `dexcise`, `pexcise` (or `texcise`), and the
puncture's coordinate velocity.  The profile is controlled by
`smooth_excision_puncture_width_fraction` and
`smooth_excision_puncture_weight_exponent`; the first-order flux mask may be
expanded with `puncture_flux_excision_radius_factor`.  Optional protections
are `smooth_excision_sigma_max`, `smooth_excision_temp_ceil`, and a minimum
puncture-frame radial inflow selected by `smooth_excision_inflow` and
`smooth_excision_inflow_speed`.

Magnetic damping is opt-in through `smooth_excision_b_damping`,
`smooth_excision_b_damping_eta`, and `smooth_excision_b_damping_cfl`.
It adds the resistive EMF `eta W curl(B)` on edges using the strict minimum of
neighboring cell weights.  Magnetic face fields are never reset directly, so
constrained transport continues to control the discrete divergence.  The
regression-only `problem/test_bz_gradient` seeds a divergence-free linear
field used to exercise this path; production inputs should leave it at zero.

## CBD refinement policies

Adaptive runs can set `problem/amr_condition` to `tracker`, `alpha_min`, or
`none` and enroll a user AMR criterion.  The legacy spelling `track` remains
accepted as an alias for `tracker`; unknown values are rejected.  Tracker
regions follow the two instantaneous analytic or tabulated puncture positions.
Their radii are `tracker_1_rad` and `tracker_2_rad` (defaulting to the legacy
`radius_thr`), and their target physical levels are `tracker_1_reflevel` and
`tracker_2_reflevel`.  The common `tracker_reflevel` supplies both defaults.

COM-centered spherical regions are added with `radius_N_rad` and
`radius_N_reflevel`, for `N=0` through 15.  Regions may be nested or overlap
the moving trackers; the highest requested level always wins.  Physical level
zero is the root mesh and `-1` means refine to the configured maximum AMR
level.  Explicit levels outside the configured adaptive range are rejected.
The point-to-MeshBlock-box distance is exact, including when a puncture or the
COM lies inside a block.

`problem/refinement_hysteresis` (default 1.25, minimum 1) expands each region
only for retention: a block at the requested level is not derefined until it
leaves the expanded radius.  This suppresses boundary chatter without
refining the buffer itself.  AthenaK's normal `refinement_interval` remains in
effect.  Other refinement criteria retain priority when they request further
refinement.

## Flux surfaces

With `problem/user_hist = true`, fixed COM-centered spheres are selected by
`flux_rsurf_inner`, `flux_rsurf_outer`, and `flux_dr_surf`.  The angular grid
uses `flux_ntheta`, `flux_nphi`, and `flux_interp_order`.  Optional moving
spheres use `flux_horizon1`, `flux_horizon2`, `flux_radius1`, and
`flux_radius2`; their centers are refreshed from the analytic or tabulated
trajectory before every history evaluation.

For each surface, the user history reports inward-positive rest-mass and energy
fluxes; outward linear- and angular-momentum fluxes, each split into fluid and
electromagnetic parts; unsigned magnetic flux; and proper area.  The ordered
labels are `mdot`, `edot_f`, `edot_em`, `pxdot_f`, `pydot_f`, `pzdot_f`,
`pxdot_em`, `pydot_em`, `pzdot_em`, `lxdot_f`, `lydot_f`, `lzdot_f`,
`lxdot_em`, `lydot_em`, `lzdot_em`, `phiB`, and `area`, followed by the
surface label.  Magnetic flux uses `0.5 integral |B^i dSigma_i|`.  Surface
objects are owned by the problem generator and rebuilt after AMR changes
without static view lifetimes.

## Angular momentum and torque outputs

`variable = angular_momentum` writes six densitized volume integrands:
`Jx`, `Jy`, `Jz` for the fluid and `JEMx`, `JEMy`, `JEMz` for the
electromagnetic field.  They use the covariant Eulerian momentum density,
`sqrt(gamma) epsilon_(alm) x^l S_m`, which is the Cartesian rotation charge.

`variable = torque` writes `Tx`, `Ty`, and `Tz`.  These include both the
cross product of position with the ADM momentum source and the product-rule
term from the spatial momentum flux.  Both outputs require ADM, MHD, and
DynGRMHD objects and assume the gamma-law EOS used by this problem.  Primitive
velocities are interpreted as `W v^i`; cell-centered magnetic fields are
interpreted as the densitized `sqrt(gamma) B^i` representation used by
DynGRMHD.

Torque uses the current `Dx<4>` stencil and therefore requires at least three
ghost zones.  Cells with a non-finite or non-positive spatial determinant, or
with invalid primitive state, are written as zero rather than NaN; the
simulation's floor/event diagnostics should be consulted when this occurs.
These vector-valued diagnostics cannot be used directly as scalar PDF axes.

The maintained native-output post-processing workflow is documented in
[`cbd_diagnostics.md`](cbd_diagnostics.md).  It produces AMR-aware radial
profiles, rest-mass and angular-momentum budgets, torque summaries, and radial
or moving-horizon flux time series without hard-coded run paths.

## Dynamical-spacetime radiation

Use a `<dyn_radiation>` block, rather than the legacy `<radiation>` solver, for
radiation transport on the time-dependent ADM background.  The dynbbh problem
requires `geometry = adm`.  For example:

```text
<dyn_radiation>
geometry = adm
nlevel = 1
angular_fluxes = true
reconstruct = plm
rad_source = true
kappa_a = 0.01
kappa_s = 0.01
kappa_p = 0.0
arad = 1.0
```

On a fresh run, dynbbh initializes the angular intensities to an isotropic LTE
field in the fluid frame using the torus temperature and the instantaneous ADM
tetrad.  The stored intensities include the `sqrt(gamma)` conservative
normalization used by the transport solver.  Atmosphere cells and invalid
metric or thermodynamic states are initialized to zero rather than a non-finite
intensity.  When a `<units>` block is present, the code-unit radiation constant
is derived from it; otherwise a positive finite `dyn_radiation/arad` is
required.

Dynamic-radiation intensities are included in normal restart files.  To enable
radiation while reading an older restart that has no `<dyn_radiation>` block or
intensity array, supply the new block's parameters on the restart command line,
including `dyn_radiation/allow_missing_restart_i0=true`.  This explicit opt-in
is the only case in which AthenaK permits a missing input block to be created by
command-line parameters.  Dynbbh then seeds the new field after primitive
recovery.  The optional
`dyn_radiation/restart_seed_erad_fraction` (default one) scales that LTE seed;
cells at the fluid floors or inside excision are left dark.

The solver uses Kokkos execution spaces throughout its transport, ADM tetrad,
geometric source, matter-coupling, AMR, and initialization kernels.  The compact
coupled regression input is
[`tst/inputs/dynbbh_radiation.athinput`](../tst/inputs/dynbbh_radiation.athinput).
