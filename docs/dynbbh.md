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

The old `adjust_mass1` and `adjust_mass2` inputs are rejected.  Table masses are
the physical Kerr-Schild masses and are never silently rescaled.

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
