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
