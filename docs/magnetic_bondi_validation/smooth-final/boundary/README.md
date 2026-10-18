# Outer reservoir audit of the completed smooth chain

Assessment: the inertial-rest conversion and shift-aware timestep handling are
consistent, and the completed run shows no explosive outer instability. The
fixed reservoir is not a demonstrated nonreflecting boundary, however. A growing
outer active-cell layer is measurable, and boundary-independent accretion has
not been established. No new evolution or boundary changes were made here.

The audited pgen and timestep source hashes match those archived from the final
Aurora executable. The velocity helper is src/pgen/dynbbh.cpp:2124, the six-face
callback starts at :2313, and the coordinate light-cone timestep bound is in
src/mhd/mhd_newdt.cpp:88–126.

For inertial coordinate rest, V_rot=-Omega cross r and
beta_rot=beta_inertial+Omega cross r. The Eulerian velocity is
(V_rot+beta_rot)/alpha=beta_inertial/alpha; the stored primitive is W times that
velocity. Thus the potentially large rotational terms cancel before the
Lorentz factor. The physical target is timelike-checked; an impossible boundary
state aborts rather than being silently velocity-limited. The GRMHD timestep
uses a conservative coordinate light-cone bound |beta^i|+alpha sqrt(gamma^ii)
at faces, including the shift. A shift larger than c is not itself a physical
superluminal gas speed or a loss of spacelike time slices.

For a=32 M, Omega=0.00552427/M and the light-cylinder radius is 181.019 M.
The box has faces at ±2048 M: Omega R=11.314 at an equatorial face center and
16 at an equatorial corner. The Cartesian cube is fixed in rotating coordinates,
so rotation has a normal component on its x/y faces; it is not tangent everywhere
as it would be on a centered sphere. Many face patches are dominated by one-way
coordinate advection. That increases the importance of the characteristic
boundary treatment even though the coordinate CFL bound is respected.

The callback resets exterior ghost density, pressure, inertial-rest velocity
and uniform vertical densitized magnetic field. It leaves physical normal
magnetic faces CT-evolved. These choices preserve the reservoir target and
avoid overwriting the evolved magnetic flux; they do not constitute outgoing
characteristic extrapolation or a wave-absorption condition. Fixing primitives
can reflect outgoing magnetosonic disturbances. The polar faces have zero
normal rotational shift, so a polar disturbance cannot simply be attributed
to an omitted normal-shift CFL term; tangential transport, magnetic geometry,
and reservoir response still matter.

The existing independent ghost tests cover inertial/rotating single holes and
a circular binary, including outside-light-cylinder patches. At 64^3 and t=2 M,
rotating/inertial exterior mean differences were density 0.0411%, pressure
0.513%, Btilde_z 0.0202%; ghost inertial-velocity errors were at float32 output
precision. These are short controls at smaller Omega R than the production
corners, not long-time reflection tests at the production parameters.

## Measured final active cells (t=29500 M)

Six cell-center layers inward are inspected, using unchanged outer-block metric
geometry. The nearest layer lies 64 M inward from each face. Face means are
area means since these exterior cells have equal area. Opposite polar faces
agree closely; corners enter each adjoining face's diagnostics.

| Quantity | Largest magnitude of a face mean | Largest local magnitude |
| --- | ---: | ---: |
| Density deviation from rho_inf | 2.50% (polar deficit) | 6.80% |
| Pressure deviation from p_inf | 4.11% (polar deficit) | 11.26% |
| Btilde_z deviation from Binf | 1.50% (polar excess) | 8.44% |
| Inertial coordinate speed / c | 0.00217 | 0.00523 |

The polar density/pressure deficit and B_z excess still grow over the sampled
smooth-chain interval. Transverse Btilde reaches about 3.3% of Binf locally.
These are departures of evolved active gas from the reservoir, not errors in
prescribed ghost values. Some response is physically expected from initially
uniform magnetized gas in a gravitational potential. The present data cannot
separate physical adjustment, reflected waves and discretization effects.
The plot has a separate t=0 calibration point and no interpolated trajectory
across the unsampled pre-switch interval. Initial inertial-rest cancellation
is recovered to 6.41e-7 c from float32 saved output.

The boundary is only 4.096 rB away along a face normal. An ambient maximum-fast
speed estimate gives c_fast≈0.0936 c and travel time from rB=500 M to a face
about 16500 M, shorter than the final global time. Actual propagation depends
on the evolving magnetized flow, but causal isolation of the reservoir from
boundary feedback cannot be assumed.

## Before claiming a boundary-independent result

A controlled magnetosonic pulse/reflection test and matched physical-boundary
inertial/rotating controls at production Omega R would be useful. Incoming
characteristics should maintain the ambient reservoir while outgoing ones
are allowed to leave; magnetic normal flux and edge EMFs must remain CT
consistent. Larger-box and outer-resolution comparisons would test sensitivity.
A spherical reservoir surface would eliminate rotational motion normal to that
surface; it would be a different numerical boundary design, not a change to
these completed results. Current several-percent layers should not be declared
harmless on the basis of positivity or divB alone. No further simulations were
submitted as part of this audit.

`outer_boundary.png` shows the six-face time series and final pressure profiles.
`profiles.json` and `summary.json` retain numerical data. These outputs are
active-cell diagnostics; ghost cells and actual boundary face fluxes were not
saved in the production chain, so no nonreflection or conservative boundary
mass-budget claim is possible.
