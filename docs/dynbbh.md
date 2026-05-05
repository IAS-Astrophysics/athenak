# `dynbbh` Problem Generator

`src/pgen/dynbbh.cpp` initializes and evolves a circumbinary disk on a
superposed boosted Kerr-Schild binary-black-hole background.  The default path
keeps the historical behavior: two black holes follow a fixed circular
Keplerian orbit and the metric is built from the two superposed boosted
Kerr-Schild perturbations.  Optional paths allow a tabulated binary trajectory,
smooth horizon excision, magnetic-field damping through the constrained
transport EMF, an unresolved-horizon sink, and user-controlled AMR around each
hole.

The pgen is intended for GRMHD circumbinary-disk tests and production setups.
It can initialize a Chakrabarti/Fishbone-Moncrief-like torus, a magnetized torus
from a vector potential, and a moving BBH metric supplied either analytically or
from a table.

## Required Runtime Blocks

A typical run needs these blocks:

```ini
<coord>
general_rel = true
excise = true
excision_scheme = puncture

<adm>
dynamic = true

<problem>
rho_min = ...
rho_pow = ...
pgas_min = ...
pgas_pow = ...
rho_max = ...
r_edge = ...
r_peak = ...
```

For MHD runs, also provide an `<mhd>` block.  The unresolved sink and
smooth-excision magnetic damping both require MHD.

## Binary Trajectory

### Default Circular Orbit

If `problem/use_traj_table=false`, the code uses the legacy circular orbit:

```ini
<problem>
sep = 25.0
q = 1.0
a1 = 0.0
a2 = 0.0
th_a1 = 0.0
ph_a1 = 0.0
th_a2 = 0.0
ph_a2 = 0.0
use_traj_table = false
```

Here `sep` is the binary separation, `q` sets the mass ratio through
`M1=1/(q+1)` and `M2=1-M1`, and the orbital frequency is `sep^(-3/2)`.
The spin angles are in degrees.  This is the compatibility mode and should
remain unchanged for older input files.

### Tabulated Trajectory and Hole Properties

Set:

```ini
<problem>
use_traj_table = true
traj_file = path/to/bbh_trajectory.dat
```

The table must be whitespace-separated and contain one row per time sample with
columns in this exact order:

```text
Time M1 M2 x1 y1 z1 x2 y2 z2 chi_1_x chi_1_y chi_1_z chi_2_x chi_2_y chi_2_z vx1 vy1 vz1 vx2 vy2 vz2
```

Rules:

- Times must strictly increase.
- `M1` and `M2` must be positive.
- Each spin vector is interpreted as dimensionless Kerr spin `chi`.
- The loader rejects spin magnitudes larger than unity, within a small
  numerical tolerance.
- The table must cover the full simulated time interval.
  At startup, the pgen aborts if `time/tlim` is larger than the final table
  time.

The trajectory uses cubic Hermite interpolation for positions with the supplied
velocities.  The same interpolation gives velocity and acceleration, which are
used by the analytic metric derivatives.  Masses and spin vectors are linearly
interpolated.

Minimal example row:

```text
0.0 0.5 0.5  2.0 0.0 0.0  -2.0 0.0 0.0  0.3 0.4 0.2  -0.5 0.1 0.8  0.0 0.125 0.0  0.0 -0.125 0.0
```

## Metric and ADM Variables

The metric is assembled as a superposition of two boosted Kerr-Schild
perturbations.  Time derivatives are analytic through lightweight autodiff and
the Hermite trajectory derivatives; they no longer require finite-difference
metric calls.  `SetADMVariablesToBBH` writes the ADM fields and puncture
metadata used by the coordinate excision masks.

Mass scaling:

```ini
<problem>
adjust_mass1 = 1.0
adjust_mass2 = 1.0
```

In table mode, the physical spin parameter used by Kerr-Schild is
`a_i = chi_i * M_i * adjust_mass_i`.  In the legacy analytic mode, `a1` and
`a2` follow the historical input convention and are scaled by `adjust_mass*`.

## Torus Initialization

The pgen initializes a tilted torus plus an atmosphere:

```ini
<problem>
r_edge = 60.0
r_peak = 240.0
tilt_angle = 0.0
rho_min = 1.0e-5
rho_pow = -1.5
pgas_min = 0.333e-7
pgas_pow = -2.5
rho_max = 1.0
l = 0.0
n_param = 0.0
pert_amp = 2.0e-2
```

Notes:

- `r_edge` is the inner torus edge.
- `r_peak` is the pressure maximum.  If using the constant angular-momentum
  branch, set `l` consistently.
- `tilt_angle` is in degrees.
- `pert_amp` applies a random pressure perturbation inside the torus.

For a circumbinary disk, choose `r_edge` outside the binary orbit for production
disk runs.  For stress tests of accretion and sink behavior, smaller values can
be useful, but they should be treated as algorithm tests rather than
well-separated circumbinary initial data.

## Magnetic Field Initialization

For MHD runs, a vector potential seeds the magnetic field:

```ini
<problem>
potential_beta_min = 100.0
potential_cutoff = 0.2
potential_rho_pow = 1.0
potential_r_pow = 3.0
potential_falloff = 400.0
vertical_field = false
```

The magnetic field is initialized from the curl of the vector potential, so the
initial face-centered field is compatible with constrained transport.  The
rotation-axis vector-potential denominator is guarded in the implementation to
avoid `0/0` on the axis.

## Excision

Recommended puncture excision setup:

```ini
<coord>
excise = true
excision_scheme = puncture
smooth_excision = true
smooth_excision_width = 0.75
smooth_excision_inflow_speed = 0.40
smooth_excision_sigma_max = 1.0e5
dexcise = 1.0e-10
pexcise = 1.0e-18
require_resolved_horizon = false
```

The pgen updates `punc_0`, `punc_1`, spins, velocities, and horizon radii from
the current trajectory.  Excision remains active whenever `coord/excise=true`;
the unresolved sink does not disable puncture masks.

For the smooth-excision equations and constrained-transport discussion, see
`docs/smooth_excision_procedure.tex` and the compiled PDF.

## Cooling Source Terms

`dynbbh` can apply two independent optically thin/source-term cooling models.
Both act as isotropic fluid-frame energy losses and are converted to Valencia
conserved-variable source terms.  They can be used separately or together.  If
both are enabled, the pgen evaluates them in one combined source kernel so the
internal-energy floor is enforced once on the total cooling decrement.

### ISM Cooling

The historical `dynbbh` cooling path is enabled through the generic user-source
flag:

```ini
<problem>
user_srcs = true
```

This calls `ISMCoolFn(T)` from `src/srcterms/ismcooling.hpp`.  The code converts
the primitive pressure to a temperature proxy,

```text
T_cgs = (p/rho) * temperature_unit,
```

and evaluates a tabulated/fit interstellar-medium cooling function
`\Lambda(T)`.  In code units the comoving cooling emissivity is

```text
q_ISM = rho^2 * Lambda(T) / cooling_unit,
cooling_unit = pressure_unit / (time_unit * n_unit^2),
n_unit = density_unit / (mu * atomic_mass_unit).
```

The internal energy density obeys

```text
de_int/dt = -(alpha/W) q_ISM,
```

where `alpha` is the lapse and `W` is the Lorentz factor computed from the
spatial metric and the primitive `W v^i` variables.  The operator is subcycled,
and the instantaneous removal rate is capped so no more than approximately the
global CFL fraction of `e_int` is removed over a hydro step.  Cooling never
reduces the internal energy below `pfloor/(gamma-1)`.

The corresponding conserved-variable decrement over an applied comoving energy
loss `q dtau` is

```text
Delta tau = sqrt(gamma) * alpha * W   * q dtau,
Delta S_i = sqrt(gamma) * alpha * u_i * q dtau.
```

The source subtracts these quantities from `u0(IEN)` and `u0(IM*)`.
Because this path is a physical-cgs cooling curve, production inputs should set
the `<units>` block consistently, especially `density_cgs`, `bhmass_msun`, and
`mu` or equivalent code-unit scales.  Leaving the defaults in a dimensionless
test problem can make the cooling rate physically meaningless or overly stiff.

### Thin-Disk Orbital Cooling

The thin-disk cooling term follows the prescription described in the tilted
thin-disk paper: internal energy above a target scale height decays
exponentially on an orbital timescale.  Enable it with:

```ini
<problem>
thin_disk_cooling = true
thin_cooling_h_over_r = 0.03
thin_cooling_timescale_orbits = 1.0
thin_cooling_cfl = 0.5
thin_cooling_r_inner = 0.0
thin_cooling_r_outer = 1.0e300
```

At each cell, the code computes the same Boyer-Lindquist-like radius `r` used
by the torus initializer and applies the source only when
`thin_cooling_r_inner <= r <= thin_cooling_r_outer`.  The target sound speed is

```text
c_s,target = (H/R)_target * v_K,
v_K^2 = 1/max(r, 1).
```

Using the non-relativistic ideal-gas relation `c_s^2 = gamma p/rho`, the target
pressure and internal energy are

```text
p_target = rho * (H/R)_target^2 * v_K^2 / gamma,
e_target = max(p_target/(gamma-1), pfloor/(gamma-1)).
```

If `e_int <= e_target`, the source does nothing.  Otherwise the excess internal
energy decays over

```text
t_cool = thin_cooling_timescale_orbits * 2*pi * max(r,1)^(3/2),
Delta e = (e_int - e_target) * [1 - exp(-Delta t/t_cool)].
```

If `thin_cooling_cfl > 0`, the decrement is additionally limited by

```text
Delta e <= thin_cooling_cfl * (e_int - pfloor/(gamma-1)).
```

The applied `Delta e` is converted to the same Valencia conserved-variable
decrements as the ISM cooling source.  Thus the cooling is a fluid-frame
thermal-energy sink, not a direct subtraction from coordinate-frame total
energy alone.

## Magnetic Damping Inside Smooth Excision

Magnetic damping is added through the edge-centered EMF, so it remains
constraint-transport compatible:

```ini
<coord>
smooth_excision_b_damping = true
smooth_excision_b_damping_eta = 0.5
smooth_excision_b_damping_cfl = 0.25
```

This adds a resistive term of the form `E_damp = eta W curl(B)` at CT edges,
with edge weights derived from the smooth-excision cell weight.  The effective
`eta` is capped by `smooth_excision_b_damping_cfl * dx_min^2 / dt` when the CFL
cap is positive.

## Unresolved-Horizon Sink

Use the sink when the grid is too coarse to resolve a horizon but you still want
to evolve many orbits before a zoom restart:

```ini
<problem>
unresolved_sink = true
sink_radius = 0.0
sink_width = -1.0
sink_cells_per_radius = 10.0
sink_resolved_cells_across_horizon = 20.0
sink_timescale = 250.0
sink_density_floor = 1.0e-6
sink_pressure_floor = 1.0e-8
```

Per hole, the local resolution is measured from the MeshBlock containing that
hole.  A sink is active when:

```text
2*r_H/dx < sink_resolved_cells_across_horizon
```

The automatic sink radius is:

```text
R_sink = max(sink_cells_per_radius*dx, sink_radius, r_H)
```

With the defaults, the sink is at least 10 cells in radius, or 20 cells wide.
As AMR refines the hole, `dx` decreases and the sink radius shrinks.  Once the
horizon is resolved by the configured criterion, that hole's sink turns off.
The other hole can remain sink-drained if it is still under-resolved.

The sink damps conserved density, momentum, energy, and passive scalars.  It
does not directly modify the face-centered magnetic field.  Magnetic cleanup
inside the horizon should be handled by smooth-excision EMF damping.

## AMR Controls

The pgen installs `user_ref_func = Refine`, which combines an optional moving
criterion with optional fixed origin-centered radius levels.

### Lapse-Based Refinement

```ini
<problem>
amr_condition = alpha_min
alpha_thr = 0.8
```

Blocks refine when the minimum ADM lapse on the block falls below
`alpha_thr`, and derefine when it rises above `1.25*alpha_thr`.

### Tracker-Based Refinement

Legacy-compatible tracker setup:

```ini
<problem>
amr_condition = tracker
radius_thr = 5.0
```

`radius_thr` seeds the moving refinement radius around both holes.  By default,
the tracker reflevel is `-1`, meaning "keep refining inside the tracker sphere
until the global AMR maximum is reached."

Per-hole tracker setup:

```ini
<problem>
amr_condition = tracker
radius_thr = 5.0
tracker_1_rad = 4.0
tracker_2_rad = 8.0
tracker_1_reflevel = 5
tracker_2_reflevel = 6
```

Common-level shortcut:

```ini
<problem>
amr_condition = tracker
radius_thr = 6.0
tracker_reflevel = 5
```

Rules:

- `tracker_1_rad` and `tracker_2_rad` default to `radius_thr`.
- `tracker_1_reflevel` and `tracker_2_reflevel` default to `tracker_reflevel`.
- `tracker_reflevel` defaults to `-1`, preserving the old refine-to-max
  behavior.
- If a MeshBlock intersects both tracker regions, the higher requested level
  wins.
- If a block is above the requested level and only inside finite-level tracker
  regions, it can derefine.

### Fixed Origin-Centered Radii

These regions are centered on the coordinate origin and are applied in addition
to `alpha_min` or `tracker`:

```ini
<problem>
radius_0_rad = 256
radius_0_reflevel = 4
radius_1_rad = 128
radius_1_reflevel = 5
radius_2_rad = 64
radius_2_reflevel = 6
```

Up to 16 regions are parsed, starting at `radius_0_*` and stopping at the first
missing radius.

## Flux and Horizon Diagnostics

Enable user history and configure spherical surfaces:

```ini
<problem>
user_hist = true
flux_ntheta = 64
flux_nphi = 128
flux_rsurf_inner = 20
flux_rsurf_outer = 400
flux_dr_surf = 10

flux_horizon1 = true
flux_horizon2 = true
flux_radius1 = 2
flux_radius2 = 2

<output3>
file_type = hst
dt = 50
```

The horizon diagnostic surfaces labeled `H1` and `H2` are recentered on the
current trajectory in `TorusHistory`.

## Suggested Input Fragments

### 1. Conservative Production-Like Resolved Run

Use this when the horizons are resolved from the start:

```ini
<coord>
general_rel = true
excise = true
excision_scheme = puncture
smooth_excision = true
smooth_excision_width = 0.75
smooth_excision_inflow_speed = 0.40
smooth_excision_sigma_max = 1.0e5
smooth_excision_b_damping = true
smooth_excision_b_damping_eta = 0.5
smooth_excision_b_damping_cfl = 0.25
require_resolved_horizon = true
dexcise = 1.0e-10
pexcise = 1.0e-18

<problem>
sep = 25.0
q = 1.0
a1 = 0.0
a2 = 0.0
use_traj_table = false
unresolved_sink = false
thin_disk_cooling = false
amr_condition = tracker
radius_thr = 5.0
tracker_reflevel = -1
```

### 2. Tabulated Tilted-Spin Binary

Use this when the trajectory and spin direction come from an external driver:

```ini
<problem>
use_traj_table = true
traj_file = trajectories/tilted_spin.dat
adjust_mass1 = 1.0
adjust_mass2 = 1.0

amr_condition = tracker
tracker_1_rad = 6.0
tracker_2_rad = 6.0
tracker_1_reflevel = 5
tracker_2_reflevel = 5
```

The table spin columns should contain dimensionless `chi`.  For high-spin
tests, use `|chi| <= 0.95` unless you specifically want to test near-extremal
behavior.

### 3. Low-Resolution Many-Orbit Sink Stage

Use this to evolve cheaply before a zoom restart:

```ini
<mesh>
nx1 = 16
nx2 = 16
nx3 = 16

<meshblock>
nx1 = 4
nx2 = 4
nx3 = 4

<mesh_refinement>
refinement = none
num_levels = 1

<coord>
excise = true
excision_scheme = puncture
smooth_excision = true
require_resolved_horizon = false

<problem>
use_traj_table = true
traj_file = trajectories/eleven_orbits.dat
unresolved_sink = true
sink_radius = 0.0
sink_width = -1.0
sink_cells_per_radius = 10.0
sink_resolved_cells_across_horizon = 20.0
sink_timescale = 250.0
sink_density_floor = 1.0e-6
sink_pressure_floor = 1.0e-8
thin_disk_cooling = true
thin_cooling_h_over_r = 0.03
thin_cooling_timescale_orbits = 1.0

amr_condition = none
radius_0_rad = 24.0
radius_0_reflevel = 0
```

Excision remains enabled.  The sink is only an additional drain where the local
horizon resolution is insufficient.

### 4. Zoom Restart With Per-Hole Resolution

Use this after a low-resolution restart when one hole needs more refinement
than the other:

```ini
<mesh_refinement>
refinement = adaptive
num_levels = 4
refinement_interval = 1
ncycle_check = 1
prolong_primitives = false

<problem>
unresolved_sink = true
sink_radius = 0.0
sink_width = -1.0
sink_cells_per_radius = 10.0
sink_resolved_cells_across_horizon = 20.0

amr_condition = tracker
tracker_1_rad = 3.0
tracker_2_rad = 5.0
tracker_1_reflevel = 2
tracker_2_reflevel = 3

radius_0_rad = 12.0
radius_0_reflevel = 1
```

The sink radius follows the local resolution around each hole.  Once a hole
satisfies `2*r_H/dx >= 20`, that hole's sink turns off even if the other hole's
sink remains active.

### 5. Legacy Tracker Compatibility

This reproduces the old tracker behavior:

```ini
<problem>
amr_condition = tracker
radius_thr = 5.0
```

Explicit equivalent:

```ini
<problem>
amr_condition = tracker
radius_thr = 5.0
tracker_1_rad = 5.0
tracker_2_rad = 5.0
tracker_1_reflevel = -1
tracker_2_reflevel = -1
```

## Practical Checks Before Production

Run these checks on a reduced problem before a long run:

- Confirm the trajectory table covers the full time range.
- Confirm each table spin satisfies `|chi| <= 1`.
- For resolved excision, check `2*r_H/dx` near each hole.
- For unresolved sink stages, check that the sink turns off after refinement
  when the configured resolution criterion is met.
- Output `mhd_divb` and verify the maximum remains at roundoff or within the
  expected prolongation/restriction error for the mesh hierarchy.
- Track `sigma = b^2/rho` inside the sink or excision region when testing
  magnetic damping.
- Keep output cadence low for many-orbit low-resolution stages to avoid
  generating excessive local files.

## Known Limitations and Conventions

- The metric is a superposed BBH background, not a solved binary spacetime.
- No attenuation/window function is applied between the two Kerr-Schild holes.
- Table positions use Hermite interpolation, while table masses and spins use
  linear interpolation.
- The unresolved sink is a coarse-grid fallback, not a replacement for a
  resolved horizon.
- Magnetic damping modifies the CT EMF, not face-centered magnetic fields
  directly.
- `amr_condition = tracker` follows the trajectory points, not an apparent
  horizon finder.
