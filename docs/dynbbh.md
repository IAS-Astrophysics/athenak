# `dynbbh` Problem Generator

`src/pgen/dynbbh.cpp` initializes and evolves a circumbinary disk on a
superposed boosted Kerr-Schild binary-black-hole background.  The default path
keeps the historical behavior: two black holes follow a fixed circular
Keplerian orbit and the metric is built from the two superposed boosted
Kerr-Schild perturbations.  Optional paths allow a tabulated binary trajectory,
smooth horizon excision, magnetic-field damping through the constrained
transport EMF, cooling source terms, and user-controlled AMR around each hole.

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

For MHD runs, also provide an `<mhd>` block.  Smooth-excision magnetic damping
and the cooling source terms require MHD.

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
spin_ramp = false
spin_ramp_timescale = 50.0
# spin_ramp_start_time defaults to the segment's starting mesh time if omitted
use_traj_table = false
```

Here `sep` is the binary separation, `q` sets the mass ratio through
`M1=1/(q+1)` and `M2=1-M1`, and the orbital frequency is `sep^(-3/2)`.
The spin angles are in degrees.  This is the compatibility mode and should
remain unchanged for older input files.  For analytic-orbit spin-up tests, set
`spin_ramp = true`; then `a1` and `a2` are interpreted as the final
dimensionless spin magnitudes and are multiplied by
`S(u) = u^2 (3 - 2u)`, where
`u = clamp((t - spin_ramp_start_time)/spin_ramp_timescale, 0, 1)`.
If `spin_ramp_start_time` is omitted, the ramp begins at the mesh time when the
segment starts.  The trajectory-table path does not use these ramp parameters;
encode time-dependent spin directly in the table instead.

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

The physical spin parameter used by Kerr-Schild is `a_i = chi_i * M_i`, where
`chi_i` is the dimensionless spin vector and `M_i` is the trajectory mass.

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
disk runs.  For stress tests of accretion and excision behavior, smaller values
can be useful, but they should be treated as algorithm tests rather than
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
smooth_excision_sigma_max = 1.0e5
smooth_excision_puncture_weight_exponent = 1.0
smooth_excision_puncture_width_fraction = 1.0
puncture_flux_excision_radius_factor = 1.0
smooth_excision_temp_ceil = -1.0
smooth_excision_inflow = false
smooth_excision_inflow_speed = 0.0
dexcise = 1.0e-10
texcise = 1.0e-8
require_resolved_horizon = false
```

The pgen updates `punc_0`, `punc_1`, spins, velocities, and horizon radii from
the current trajectory.  Excision remains active whenever `coord/excise=true`.
By default, explicit `excise_1_rad` and `excise_2_rad` values are used when they
are positive; otherwise the local Kerr-Schild horizon radius is used.  Set
`excise_to_horizon = true` to ignore explicit radii and use the horizon radius
for both holes.  Set `excise_shrink_to_horizon = true` to smoothly reduce the
current explicit radius to the horizon radius over
`excise_shrink_timescale` (default `50 M`) after each run/restart begins.  The
puncture smooth-excision weight always uses a transition width equal to
`smooth_excision_puncture_width_fraction * punc_rad`, so the smooth layer
follows the time-dependent radius.  `smooth_excision_puncture_weight_exponent`
selects the radial weight profile
`s^(2n) * ((2n + 1) - 2n*s)`, with
`s = clamp((punc_rad - r)/width, 0, 1)`.  The default `n=1` preserves the
traditional `s^2(3-2s)` profile; `n=2` matches the previous `slow_start`
profile, and `n=3` matches the previous `smoother_start` profile.
The default width fraction is `1.0`; with this default, the puncture smooth
weight tapers across the full radius.  A value of `0.5` recovers the older
behavior where the inner half of the radius is fully weighted and the outer half
is the transition.  The same updated `punc_rad` values feed flux excision,
smooth primitive damping, and the smooth magnetic damping weights.
Set `puncture_flux_excision_radius_factor > 1` to make the first-order flux
region larger than the puncture smooth-excision region without changing the
primitive damping or magnetic damping support.  For example, `1.2` extends the
FOFC mask to `1.2 * punc_rad`.
If `smooth_excision_temp_ceil > 0`, the post-blend thermal state is also
strictly capped inside cells with nonzero smooth-excision weight.  In the
Valencia/dyn GRMHD path this caps pressure through the EOS.  This is a hard
local ceiling for keeping the excision region cold; it is not applied where the
smooth puncture weight is zero.
`texcise` specifies the smooth-excision target temperature directly; when
`texcise > 0`, the code derives the excision pressure target from the EOS and
`dexcise`.  If `texcise < 0`, the legacy `pexcise` pressure target is used.
If `smooth_excision_inflow=true`, the smooth-excision velocity target is
modified so the puncture-frame radial 3-velocity is at least inward by
`smooth_excision_inflow_speed * weight`.  Gas that is already flowing inward
faster than this target is not slowed down.  The Valencia/dyn GRMHD path
computes this guard using the Lorentz factor to convert between the stored
spatial 4-velocity components and coordinate 3-velocity, then applies the
existing velocity ceiling before converting back.

For the smooth-excision equations and constrained-transport discussion, see
`docs/smooth_excision_procedure.tex` and the compiled PDF.

## Staged Zoom Workflow

`scripts/setup_dynbbh_stage.py` has two modes.  The default `single` mode keeps
the original one-case helper behavior.  The `zoom-survey` mode creates a
restart-to-restart workflow for a SANE/MAD/BONDI bundle:

```bash
python scripts/setup_dynbbh_stage.py \
  --workflow zoom-survey \
  --base-dir /home/hzhu/scratch3/<fresh-workdir> \
  --exe /home/hzhu/athenak/build_cb/src/athena \
  --case-source SANE=/path/to/SANE_stage1.par,/path/to/SANE/rst/rank_00000000/torus.NNNNN.rst \
  --case-source MAD=/path/to/MAD_stage1.par,/path/to/MAD/rst/rank_00000000/torus.NNNNN.rst \
  --case-source BONDI=/path/to/BONDI_stage1.par,/path/to/BONDI/rst/rank_00000000/torus.NNNNN.rst
```

The setup verifies that each single-file-per-rank restart is complete for the
requested rank count, copies all rank files into the stage-2 run directories,
and reads the actual mesh time from the binary restart header.  Stage 2 keeps
both holes at zero spin, starts from the current smooth-excision radius of
`4M`, and sets `coord/excise_shrink_to_horizon=true` for one orbit.  The
generated debug-scaling PBS bundles SANE, MAD, and BONDI with 22 nodes each
and `-t 00:50:00`.
By default, generated zoom inputs use `dt=1000` for single-file-per-rank
restarts, `dt=50` for angular-momentum output, and `dt=25` for each
`slice_x1`, `slice_x2`, and `slice_x3` output.

After the horizon-zero-spin runs have produced clean restarts, the generated
stage-3 PBS runs the spin survey: `chi=-0.7`, `0`, `+0.7`, and a tilted
`chi=0.7` case for each SANE/MAD/BONDI case.  In the tilted case, BH1 is
30 degrees above the orbital plane (`th_a1=60`) and BH2 is aligned with the
orbital angular momentum (`th_a2=0`).  The nonzero-spin cases use
`problem/spin_ramp=true`; their launch scripts override
`problem/spin_ramp_start_time` from the actual restart they select, avoiding
stale timestamps from restart headers.  The stage-3 inputs default to forty
post-horizon orbits so short integration tests are wall-clock limited rather
than physics-time limited, and the PBS is sized for 12 concurrent 22-node runs
on the prod queue with two 50-minute application segments inside one two-hour
allocation.

## Cooling Source Terms

`dynbbh` can apply one of two optically thin/source-term cooling models.  Select
the model explicitly:

```ini
<problem>
cooling_source = none       # none | ism | thin_disk
```

Both cooling models act as isotropic fluid-frame energy losses and are converted
to Valencia conserved-variable source terms.  `cooling_source` is single-valued,
so ISM cooling and thin-disk cooling cannot be combined.

### ISM Cooling

Enable ISM cooling with:

```ini
<problem>
cooling_source = ism
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
cooling_source = thin_disk
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

This adds a resistive term of the form `E_damp = eta W curl(B)` at CT edges.
The edge weight is the minimum of the adjacent smooth-excision cell weights, so
the damping EMF is nonzero only on edges strictly inside the smooth puncture
region.  The effective `eta` is capped by
`smooth_excision_b_damping_cfl * dx_min^2 / dt` when the CFL cap is positive.
The user EMF is applied before the normal EMF exchange/restriction step, so
same-level and coarse/fine boundaries use the synchronized CT EMF before the
magnetic update.

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
cooling_source = none
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

amr_condition = tracker
tracker_1_rad = 6.0
tracker_2_rad = 6.0
tracker_1_reflevel = 5
tracker_2_reflevel = 5
```

The table spin columns should contain dimensionless `chi`.  For high-spin
tests, use `|chi| <= 0.95` unless you specifically want to test near-extremal
behavior.

### 3. Coarse Many-Orbit Trajectory Stage

Use this for a cheap trajectory/excision algorithm test.  Production runs should
keep horizons resolved or use it only with `require_resolved_horizon=false`
after checking the resolution warnings.

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
cooling_source = thin_disk
thin_cooling_h_over_r = 0.03
thin_cooling_timescale_orbits = 1.0

amr_condition = none
radius_0_rad = 24.0
radius_0_reflevel = 0
```

Excision remains enabled.  The pgen prints a warning if the local excision
radius has fewer than 10 cells across its diameter.

### 4. Staged Schwarzschild-to-Spin Workflow

For the analytic circular-orbit branch, the most conservative high-spin setup is
usually to change one hard thing at a time:

1. **Schwarzschild burn-in**: keep `a1 = a2 = 0`, `spin_ramp = false`, and use a
   generous explicit puncture radius while the disk and sink settle.
2. **Refine and shrink excision**: restart from the burn-in and either shrink the
   explicit puncture radius to the local Kerr-Schild horizon with
   `excise_shrink_to_horizon = true`, or switch directly to
   `excise_to_horizon = true` once the horizon-resolved run is stable.
3. **Spin ramp**: restart from a stable Schwarzschild/horizon-sized segment,
   set the desired final `a1` and `a2`, and enable `spin_ramp = true` over a
   long timescale such as one orbit.
4. **Static-spin continuation**: restart from the end of the ramp with
   `a1 = a2 = a_target` and `spin_ramp = false` to test long-term stability
   without further metric forcing.

The helper script `scripts/setup_dynbbh_stage.py` automates the parfile and PBS
edits for these restart-to-restart stages.  Example sequence:

```bash
# 1. Start from a known restart/template and keep the analytic holes nonspinning.
python scripts/setup_dynbbh_stage.py \
  --base-dir /path/to/runs \
  --case bbh_schwarzschild_burnin \
  --stage burnin-schwarzschild \
  --template-parfile /path/to/template.par \
  --restart /path/to/original.rst \
  --exe /home/hzhu/athenak/build_cb/src/athena \
  --submit

# 2. Continue by shrinking the explicit excision region to the horizon.
python scripts/setup_dynbbh_stage.py \
  --base-dir /path/to/runs \
  --case bbh_shrink_horizon \
  --stage shrink-to-horizon \
  --source-run /path/to/runs/bbh_schwarzschild_burnin/run_0 \
  --shrink-timescale 785.0 \
  --submit

# 3. Continue from the stable horizon-sized Schwarzschild run and ramp to spin.
python scripts/setup_dynbbh_stage.py \
  --base-dir /path/to/runs \
  --case bbh_spin09_ramp \
  --stage spin-ramp \
  --source-run /path/to/runs/bbh_shrink_horizon/run_0 \
  --spin-target 0.9 \
  --spin-ramp-timescale 785.0 \
  --submit

# 4. Continue at fixed high spin.
python scripts/setup_dynbbh_stage.py \
  --base-dir /path/to/runs \
  --case bbh_spin09_static \
  --stage spin-static \
  --source-run /path/to/runs/bbh_spin09_ramp/run_0 \
  --spin-target 0.9 \
  --submit
```

The script chooses the latest `rst/rank_00000000/torus.*.rst` from
`--source-run`, copies the executable into the new run directory, writes a fresh
`launch.sh`, and generates a PBS script next to the run cases.  Use
`--set block/key=value` for local overrides that are not covered by a stage, for
example `--set coord/smooth_excision_b_damping_eta=0.01`.

### 5. Zoom Restart With Per-Hole Resolution

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
amr_condition = tracker
tracker_1_rad = 3.0
tracker_2_rad = 5.0
tracker_1_reflevel = 2
tracker_2_reflevel = 3

radius_0_rad = 12.0
radius_0_reflevel = 1
```

### 6. Legacy Tracker Compatibility

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
- Output `mhd_divb` and verify the maximum remains at roundoff or within the
  expected prolongation/restriction error for the mesh hierarchy.
- Track `sigma = b^2/rho` inside the excision region when testing magnetic
  damping.
- Keep output cadence low for many-orbit low-resolution stages to avoid
  generating excessive local files.

## Known Limitations and Conventions

- The metric is a superposed BBH background, not a solved binary spacetime.
- No attenuation/window function is applied between the two Kerr-Schild holes.
- Table positions use Hermite interpolation, while table masses and spins use
  linear interpolation.
- The analytic-orbit spin ramp changes only the prescribed background spin; it
  is not a self-consistent binary spin evolution.
- Magnetic damping modifies the CT EMF, not face-centered magnetic fields
  directly.
- `amr_condition = tracker` follows the trajectory points, not an apparent
  horizon finder.
