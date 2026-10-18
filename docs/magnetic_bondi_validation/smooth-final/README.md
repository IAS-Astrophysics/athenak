# Completed two-cycle smooth-excision MHD comparison

Both requested new roundtrips completed. Final job **8940829** exited normally
at native t=29500 M, cycle 346970, finest zone 0/event 42, with 1408 active
blocks. The accepted controller logged TARGET_REACHED and did not submit another
continuation. All 13 physical jobs passed all twelve rank resource/memory reports,
positive finite saved density/pressure and scaled magnetic-divergence checks.
Maximum conservative RSS plus tracked allocation peaks: **23.474 GB**; total
guard budgets: 28.4 GB. Maximum scaled divergence across the chain: 1.68e-11.
The chain consumed approximately 11.20 one-node hours of process runtime,
including restart overhead, excluding queue waiting and the separate switch test.
The selected postprocessing run used 0.546 GB peak RSS under an 8 GB guard.

## Actual finest visits

| Visit | Event | Entry time / M | Available saved phase range / M |
| --- | --- | --- | --- |
| Smooth restart of evolved hard-excision data | 14 | 9903.615850407483 | 96.38–248.84 |
| First new return | 28 | 19586.65420246086 | 63.35–232.65 |
| Second new return | 42 | 29269.69255450416 | 30.31–230.31 |

The initial fresh finest dwell and first-return dwell were each 250 M; the final
visit evolved 230.307 M before the authorized target. Thus the finest stage was
active for 730.307 M, about 0.642 binary orbital periods (P=1137.378 M), while
the global clock advanced 19596.384 M. These are different exposures; freezing
precludes interpreting global elapsed time as horizon evolution.

Output counters were preserved through clean wall cutoffs. This leaves gaps in
the saved series. Phase 50 M is not available in every visit, so comparisons use
**100,150,200 M** with bracketed linear temporal interpolation, never extrapolation.
These three-phase means are not whole-visit time averages or statistical ensemble
averages. The first visit is a physics-switch baseline, not the original uniform
seed and not an already equilibrated smooth solution.

## Accretion and reservoir assessment

At coordinate extraction radius 2 M around the fixed corotating hole centers:

| Visit | Hole at x=-16 M | Hole at x=+16 M |
| --- | ---: | ---: |
| Smooth restart | 725.03 | 658.59 |
| Return 1 | 1094.25 | 583.97 |
| Return 2 | 991.33 | 875.35 |

Rates are rest mass per coordinate time in raw code units, positive inward.
From the first to second return, these means change **-9.41%** and **+49.90%**.
Across extraction radii 1.5,2,3 M the corresponding changes are -9.55 to -8.66%
and +48.94 to +50.25%, so the qualitative conclusion is not confined to r=2 M.
The 24x48 versus 48x96 angular sensitivities at r=2 are below 0.46% for these
means; nearest-cell versus bounded-trilinear differences are below 0.14%.
These are diagnostic sensitivities, not rigorous spatial error estimates.

Fixed-shell mass changes from Return 1 to Return 2 are **+19.74%, +8.02%,
+13.87%, -13.02%, -5.35%** for shell edges [24,32,64,128,256,500] M.
Topology and spatial cell assignments are identical across these comparisons.
Shell assignment is nevertheless coarse, and no continuous mass-conservation
residual is inferred across frozen intervals.

COM net flux is strongly nonuniform with radius and changes sign in some shells.
At r=500 M, the matched-phase mean is +8527.6 at the switch baseline, -1424.6
at Return 1 and -6701.4 at Return 2: the final sampled mean is **net outward**.
The accretion plot uses absolute differences in net inward rate, avoiding
misleading percent changes when a signed flux crosses zero. The raw signed
percentage calculations and sign-change flags remain in the JSON for audit.
COM angular sensitivities reach 3.81% and the nearest-cell comparison 6.80% of
net flux, much less than several observed inter-visit changes, but neither
bounds interpolation, spatial resolution, temporal sampling, or cyclic-zoom
conservation errors.

**Cyclic repeatability and inflow equilibrium out to the Bondi radius are not
established.** Two new cycles show substantial changes, despite small angular
sensitivities. These samples alone cannot separate a secular transient from
long-time turbulent fluctuations. Higher-resolution/conservative face-flux and
source accounting would be needed for a stronger conclusion; additional runs
are not automatically authorized.

## Sampling correction and remaining limitations

The old AMRSampler extrapolated over half-cell edges of a block without saved
ghost cells. At t=19700 M, one r=1.5 M surface point around x=-16 M produced
rho=-0.2227 although every actual cell had positive density (minimum 1.65e-6).
This is a diagnostic extrapolation artifact, not a negative-density simulation.

The new smooth-chain analysis injects a bounded trilinear sampler into the
shared metric/flux functions. All interpolation weights are convex; half-cell
edges clamp to the available center hull. A nearest-cell calculation separately
measures sensitivity to this choice. No fluid values are edited or floored in
postprocessing. The original sampler remains the default for historical scripts,
so old hydro/ApJ results are not silently rewritten. A synthetic positive jump
regression reproduces the old negative extrapolation, checks positive bounded
sampling, preserves constants, and verifies affine interpolation in the interior.
This correction does not supply missing ghost data or establish a conservative
surface-flux diagnostic.

Flux is -integral rho*sqrt(gamma)*(alpha*utilde^i-W*beta^i)*n_i dA_coordinate;
shell mass is integral rho*W*sqrt(gamma) dV_coordinate. Fluid-frame plasma beta
uses B=Btilde/sqrt(gamma) and b^2=[B_i B^i+(B_i utilde^i)^2]/W^2.
The stationary initial ADM metric is reused only after exact finest topology
and block-geometry checks. The user smooth sink remains per primitive-recovery
call; large near-drain pressure frame differences and timestep sensitivity from
the short controls remain a limitation. No new C++ evolution changes were made.

## Artifacts and provenance

- `accretion.png/pdf`: hole fluxes by visit phase, radial net flux, successive differences.
- `shell_mass.png/pdf`: identical-topology shell masses and inter-visit changes.
- `finest_xy.png/json`: final t=29500 actual finest-stage xy density, p/(rho c^2), physical beta.
- `smooth_zoom_xy.mp4`: 52 selected actual saved states, t=10000–29500 M; no temporal interpolation.
- `movie/frames.json`, `selection.json`, `metric-validation.json`: exact sources, stages and metric checks.
- `inflow_equilibrium.json`: all 16 finest samples, radii, phase coverage, sensitivities and matched means.
- `provenance`: every job's validators, all twelve resource/memory reports, rank-0 logs,
  build/runtime hashes, final PBS record and accepted continuation receipts.

The movie starts at the first saved smooth state, 96.38 M after the restart,
not at an invented initial frame. Zoom stage labels come from the saved AMR
topology rather than rounded event timestamps. Coarse-stage frames show actual
active coarse data; frozen finer states are not displayed. Black masks mark
the smooth puncture-drain regions. The temperature panel is a dimensionless
proxy, not kelvin. The historical ApJ report is preserved as an earlier status.

## Return labels and active-block movie

All three plotted visits belong to the **post-smooth-switch** evolution:
"Smooth restart" is the fresh finest dwell starting at 9903.61585 M, "Return 1"
is the first full post-switch roundtrip ending at 19586.65420 M, and "Return 2"
is the second ending at 29269.69255 M. No pre-switch hard-excision returns enter
these matched comparisons.

`smooth_zoom_xy_meshblocks.mp4` adds active leaf-meshblock outlines on all three
xy panels. Each block contains 8x8x8 cells. Edges come directly from the saved
active block geometry intersecting the plotted slice, with duplicate shared
edges removed and clipping at the view limits. The same 52 files, times, stages,
color scales and playback cadence are retained. `movie-meshblocks/frames.json`
records overlay flags, visible-block counts and edge counts; the final frame is
also saved as `finest_xy_meshblocks.png`. This overlay visualizes the grid
actually present at each stage, including the intentionally coarse phases.
