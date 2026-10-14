# Individual-hole accretion convergence study

Continuation job 8925878 was submitted from the final checkpoint of 8925561,
with target t=70000 M and fluid output every 10 M. This should sample about five
states per 50 M finest-zone visit during three additional cycles. The physical
model and validated executable are unchanged. Each allocation requests one
debug node for an hour, twelve GPU tiles, the 55-minute solver stop and existing
memory guards. Later clean continuations preserve the checkpoint output clocks.
The thread heartbeat follows receipts until this bounded target is reached.

The initial plots were a **baseline**, not a convergence result: the previous
100 M output cadence saved only the initial state and t=11500 M during zone zero.
The latter is the most recent saved fully refined state available at submission.
Its image is `../gradual_evolution_movie/finest_return.png`. Later fully refined
visits occurred but were missed by the old regular output cadence.

`scripts/bondi_hole_accretion.py` extracts rest-mass flux through coordinate
spheres of radii 1.5, 2 and 3 M centered at (-16,0,0) and (16,0,0). These spheres
lie outside the 0.8 M excision radii. They are near-hole accretion diagnostics,
not apparent-horizon integrals. Positive flux means inflow. With Athena's
normal-frame spatial four-velocity utilde, the integrand is

```
mdot = -r^2 integral sqrt(det gamma) rho (alpha utilde^i - W beta^i) n_i dOmega
W = sqrt(1 + gamma_ij utilde^i utilde^j)
```

Here n is the Euclidean coordinate sphere normal. The lapse, shift and spatial
metric come from the initial finest ADM output, reusable because the circular
corotating metric and extraction spheres are stationary. The shift includes the
rotating-coordinate contribution; no extra inertial radial velocity is added.
The metric and fluid are sampled independently on their respective AMR grids.

Rates are normalized to the analytic Michel/Bondi rate for **total mass one**.
At t=11500 M the r=2 M rates are 0.3067981 and 0.3067626 of that total rate.
Across radii 1.5–3 M the respective rates span 0.3058–0.3097, about 1.3%.
Doubling angular sampling from 24x48 to 48x96 changes the rates by at most
0.000279 of the total Bondi rate. This estimates quadrature sensitivity, not
spatial interpolation error or grid convergence. A synthetic radial flow with
nonzero lapse and shift reproduces its analytic integrated flux to 1e-12
relative tolerance. The two holes' close agreement is also a useful symmetry
check, but does not establish temporal convergence.

The script only samples finest-zone visits and does not infer flux across
frozen intervals. New visits will be compared at common elapsed times 10, 20,
30 and 40 M after entry, using interpolation only within each sampled visit.
Visits lacking that phase coverage are excluded from the matched-phase means.
The plot then shows changes between the available phase-matched means; holes
and extraction radii remain separate.

This is an unmagnetized hydrodynamic run using the MHD solver: b_inf=0, and all
three saved magnetic components are exactly zero in the checked finest states.

Run the script with chronological `--runs` directories, the initial ADM file
as `--metric`, and this directory as `--out`. `--list-only` lists the finest
snapshots without loading their fluid arrays, to permit selective transfer
from Aurora. Preserve each run's `rank-0.log` for visit identification.

## Completed six-cycle comparison

Jobs 8925878 and 8925972 stopped cleanly at 54189.3 and 69880.4 M. Job 8926068
finished at 70000 M, cycle 93958, with PBS exit status zero. All five jobs in the
chain passed twelve-rank resource checks; conservative combined RSS/allocation
peak bounds are below 20 GB (resources.json). Final fields are finite, density
and pressure positive, and magnetic components zero. No further run is scheduled.

Seventeen finest-stage snapshots are available: initial state, one from visit 1,
and five each from visits 4, 5 and 6. Visits 2 and 3 were missed by the previous
cadence. The latest finest state is t=68870 M, cycle 91135, with 848 MeshBlocks
and near-hole dx=0.125 M (latest_finest.png). The final global state is coarser.

Matched-phase means at r=2 M for the negative-x hole are 0.310954, 0.306701 and
0.305445 of the total analytic Bondi rate; positive-x values are 0.310995,
0.306789 and 0.305425. Successive changes decrease from about -1.36% to -0.43%.
Across all extraction radii the last change is -0.32% to -0.48%. Final mean
extraction-radius spread is under 0.1%, and inter-hole differences under 0.03%.
Maximum angular-quadrature sensitivity is 0.000373 of the total Bondi rate,
about 0.12% of an individual-hole rate. This does not bound interpolation error.

These results support approach to a repeatable cyclic inner inflow, but do not
prove continuously evolved stationarity, exact horizon flux or spatial
convergence. Six returns plus the initial dwell give about 350 M of zone-zero
exposure, despite the global clock of about 61.55 binary orbits. This remains
unmagnetized hydro using the MHD solver. The completion timer is paused.
