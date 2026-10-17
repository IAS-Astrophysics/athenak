# First physical magnetic zoom-cycle movie and outer-boundary assessment

The movie contains78 strictly time-ordered physical snapshots from t=0 to
9782.47 M, across jobs8937401,8937572,8937793,8938001,8938136,8938200.
Only completed, validated jobs are used; accelerated smoke tests and the
ongoing8938971 continuation are excluded. Source paths and exact times are in
frames.json. Five frames/second gives a17.6-second H.264 movie including a
2-second final hold. The complete encoded file passed a decode check.

Top panels show density and p/rho near the binary in the x-z plane. Bottom
panels show the complete domain with fixed +/-8% color scales; saturated
central values are intentional. The dashed circle marks rB=500 M. Actual active
AMR states are shown, with no frozen finer data substituted. Mesh changes and
visible coarsening are part of cyclic zoom. Equal playback time between frames
does not imply exactly equal simulation-time spacing. A single physical
roundtrip and349 M finest exposure do not establish relaxed accretion.

## Outer-boundary findings

The six box faces are at +/-2048 M (4.096 rB). At the last snapshot, adjacent
active cells show at most1.61% density departure,2.66% pressure departure,
1.07% departure of stored vertical densitized field Btilde_z/Binf, and0.93%
transverse Btilde/Binf. The largest absolute face-mean departures are0.75%
in density,1.24% in pressure, and0.234% in Btilde_z. The maximum adjacent
inertial coordinate speed is0.00153 c; face-mean normal velocities are inward.
Initial inertial velocity recovery is within9.6e-7 c after subtracting frame
rotation, consistent with saved single-precision primitive/metric data.

There is a mild boundary-associated depletion/adjustment layer, strongest at
the two z faces parallel to the imposed field. The time histories are smooth,
with close agreement of opposite faces, and no large oscillatory or growing
boundary instability is apparent through9782 M. Polar density/pressure
depletion is strongest in the first active layer and weakens several cells
inward; this should not be called an artifact-free boundary. Physical magnetic
anisotropy and reservoir adjustment cannot be separated cleanly from numerical
boundary influence by one box-size run. The x/y faces also differ slightly.

The fixed exterior reservoir has rho=1, p=pinf, inertial coordinate rest and
prescribed exterior magnetic data; physical normal magnetic faces remain
CT-evolved. It does not impose an accretion rate. Its subsonic MHD waves can
reflect, and active-cell inflow differs from the imposed zero exterior velocity.
These plots do not demonstrate absence of reflection or its effect on the
inner flow. A larger-box/matched-resolution comparison and characteristic
boundary or outgoing-wave control are needed before claiming boundary
independence. No physical boundary prescription was changed for this movie.

## Diagnostic definition and limitations

boundary_profiles.png compares face-averaged density, pressure and outward
inertial coordinate normal velocity over time and across six inward layers.
boundary_magnetic.png shows stored densitized magnetic components and recovered
inertial coordinate motion. These are active-cell values at64,192,...704 M
inside the faces, with128 M cell width. Ghost cells and physical face fluxes
are not present in the physical outputs. Corners are counted on each adjoining
face. No continuous mass budget or nonreflection proof can be inferred.
The stationary metric is reused only after exact outer-block geometry matching.
Full fluid/metric conversion is vcoord=alpha*utilde/W-beta followed by removal
of coordinate rotation. Magnetic departures use stored Btilde, not fluid-frame
beta or invariant magnetic flux. The initial2% pressure noise also contributes
to cell-wise pressure departures; face averages reduce that contribution.

boundary.json retains all sampled profiles including extrema; the smaller
boundary_summary.json contains the reported values. All movie frames were
checked finite with positive density and pressure. Rendering used one analysis
thread and a4 GB guard on Aurora, peaking at326 MB RSS; fluid evolution remains
in PBS. The final resource-file download was interrupted by expired SSH
authentication, after all frames and diagnostics had already been downloaded.
The render success and resource values were returned by the bounded tool.

## Reproduction

Run scripts/make_magnetic_bondi_movie.py --runs [six chronological runs]
--metric [initial ADM] --out [directory] --frame-dt100, then
scripts/plot_magnetic_boundary.py [directory]/boundary.json. Encode frames/%04d.png
at5 fps with H.264/yuv420p, a2-second final hold and faststart. Raw frames remain
in ignored build-mac-zoom-opt/aurora-magnetic/movie-first-return/frames locally;
the deliverable movie and source manifest are committed here.
