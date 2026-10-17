# Equatorial movie from the initial magnetic reservoir

magnetic_bondi_xy.mp4 shows79 real saved physical states from t=0 through
9903.62 M, at5 fps with a2-second final hold (17.8 seconds total). Panels show
rho/rho_inf, temperature proxy p/(rho c^2), and physical fluid-frame beta=2p/b^2.
The source sequence includes completed validated jobs8937401,8937572,8937793,
8938001,8938136,8938200,8938971. It excludes accelerated smoke tests and the
ongoing continuation. Actual saved times and source paths appear in frames.json.

This is an evolution from the uniform seed, not evidence of completed relaxation.
It includes the first complete outward/inward sweep and its finest return. The
finest grid has accumulated about470.58 M exposure at the end, much less than
the global clock. Coarse frames show the actual active AMR fields; they do not
replace them with frozen fine data. The positive-z cell slice touching z=0 is
used, at its actual cell-center z, so a coarse slice has finite thickness.

## Plasma beta on changing meshes

magnetic_pilot_metric.py specializes the simulation's superposed boosted
Schwarzschild KS spatial metric for q=1, spins0, a=32, corotating at t=0.
It evaluates this stationary metric at every active slice cell center, including
coarsened cells. No fine-metric interpolation is used. The specialization
rejects incompatible binary configurations. Against every cell of the saved
initial ADM, all6 spatial components agree to5.96e-8 when normalized by
max(1,abs(saved)), consistent with the saved single-precision metric.

The stored field is converted via B_Eulerian=Btilde/sqrt(det gamma). With stored
utilde=Wv, W^2=1+gamma_ij utilde^i utilde^j and
b^2=[gamma_ij B^i B^j+(gamma_ij B^i utilde^j)^2]/W^2. The resulting beta at the
latest finest snapshot agrees with direct saved-ADM conversion within1.27e-7
relative outside puncture excision; the largest7.81e-7 relative difference is
inside the masked puncture core, where conditioning amplifies metric rounding.
Exact metrics and crosscheck results are retained in the accompanying JSON.

## Excision actually used

The saved fluid header explicitly records coord/excise=1,
excision_scheme=puncture, excise_1_rad=excise_2_rad=0.8,
dexcise=1e-8, pexcise=1e-12, smooth_excision=0 and
smooth_excision_b_damping=0. See excision-from-saved-header.json.
Thus this chain uses hard primitive resets, not the available smooth drain.
The active dynamic-MHD primitive solver resets masked cell-center density,
pressure and stored primitive velocities, then reconstructs conserved variables
using the current magnetic field. The field itself remains CT-evolved; optional
interior magnetic damping is disabled.

The smooth implementation is present at src/eos/primitive_solver_hyd.hpp near
line601 and the weight is constructed in src/coordinates/excision.cpp.
coord/smooth_excision defaults false in coordinates.cpp. The pilot input left
that flag disabled. The smooth path blends toward target density/pressure and
velocity using a puncture-distance weight, with optional caps/inflow controls.
No physical option was silently changed for visualization or on continuation.
A smooth-excision comparison needs its own tested run and provenance.

Black plot masks show the nominal boosted puncture excision regions, not an
apparent-horizon measurement. At coarse stages the cell-center criterion may
select no active cells inside a subcell-size0.8 M mask; the finer stored mesh
has different evolution exposure. No plotted quantity inside excision is given
physical interpretation.

## Reproduction and checks

Use scripts/make_magnetic_xy_movie.py --runs [chronological completed runs]
--metric [initial saved ADM] --out [directory] --frame-dt100. It validates
finite positive fluid, the pilot configuration, saved excision flags and metric
agreement before plotting. Raw rendered frames remain under ignored
build-mac-zoom-opt/aurora-magnetic/xy-movie/frames. Encode at5fps using H.264,
yuv420p and faststart with a2-second final hold. The full89-frame encoded movie
passed a decode check; first, coarse and final images were visually inspected.
Rendering used one analysis thread, a4 GB guard and238 MB peak RSS on Aurora;
no fluid evolution or Athena implementation change was needed for the movie.
