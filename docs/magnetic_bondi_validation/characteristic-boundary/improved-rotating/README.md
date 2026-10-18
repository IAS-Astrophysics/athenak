# Improved characteristic boundary: rotating tests completed

**Completed:** capacity job 8942061 passed all 30 cases/360 rank audits.
[Results, acoustic comparisons and runtimes](aurora/README.md) recommend retaining
the fixed production default. The quadratic trace helps planar acoustic pressure
errors, but rotating 3D errors and cost remain worse.
[Explicit seven-wave decomposition](FULL_DECOMPOSITION.md) is implemented as an
offline diagnostic and agrees with the native full projector; no production
eigensolver replacement is implied.

Original user authorization: improve the acoustic treatment and cost, then test in the
rotating frame before deciding on production use. The completed physical smooth
cycle chain remains stopped, and the production default remains fixed.

The previous method lowered the normal interior trace to first order and rebuilt
the numerical Jacobian/sign projector twice per stage. These are plausible causes
of the acoustic error and measured cost, not yet a completed error attribution.
The new implementation:

- Extrapolates perturbations relative to the local inertial-rest state to the face
  with a limited quadratic extrapolation of cell averages (order 3); orders 1/2
  remain explicit ablations. The unlimited face value is
  (11q0-7q1+2q2)/6. Constant, linear, quadratic and extremum checks passed. No prescribed background metric gradient is extrapolated as a wave.
- Computes the closure before the FOFC predictor. Only faces actually overwritten
  by FOFC/excision are recomputed, at first order, before flags are reset. A forced
  flag test makes first/second-order runs bitwise identical in saved fields.
- Uses the native conservative fast-wave-speed bounds for all-incoming/outgoing
  regions before constructing a numerical Jacobian. This includes lapse, shift,
  magnetic field, enthalpy and spatial metric; fluid normal velocity alone is not
  used as a diode. The 900 independent native analytic-mode checks still pass.
- Retains normal CT B and the paired fluid/induction flux path. Incoming modes
  still refer to the inertial-rest exterior reservoir. This is not an exact
  transparent exterior or a cure for the coordinate CFL restriction.

The suite uses compact finite-support perturbations with physical user faces in
all three directions. Acoustic packets at production Omega=32^(-3/2) have a
larger-domain control at the same cell size. A pressure-constant density bump is
exactly stationary in the inertial frame, so its rotating-coordinate advection
provides an independent analytic check. The high-rotation box has maximum
coordinate corner speed 16c; it is stopped at t=0.4 M before the perturbation can reenter,
when an ambient exterior would no longer describe its history. Diagnostics use
Eulerian velocity utilde/W and transform vector components back to the inertial
basis, rather than interpreting the large coordinate shift as physical speed.

Only isolated source-characteristic/build-characteristic is modified. Previous
executable/source are archived remotely in characteristic-v1-8941903. New source
hashes, local gates, capacity receipt and results will be recorded here. One
capacity node/all 12 GPU tiles, existing 28.4GB guards, PBS one hour and solver
55-minute cutoff remain mandatory. No duplicate accepted or unknown receipts,
no blind retries and no changes to other jobs or production files.


## Preflight evidence and completed GPU batch

All local preflight gates pass: 900 independent native mode checks, 4 trace
polynomial checks, weak acoustic/Alfven full-solver packets, uniform high-shift
reservoir, forced FOFC (first/quadratic saved fields identical when every cell is
flagged), matched-timestep compact rotating packets and an 8x8x4 meshblock control.
The optimized first-order acoustic run reproduces all saved float32 fields of the
previous first-order implementation exactly. Coarse local error comparisons still
show mixed accuracy, so no production adoption is warranted. Timing comparisons
from concurrent laptop runs are not GPU speed evidence. The older limited-linear
preflight is retained as an ablation. One serial32-cell 3D reference was stopped
by the agent for local cost; completed coarse controls and the GPU suite cover
its purpose. This was not a numerical failure.

The short and reference boxes have identical cell sizes AND timesteps in rotating
controls. For flat rigid rotation, dx=16/n and the reference half-width H determine
max coordinate light speed 1+abs(Omega)*(H-dx/2). Only the short-box CFL is reduced
from 0.25 by the ratio of short/reference maximum light speeds. This is a test
isolation choice; no production CFL rule is relaxed. Initial common-grid fields
must match to float32 output precision. Vectors are compared in the inertial
basis. Scalar density entropy advection also has an analytic solution.

The one-shot suite has 30 cases: 6 production-rate compact acoustic controls at
n=16/24 ; 12 high-shift acoustic/entropy controls at n=16/32 ; 10 planar acoustic
controls at nz32/64 (fresh fixed/reference and trace orders 1/2/3); and 2 actual
curved binary ADM/AMR smokes. Rotating GPU controls use 8x8x4 meshblocks so even
n=16 gives 16 blocks for 12 ranks. Acoustic fluid/divergence output every 20 M
(rotating) or 10 M (planar), and high-shift output every 0.05 M, permit inspection
through the encounter rather than only the final residual. Pressure-based
incoming proxies supplement the density proxy, separating entropy contamination.
These remain diagnostics, not exact nonlinear reflection coefficients or a
face-flux conservation audit.

Final isolated build/controller files are characteristic-improved-final.pid/log,
characteristic-improved-gate.pid/log/state.json and characteristic-rotating-submit.pid/log.
The first broad rebuild is characteristic-improved-build.log; the finalizer waits
until it exits, then overlays the frozen final header/pgen/call-site source and
rebuilds. Only improved-ready.json with final_source=true, trace_order=3 is final.
The publisher verifies all 6 source hashes and the executable before atomically
publishing the completed local gate. The submit controller alone owns qsub.
Inspect boundary-characteristic-rotating-20261018-v2/submission.json before any
manual action; accepted or unknown receipts prohibit duplicate submissions.
No job has been authorized to extend the completed physical zoom chain.

After completion run analyze_improved_boundary_suite.py with --previous pointing
to full-grmhd-aurora/results.json. Audit all 30 cases/360 rank report pairs, inspect
matched 3D differences, analytic entropy error, planar time series and runtimes.
Keep raw binaries ignored; archive small logs/manifests/resources. If the suite
cleanly stops before starting a new case near the wall deadline, diagnose and run
only unlaunched cases in a unique capacity batch after the prior job finishes.
Do not blindly repeat accepted or failed cases. Keep one node/all 12 GPU tiles,
28.4GB guards, one-hour PBS allocation and 55-minute solver cutoff throughout.

The additional planar fresh fixed/reference outputs and time histories are important:
the fixed ghost state enters an upwind HLLE solve, which can already transmit a
weak outgoing eigenmode with little reflected amplitude. Thus the word
characteristic alone does not establish an improvement. Incoming density proxies
can also include nonlinear entropy content. Compare pressure and velocity, time
of return and total state error against the larger domain before attributing a
late residual to reflection. Existing analytic reservoir ghosts still enter the
interior high-order reconstruction and CT stencil; replacing only the face trace
does not make that closure fully consistent or exactly nonreflecting. Local
results are deliberately reported as inconclusive on adoption.
