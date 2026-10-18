# Characteristic ghost reconstruction and native CT — partial validation

**Job 8942262 finished with a failure in its final curved-AMR case.** The other32
cases pass their audits. The rotating acoustic residual remains worse than
fixed, with a 2.35x runtime cost at n24. Keep the fixed production default.
See [results, plots and refined-physical-face diagnosis](aurora/README.md).
No retry was submitted; monitoring is paused after delivery.

The user authorized this implementation and repeated rotating tests on one Aurora
capacity node. The production default remains fixed. No physical smooth-excision
accretion/zoom continuation is authorized or submitted by this study.

## Discrete closure

Enable `problem/reservoir_boundary=characteristic` and
`problem/reservoir_ghost_ct=true`. The latter defaults false, preserving the v2
face-only method as an ablation. `reservoir_trace_order=1/2/3` now controls the
outgoing ghost continuation when the new flag is true.

The new closure runs **after primitive recovery**, including initialization and
each RK stage. The old user callback runs before recovery, so its interior
primitives are stale for an extrapolating boundary. It still supplies temporary
analytic physical ghosts for primitive recovery; the new closure replaces these
using the recovered interior and current ADM before reconstruction/fluxes.

1. Subtract the cell-local inertial-rest reference, including lapse, shift and W.
   At each physical boundary column form the native seven-variable projector at
   the face ADM metric, using the evolved CT normal field. Freeze this projector
   for the three interior samples in that column, project outgoing perturbations,
   and continue them into every ghost layer. Incoming perturbations refer to the
   same inertial-rest uniform reservoir. Component-limited continuations are
   reprojected with the same column projector before a shared positivity, density-floor and temperature-floor
   contraction, so limiting alone does not leave its outgoing subspace. A limited cell-average polynomial is
   used; positivity is enforced by contracting the perturbation toward the
   reservoir, without silently switching the boundary model.
2. Construct exterior staggered tangential fields from the proposed ghost B.
   March outward to solve the discrete divergence constraint for exterior normal
   faces, with the evolved physical normal face as integration constant. The
   physical normal faces and all active faces are never written by this extension.
   Recompute ghost bcc from those same staggered faces and regenerate ghost
   conservative state/temperature. B is densitized for the coordinate divergence
   constraint and undensitized at the projection metric.
3. Native PPM/HLLE consumes those ghosts for both fluid and induction fluxes.
   Volume reconstruction remains component-wise PPM; this does not claim exact
   incoming-mode cancellation in its nonlinear reconstructed face traces.
   Native cell electric fields use the same ghost primitives/B with transport
   velocity alpha*utilde/W-beta. Native SG07 builds a single shared edge EMF and
   native CT updates the staggered field. **No separate boundary flux or EMF
   overwrite remains in this mode.** Native FOFC sees the same states and uses
   its paired fluid/induction fallback; the old post-FOFC face projector is skipped.

Edges/corners have a unique ordered tensor-product extension: x slabs first,
then y including x ghosts, then z including x/y ghosts. The divergence solve uses
that same slab ownership so later slabs do not change faces of earlier cells.
This is an explicit approximate multidimensional corner closure, not an exact
transparent exterior or an axis-independent scattering solution. Both rotation
senses and corner crossings are covered by local gates; GPU comparisons remain
necessary. No velocity diode, gas damping or magnetic damping is introduced. Periodic
characteristic counters now count projected ghost columns and limited ghost
cells, rather than boundary face evaluations; they are not comparable as the
same operation count with v2.

Support is deliberately restricted to 3D blocks with at least three cells per
axis and **unrefined physical faces**. Interior static AMR is allowed and tested
with a coarse curved binary hierarchy. Refined physical faces are rejected
because they need a separate coarse physical-ghost/reflux closure; the code does
not claim that missing treatment is consistent. Strong/oblique curved waves and
long-time exterior reentry remain outside the present validation. Coordinate CFL
restrictions are unchanged.

## Gates and Aurora ownership

Initial local gates passed: all seven native mode projections (900 metric/lapse/
shift cases), face and ghost polynomial tests, uniform inertial-rest reservoirs
at zero/production/high rotation with both signs, compact near-corner packets, a
transverse magnetic Alfvén perturbation and forced FOFC. Exported **all-layer
physical ghosts**, including edges/corners, remain finite/positive and
solenoidal. Initial coarse rotating integration reaches t180 with matched grid
spacing/timesteps. Initial reports are archived separately from the final ones.

**All final-source local gates passed**, as recorded with exact source hashes in
`local-gate.json`: eight ghost/CT cases, six planar controls, three matched
rotating controls, two coarse curved-AMR cases, 900 native modes and 1,620 full
spectral-decomposition states. The coarse curved pair preserves 136 inner cells
bitwise. All-layer ghost divergence stays below 2.41e-16 in the dedicated gates.
An additional 8x8x4 MeshBlock control matching the GPU layout passes. In the
compact flat control, x/y exchange plus rotation reversal gives identical saved
density/pressure and velocity/B differences below 2.3e-15. These bounded checks
do not establish arbitrary corner scattering symmetry, production accuracy,
interior convergence or horizon-resolved drain behavior. The GPU pair attempted
the full hierarchy, but its characteristic case exposed refined physical faces
and aborted at the support guard. The coarse preflight missed that topology.

Remote root:
`/lus/flare/projects/MHDTidal/hzhu/cyclic_zoom_profile_20261013`.
Unique trial: `boundary-characteristic-rotating-20261018-v3`.
The v2 executable/source are archived in `characteristic-v2-8942061`.
Only `source-characteristic/build-characteristic` are modified.

**Aurora capacity job 8942262 was accepted on 2026-10-18**, and its scheduler
record confirms queued state, one node and one-hour walltime. The final guarded
GPU build passed with executable SHA256
`d48198a64798b54ab7e334b7b14f7f0482f0285fdbec19ae53e34d38c71689c9`.
The accepted receipt, source/executable hashes, build resource reports and initial
scheduler record are preserved in `provenance/`. Final qstat records F/143; the
last characteristic case aborted because 32 outer x-face blocks are refined.
The other32 cases passed384 rank audits. This is not an all33-case pass and there
is no full-hierarchy curved comparison. The controller finished, no retry is
submitted, and the final [analysis](aurora/README.md) retains all failure evidence.

Historical serialized owners (all finished):

- `trial/build.sh`, with `build.pid/log/started`, owns the broad guarded build.
- `trial/finalize_characteristic_ghost_ct.sh`, with `finalize.pid/log/started`,
  waits for it, installs the final support guards/surface-only divergence kernels
  and does one incremental rebuild. Only `build-ready.json` with
  `final_source=true,ghost_ct=true` is final. Never mutate source during a build.
- `trial/submit_characteristic_ghost_ct.py --wait` alone may submit after the final
  source/executable hashes and `local-gate.json` match. `submission.json` is
  created exclusively before qsub; an accepted or unknown receipt prohibits
  duplicates. Do not restart healthy controllers or submit on partial gates.

The suite repeats the 30 v2 controls and adds three planar Alfvén cases
(reference/fixed/ghost-quadratic, nz32). It uses capacity only, one node/all twelve
GPU tiles, validated SANE environment, PBS one hour, solver `-t00:55:00`, 28.4GB
aggregate guards, per-case 900s watchdog and launch cutoff 2400s. No other jobs,
production source/build, old cycle controllers or debug queues may be touched.
If launch cutoff cleanly leaves cases unstarted, diagnose and continue only those
in a unique one-node capacity allocation after this accepted job finishes.
Actual failures require evidence and diagnosis, never a blind retry.

At completion preserve all outputs/logs/source and executable hashes/all twelve
rank resources and memory reports for each of33 cases. Compare against both fresh
fixed/reference controls and v2 face-only results. Require normal targets,
finite/positive fields, scaled divB<1e-9, one-host/tile affinity and guards. Use
full pressure-mode decomposition and time histories, rather than interpreting
an entropy-contaminated density proxy as a reflection coefficient. Compare
actual curved-AMR inner r100 and native-CT Alfvén controls. Assess accuracy and
runtime honestly; no adoption without a consistent rotating-frame benefit.
Archive small provenance, keep raw binaries ignored, commit/push this branch,
deliver plots and qualified findings, then pause the automation.
