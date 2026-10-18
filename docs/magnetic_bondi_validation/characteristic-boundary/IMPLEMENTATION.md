> New implementation: [characteristic ghosts and native CT](ghost-ct/README.md).
> Final local preflights pass; a guarded one-node capacity repeat is being built.
> Production selection remains fixed pending the new rotating comparisons.

> Rotating suite **8942061 completed**: all 30 cases and 360 rank audits pass.
> The quadratic trace improves planar acoustic pressure diagnostics, but remains
> worse and slower for the rotating 3D packet. Keep the fixed production default.
> See [updated results](improved-rotating/aurora/README.md) and the
> [explicit seven-wave decomposition](improved-rotating/FULL_DECOMPOSITION.md).
> Job 8941903 below describes the historical first-order version.

# Full GRMHD characteristic implementation — bounded validation complete

User authorization on 2026-10-18 covers resubmitting corrected controls and
implementing/validating this boundary, using one capacity node at a time.
Corrected fixed-boundary control job **8941753** completed normally. All 48 rank
resource/memory audits passed. Peak conservative RSS+tracked allocation sum was
10.622 GB, scaled divergence <=5.09e-15, and maximum saved-ghost inertial velocity
error <=4.77e-7. First-active-layer pressure deviations at t64 are <8e-6 relative
across these four controls. These are SHORT fixed-reservoir controls, not evidence
of characteristic reflection performance or long-term equilibrium.

## Implementation

Add `reservoir_boundary = characteristic` under `<problem>` in the input file
(default `fixed`), or prepare the explicit keys with
`prepare_characteristic_control.py --characteristic` before a CLI override.
AthenaK command-line overrides cannot introduce missing keys.
The current implementation supports ideal EOS, uniform inertial-rest reservoir,
HLLE and no passive scalars. `src/dyn_grmhd/characteristic_reservoir.hpp` forms
the seven-variable primitive-space flux Jacobian by differentiating the existing
Valencia conservative conversion and flux routines in the **actual face ADM
metric**. The face-normal CT magnetic field is held constant during differentiation.
The final Jacobian is evaluated at the positive arithmetic mean of the interior
and reservoir primitive states, so the contact/entropy speed tracks the actual
boundary flow. The earlier ambient-point linearization is retained as an ablation,
not the final algorithm. A dimensionless scaled matrix-sign Newton iteration constructs the outgoing
projector, avoiding a separate approximate seven-wave eigenvector formula. The
matrix-sign method follows the standard scaled Newton construction described by
[Higham](https://nhigham.com/2020/12/15/what-is-the-matrix-sign-function/).

The reference state is the position-dependent inertial-rest reservoir converted
with local lapse, shift and spatial metric. The evolved normal B is used in both
states, rather than forcing it back to the ambient value. The outgoing projection
is applied to the primitive perturbation relative to that reference. Stationary
modes receive an outgoing bias of 1e-8 times the normalized Jacobian scale. Causal
light-cone bounds short-circuit faces where every MHD characteristic necessarily
enters or necessarily leaves. This does not alter the coordinate CFL bound.

The cell trace subtracts its own local inertial-rest background velocity before
adding the face reference; a varying ADM background is not treated as an outgoing
perturbation. Baryon mass and the velocity ceiling come from the native EOS.
A candidate with nonpositive density/pressure or excessive Lorentz factor is
contracted toward the reference state, with a counter reported per rank. Matrix
or positivity failure aborts rather than silently reverting to a fixed boundary.
The interface trace is currently first order: adjacent interior cell plus projected
exterior state. Both enter the existing nonlinear HLLE solve. Fluid fluxes and
face electric fields are set together BEFORE the FOFC positivity predictor and
restored after its fallback replacements, then use the normal
edge-EMF/CT path; physical normal B is never overwritten. Tangential ghost strips
are covered for the CT corner stencil. Existing analytic reservoir ghost data
remain the reconstruction closure. The combined scheme needs explicit accuracy,
reflection and multi-face-corner tests; it is not an exact nonlinear transparent
boundary and is not approved for production use.

## Local verification and test infrastructure

A separate `build-mac-characteristic` binary compiled with bounds checking.
The production C++ inverse/sign kernels passed 140 cases against independent
NumPy eigensystems, max absolute error6.67e-16. Full-solver flat-ADM uniform
reservoir tests passed at Omega=0 and sqrt(2), box+/-8, nx16. The latter reproduces
maximum directional shift11.31 and equatorial-corner magnitude16. Saved density
and pressure are unchanged; maximum absolute divergence2.14e-16.
A real binary ADM/production-domain characteristic smoke test reached t4 normally.

`problem/reservoir_test_flat_metric=true` is restricted to validation with no
Z4c, no zoom and no excision. It fills analytic flat spatial metric, lapse1 and
rigid-rotation shift. Optional small acoustic or Alfven Gaussian packets propagate
along z. Zero-rotation tests may use periodic transverse boundaries. These are
limits of the FULL nonlinear GRMHD evolution/HLLE/CT code, not the standalone
linear prototype. They isolate boundary errors before curved-space controls.

The first packet experiment exposed a mixed-boundary bug: PrimToConBoundary
converted ALL outside-box ghosts, including periodic ghosts whose primitive
values had not been refreshed after conserved exchange. This reintroduced stale
initial packet values at transverse faces. Its results in pulse-tests-slim are
INVALID as reflection measurements. The fix restricts conversion to USER faces;
production runs with six USER faces take the same filter as before. New packet
tests assert transverse density uniformity to detect this failure. Those reruns
are in `build-mac-characteristic/pulse-tests-periodic-fix` (32 cells along z).
`test_grmhd_characteristic.py` now also supports extended-domain references;
subtract their matched-grid incoming-mode residual before attributing it to the
boundary. A final-time incoming amplitude alone mixes reflection with interior
truncation error and initialization transients.

## Aurora state and next required actions

Corrected fixed-boundary controls **8941753** passed and all outputs are downloaded
under `build-mac-characteristic/aurora-controls/boundary-characteristic-20261018-r1`.
The plots, audit and small provenance are in `corrected-controls/`.

Full-GRMHD validation job **8941903** completed normally in `capacity`, one node/all
12 GPU tiles. All 14 cases and168 rank resource/memory audits passed. Detailed
results and plots are in [full-grmhd-aurora/README.md](full-grmhd-aurora/README.md). Its immutable receipt and submitted files are preserved in
`full-grmhd-aurora/provenance/`. The isolated remote source/build are
`source-characteristic` and `build-characteristic`; the original production
source/build and completed physical study remain untouched. The final executable
SHA256 is `4cbb3de390c9f1c5697f884b1439202812d4ae56ae9fb640be566679735ef065`.
The final build includes the periodic ghost fix, primitive-mean Jacobian and
pre/post-FOFC flux integration. `final-ready.json` records
`fofc_boundary_predictor=true`. Both final header and flux call-site source hashes
were checked before publishing the local packet gate. The original wait helper
timed out without submission; after all gates passed a single one-shot invocation
accepted 8941903. Never duplicate its accepted receipt.

The suite has **14 cases**: acoustic and Alfven packets at nz32/64 with fixed,
characteristic and larger-domain reference cases (12), plus paired fixed and
characteristic real binary ADM/AMR/smooth-drain smoke cases at t=0.05 M. GPU packet
seed is 0.01 (local seed 0.001), to reduce float32 output quantization. The larger
reference domain has the same cell size and the pulse has not encountered its
boundary. All cases retain solver `-t 00:55:00`, one-hour PBS walltime, validated
SANE environment and existing28.4GB guards. No further physical zoom cycles are
being submitted. Monitoring is paused after final delivery; no retries or
physical continuations are pending.

Every saved output and all 12 resource/memory reports per case were downloaded
and audited. The two analysis scripts, source/receipt verification, output hashes,
plots and small provenance are retained. All fields are finite with positive
density and pressure; maximum scaleddivB is 2.186e-15. The conservative sum of
per-rank peak RSS plus tracked allocations is at most 21.635 GB. Last logged limiter
samples show zero contractions; these samples need not be final counters.
The paired real binary ADM/AMR cases have identical saved fluid values in 447528
cells at r<100 M,t=0.05 M. This is an integration-isolation check, not a curved-space
reflection or long-time stability result.

At nz64, incoming-mode residual relative to seed is 0.01908% fixed vs 0.03370%
characteristic for acoustic, and 0.33488% vs 0.13826% for Alfven. Accuracy is mixed;
Alfven residuals increase from nz32 to nz64, so these tests do not establish spatial
convergence. The characteristic packet runs take 2.30–2.53 times the fixed run time
on these small boundary-heavy grids, with identical timestep counts. This cost
ratio is not a production scaling prediction. Keep the fixed default.

Remaining requirements before adoption: accurate higher-order outgoing traces,
strong/oblique fast/slow packets, curved-space wave reflection, long-time rotating
multi-face/corner behavior and exterior-memory assumptions, plus performance
optimization and stronger convergence evidence. The seed 0.01 nonlinear tests use
a linear incoming-mode diagnostic; their differences are not pure reflection
coefficients. No global equilibrium or exact transparency is established. Do not
resume the completed smooth-excision cycle chain.

## Final local packet and mode results

The final native full-Jacobian analytic-mode check passed **900 cases** across
three directions, both outward orientations, lapse0.7/1.3, five shifts and three
spatial metrics including off-diagonal components. Acoustic, Alfven and entropy
mode selection agrees with independent analytic expectations; maximum normalized
projector error5.02e-6 (test tolerance1e-3). These are projector checks, not bounds
on the evolved solution error.

At nz32 and seed 0.001, maximum reference-subtracted incoming residuals as fractions
of seed are:

| Packet | Fixed | Characteristic (final mean) |
|---|---:|---:|
| Acoustic | 2.921e-5 | 8.754e-5 |
| Alfven | 1.088e-5 | 3.101e-5 |

The first-order characteristic trace has **not demonstrated improvement** over
fixed in these weak-packet tests. Absolute residuals are small and snapshot
float32 quantization is material. The earlier ambient-linearization acoustic
value1.832e-4 is an ablation, not the final result. Plots and paired comparisons
are in `local-packet-reference/`; historical ambient ablations are separate.
The final pre/post-FOFC packet reruns are bitwise identical in all saved float32
fields to the preceding primitive-mean post-only runs, as recorded in
`fofc-integration-comparison.json`. This does not make the predictor integration
optional: it must test the actual boundary flux.

The pre/post-FOFC closure is identical within a stage because FOFC's test
conversion does not mutate w0 or bcc0. The first evaluation makes the predictor
check the actual flux; the second prevents flagged boundary faces reverting to
fixed ghost states. Counters count face evaluations, including these two passes;
periodic log samples need not be final totals. Tests have only been run in double
precision. Single precision, strong disturbances, oblique fast/slow packets,
curved-space reflection and long-time corner/exterior-memory behavior remain
outstanding. Keep the default fixed boundary pending evidence of benefit.
