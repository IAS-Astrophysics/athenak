# Smooth-excision continuation selected by the user

The restart anchor is the native final checkpoint of job 8938971,
`evolution-magnetic-8938971/rst/magnetic_bondi_binary.00022.rst`:
time 9903.615850407483 M, cycle 125336, 1408 blocks, finest zone 0,
zoom event 14. This is evolved magnetized data, not a new uniform reservoir.
The user explicitly selected this finest checkpoint and two full subsequent
zoom cycles. The later hard-excision job 8939546 is excluded from the new chain.
Its automatic continuation was stopped; it exited normally at 10940.7 M.
All twelve ranks passed the original field/resource validator (conservative
sum 22.67 GB; scaled divergence maximum 9.88e-12).

The new source/build/output names use `magnetic-smooth` to preserve the old
experiment and the validated SANE production installation. Build flags,
runtime modules, CPU/tile binding, and guards are the same as the magnetic
pilot. Only debug-scaling, one node, twelve GPU tiles are permitted for this
study. Each job requests one hour and uses the solver's `-t 00:55:00` cutoff.
Tracked allocation plus RSS guard budgets total 28.4 GB.

## Physics and restart change

Enable the user's existing `coord/smooth_excision=true`, puncture width fraction
1 and polynomial exponent 1; retain radius 0.8 M, density target 1e-8 and pressure
target 1e-12. The weight is w=s^(2n)[(2n+1)-2ns], with s=1-r_KS/r_ex inside
the boosted puncture region and zero outside. Density, pressure and Eulerian
spatial four-velocity are blended during primitive recovery. The optional
additional inflow target and resistive magnetic damping remain disabled.
The face-centered field remains evolved by constrained transport.

The boost velocities used for the puncture shape and the target spatial vector
are physical vectors. A rotating coordinate grid does not set those vectors to
zero. With unchanged time slices, the Eulerian velocity transforms by spatial
rotation; the coordinate velocity additionally changes by the grid rotation.
The reservoir initial data and all six user boundaries continue to use the
shared inertial-rest helper, including lapse and shift. No boundary or seed
velocity prescription is changed for this continuation.

`cyclic_zoom/restart_reset_finest_dwell=true` is an explicit one-shot restart
option. It is accepted only at a finest checkpoint with a positive finite dwell.
It sets next_time=t_restart+runtime, retains the event id, and consumes itself
in the parameter header before any subsequent checkpoint. Thus this physics
switch receives a full 250 M finest dwell; wall-time restarts do not reset it.
There is no change to the checkpoint binary format or to other stage dwell times.

Anchor event 14 -> first full return 28 -> second full return 42. The nominal
roundtrip is 9682.35 M; transition timesteps slightly extend it. A bounded global
target 29500 M provides roughly 230 M on the final finest visit. Completion is
accepted only if the final native checkpoint is in zone 0 at event 42 with at
least 200 M since entry. Two full new cycles must not be inferred from global
time alone. If these checks fail, the controller stops for review. No additional
cycles are automatically authorized.

## Validation and limits

`test_zoom_dwell_reset.py` verifies the new dwell, its consumed checkpoint flag,
unchanged next_time on a second restart, and rejection on a fresh initialization.
The local bounded regression uses less than 8 GB and passed.

`test_smooth_excision_frame.py` measures single/binary inertial versus rotating
magnetized controls at 32^3 and 64^3. These are short, deliberately coarse tests;
the 64^3 excision diameter has only 6.4 cells. Finite positivity and CT checks
alone are not a frame-convergence claim. In particular the existing drain is
applied per primitive-recovery call, so changed CFL timesteps can change its
numerical strength. Pressure errors near the drain need to be reported rather
than hidden by a broad tolerance. The production hierarchy resolves the physical
puncture diameter much more finely. A separate, short switch test on the actual
GPU checkpoint is required before the physical continuation.

`submit_magnetic_smooth.py` uses receipts written before qsub, verifies the
native checkpoint and validated predecessor, and checks debug-scaling is empty.
`watch_magnetic_smooth.py` follows only accepted descendants and stops on failed
jobs, validation failures, unknown submission outcomes, or its own stop file.
`validate_magnetic_smooth.py` checks all twelve rank/resource reports, finite
positive fluid states, relative magnetic divergence, inherited smooth settings,
one-shot reset logs, and final event/phase accounting. Physical relaxation still
requires accretion and shell-mass comparisons at matched finest visit phases.

Completed local frame controls (t=0.5 M, identical inertial-rest reservoir,
zero pressure perturbation; the inertial binary metric is updated dynamically):

| Quantity, r_A > 3 M | Single 32^3 -> 64^3 | Binary 32^3 -> 64^3 |
| --- | --- | --- |
| density relative L1 | 0.0275% -> 0.00622% | 0.0451% -> 0.0111% |
| pressure relative L1 | 1.036% -> 0.219% | 2.808% -> 0.341% |
| spatial u vector relative L1 | 0.395% -> 0.111% | 0.384% -> 0.0972% |
| Btilde vector relative L1 | 0.0261% -> 0.00435% | 0.0534% -> 0.0217% |

Every run finished with finite positive density/pressure and absolute divB below
1e-8. All far-exterior discrepancies decreased by at least a factor of two.
This passes a short far-exterior coordinate consistency check, not an interior
sink or long-time convergence test. In the broad region r_A>1.05 M, pressure L1
errors at 64^3 remain 39.6% (single) and 31.0% (binary); that region includes
inside the single-hole horizon and unresolved near-drain gradients. Drain
pressure errors are 40.1% and 35.8%, respectively. No coordinate-specific change
to the physical boost reference is justified by these measurements; retaining
the original smooth prescription avoids changing the intended sink physics.
Full measurements and resource records are in `smooth-controls`.

## Launch state, 2026-10-17 20:04 UTC

The isolated GPU build passed (903.35 s, peak RSS 5.224 GB under the 24 GB
build guard). Its executable SHA256 is
`59805f259870066369f5d0525660ff599a05933c172336e18627b4f76829e682`.
Build modules, compiler, complete source hashes, CMake flags and resource record
are saved in `smooth-controls/build`.

The half-M switch test was accepted as **8939714**, with its immutable receipt
in `smooth-controls/magnetic-smooth-check.receipt`. This is a separate test and
is not counted as a physical smooth cycle. `start_magnetic_smooth_chain.py`
waits for its normal exit, validates all ranks and fields, then submits the
physical anchor from the original finest checkpoint. It does not continue the
half-M test checkpoint. The physical chain has not passed its two-cycle target
at this launch-state record. Follow accepted descendant receipts and native
checkpoints for current status; do not interpret this document as a completion
claim. A quiet 30-minute completion check replaces the old hydro automation.

## Switch validation and physical launch, 2026-10-17 20:36 UTC

Switch test 8939714 completed normally at native t=9904.115850407483 M,
cycle 125405, zone 0/event 14, with the reset next_time retained at
10153.615850407483 M. All twelve ranks passed; conservative RSS plus tracked
allocation peaks totaled 23.396 GB, and maximum scaled magnetic divergence
was 2.03e-12. Saved density and pressure were finite and positive. This is
only a half-M launch check, not evidence of relaxation. Its complete validator
and resource records are in `smooth-controls/gpu-switch-8939714`.

The guarded bootstrap accepted physical job **8939726** from the original
9903.61585 M anchor (receipt `smooth-controls/magnetic-smooth-anchor.receipt`)
and transitioned to the physical-chain monitor. At this check the job is running
normally in its initial fresh finest dwell, near t=9948 M; neither new full
cycle has completed. No independent continuation was submitted by the heartbeat.

## First smooth return and renewed monitoring

After renewed SSH access, job 8940293 was verified running under the healthy
existing controller. The first requested roundtrip completed at event 28,
t=19586.65420246086 M; the second is pending. The latest validated finest saved
state is cycle 236032 at native t=19700.932791173836 M from job 8940212.
Updated xy density/temperature/physical-beta panels and provenance are in
[smooth-latest](smooth-latest/README.md). The completion automation is active
again, with its original two-cycle target and hardware restrictions unchanged.
