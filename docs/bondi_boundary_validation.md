# Finite-radius Bondi boundaries and a rotating single-hole control

Use `inputs/dyngr/bondi_matched_zoom.athinput` for the new inflow test.
The earlier `dynbbh_bondi_zoom.athinput` remains an ambient-reservoir infrastructure
test, with its original constant ghost state.

## Boundary prescription

Zero Eulerian velocity in the old reservoir did not force grid corotation.
However, in ingoing Kerr-Schild coordinates it implies coordinate velocity
`-beta_inertial`, rather than the velocity of a prescribed Bondi solution at a
finite radius. The old smoke box also lay inside its nominal Bondi radius.

`problem/initial_condition=bondi_inflow` now supplies the spherical relativistic
Michel/Bondi density, pressure, and radial inertial KS coordinate velocity at
each exterior ghost cell. The solution uses the HSW formulation already present
in the upstream `gr_bondi` problem generator. Its KS time component is evaluated
with a rationalized expression that remains finite at the Schwarzschild horizon.

With grid velocity `w=Omega cross r`, the conversion is

```
V_rot = V_Bondi - w
beta_rot = beta_inertial + w
v_Eulerian = (V_rot + beta_rot)/alpha
W = 1/sqrt(1 - gamma_ij v_Eulerian^i v_Eulerian^j)
```

The code stores `W v_Eulerian^i` and converts to conserved variables using the
actual local ADM metric. The rotation cancels between coordinate velocity and
shift, including outside the light cylinder. Adding `-Omega cross r` directly
to an Eulerian velocity would be incorrect.

Only exterior ghosts are prescribed; internal AMR ghosts use exchanged data.
Magnetic exterior faces retain the previous CT treatment. These hydro controls
use zero magnetic field. This is a fixed analytic inflow boundary, not a
nonreflecting characteristic boundary for binary-driven feedback.

The central monopole is an exact single-hole solution, but only an initial seed
and an outer feeding condition for a binary. Near individual binary horizons,
the initial seed is limited to Eulerian `v^2 <= 0.9` if necessary. The outer
boundary is never velocity-limited: it aborts if the requested state is not
within the conservative `v^2 < 0.9` safety bound. Early binary evolution therefore
contains initial transients.

## Matched configuration

The matched single-hole and binary runs differ only in
`problem/single_black_hole=true/false`; their gas, grid, rotation, and zoom
settings are identical. A separate upstream hydro control uses the original
`gr_bondi` pgen and central zoom mask, with the same Bondi parameters and finest
resolution. It is selected by `problem/pgen_name=gr_bondi` in this executable.

| Parameter | Value |
| --- | --- |
| Reference mass | 1 |
| Binary masses | 0.5 + 0.5 |
| Single-hole mass | 1, at the origin |
| Separation | 32 M |
| Adiabatic index / adiabat | 5/3 / 1 |
| Sonic radius | 8 M |
| Derived Bondi radius `2/cs_inf^2` | 161.227 M |
| Outer box | +/-256 M |
| Frame frequency | `32^(-3/2) = 0.005524271728019903 / M` |
| Light cylinder | 181.019 M |
| Root grid / block | 64^3 / 8^3, four ghost cells |
| Physical refinement levels | 8 |
| Finest cell width | M/16 |
| Cyclic zones / residence time | 4 / 10 M per zone |
| Intended debug evolution | 64 M, about one complete zoom cycle |

The sum of the input hole mass parameters is exactly one. The prescribed
boosted superposed metric is not a constraint-solved binary spacetime, and its
asymptotic energy need not equal that sum. The Newtonian binding-energy scale
at separation 32 is `|E_bind|/M = 1/(8*32) = 0.00390625`. From the initial
binary field, the mean diagnostic `r(1+g_tt,inertial)/2` over `128<r<192`
is 1.01018 (angular scatter 0.01064); this finite-radius metric diagnostic is
not an ADM mass measurement. Thus the far-field potentials are close but not
identical. A 64 M run is short compared with either an orbital or Bondi time.
The tests are adiabatic hydro controls with no gas self-gravity, cooling, star
formation, or radiation feedback. They cannot determine fragmentation at a
physical parsec-scale Bondi radius; the gas density normalization also remains
free in this non-self-gravitating calculation.

## Local validation

On the Mac, the single-hole initial density matches the independently evaluated
upstream pgen exactly in float32 output; maximum pressure/velocity differences
are `4.66e-10` / `2.33e-10` in the comparison shell `3<r<7`.

For a domain +/-8 M, evolution to t=1 M, and prescribed Omega=0.2/M, the rotating
grid extends beyond its light cylinder. The L1 density errors against the
initial analytic solution are:

| Root resolution | Inertial | Rotating | Frame difference |
| --- | ---: | ---: | ---: |
| 16^3 | 0.4239% | 0.4595% | 0.2915% |
| 32^3 | 0.1038% | 0.1144% | 0.0760% |

Pressure frame differences fall from 0.5570% to 0.1418%. Both solutions and
their difference converge at approximately second order. A profiled fixed-grid
run recorded one metric evaluation over all timesteps.

The matched 1,520-block binary mesh also completed two bounded Mac steps at
M/16, with positive density/pressure, zero div B, one metric evaluation, and
sampled peak RSS 11.20 decimal GB. This is not a steady accretion validation.

A restart regression exposed six-digit serialization of computed real defaults,
which truncated the new frame frequency. Real defaults and updates now serialize
with `max_digits10` precision. The upstream Bondi pgen also now restores its
analytic constants before returning on restart.
Both regressions pass after the fixes; the upstream continuous/restarted saved
fields match exactly in float32 output at t=2 M. Full numerical records are in
[the local validation report](bondi_boundary_local_validation.json).

The first Aurora attempt (8924152) reached the first return event at t=40 M,
then its original hydro geometric refinement tried to grow 736 blocks to 2,752,
exceeding the configured 1,600-block limit. The existing complete-family storage
and saved-parent restoration now apply to hydro as well as Valencia. Restart
format 2 marks these semantics; format-1 prototype and upstream checkpoints
are rejected. The failed attempt's outputs and resource report are retained in
`results-attempt1` on Aurora.
The new bounded hydro regression completes six transitions and restores all
288 initial blocks, with finite positive saved fields. The complete Mac
boundary suite takes 75.36 s with sampled peak RSS 2.14 GB.

## Aurora protocol

The source/build/results live in an isolated
`/lus/flare/projects/MHDTidal/hzhu/cyclic_zoom_boundary_20261012` directory.
`submit_bondi_debug.py` checks that the user's debug queue is empty, submits
only to `debug`, and records a receipt to prevent duplicate submissions.
It does not touch other jobs or submit to debug-scaling.

One node runs the four cases serially, on one GPU, within a 30-minute allocation.
Each case has a six-minute graceful Athena wall limit and a seven-minute outer
deadline. A wall-limited case may finish before t=64; its actual time must be
reported, not treated as a completed physical-time comparison.

The Kokkos Tools guard limits combined tracked host/device allocations to
24 decimal GB. The process-tree RSS guard is 20 GB. GPU runs use these physical
allocation/RSS checks instead of a virtual-address-space limit, with headroom
below the requested 30 GB. Runtime and driver allocations are not all visible
to Kokkos; the report distinguishes tracked allocations from resident memory.

Kokkos 4.6.1 is incompatible with the default oneAPI 2026.1 USM extension API.
The isolated build pins the installed oneAPI 2025.3.1. Mac runs retain full
Kokkos bounds checks; the GPU build uses generic SYCL/SPIR64, optimization -O2,
general debug and DualView checks, with device bounds checks disabled. This
avoids very slow ahead-of-time compilation of unrelated kernels in this older
source tree. SYCL persistent caching is confined to the campaign directory.

Reproduce the local checks with `test_bondi_frames.py` under `run_bounded.py`.
`make_bondi_campaign.py` prepares the inputs without submitting anything.
`analyze_bondi_campaign.py` checks saved fields, mesh restoration, resource
reports, and frame differences after the debug job completes.
