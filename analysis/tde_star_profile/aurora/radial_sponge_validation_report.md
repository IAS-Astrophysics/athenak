# Large-radius residual Z4c sponge: Aurora validation

## Configuration

- Source commit: `5c9c20c53abd1e66d96d59af5904aad110c6c818`
- Aurora executable SHA-256:
  `b44c4a9ce347c84e6600ce9a30ff22f5dfbf24028278fdcf410f0a9260297b09`
- Build: Release, precise floating point, MPI, SYCL, Intel PVC, Level Zero
- Queue/project: `debug-scaling` / `MHDTidal`
- Production domain: `[-1024,1024]^3 M`
- Sponge: Kerr-Schild radial profile, start `512 M`, quintic ramp `128 M`,
  full damping time `16 M`

The pgen supplies the source through a dedicated `Z4c_UserSrc` task. The task
runs after the bulk Z4c RHS is constructed and before the existing Sommerfeld
boundary RHS and explicit RK update. The source kernel loops over all
`nz4c = 25` residual fields. It never modifies the analytic background.
At the outermost active cell, the later Sommerfeld task replaces the RHS for
the 11 fields that it covers; the other 14 retain the sponge contribution.
Thus all 25 fields are damped through the sponge volume, while the physical
surface itself retains the existing boundary prescription.

## Mesh preflight

Aurora job `8712773` completed successfully in mesh-only mode.

- Root grid: `2 x 2 x 2` MeshBlocks
- Initial MeshBlocks: `1184`
- Physical levels: `0` through `13`
- Level histogram for physical levels 1 through 13:
  `52, 84, 84, 84, 80, 108, 140, 148, 84, 84, 84, 88, 64`
- Planned 32-node load: `1184 / 384 = 3.083` MeshBlocks/rank
- Configured allocation: `max_nmb_per_rank = 16`

## Profile samples

The initialization log reports:

| Radius | Measured sigma |
| ---: | ---: |
| `512 M` | `0` |
| `576 M` | `0.03125 / M` |
| `640 M` | `0.0625 / M` |
| `1024 M` | `0.0625 / M` |

These are the exact expected values for the quintic ramp and `tau = 16 M`.

## One-node pulse tests

All accepted pulse jobs used 12 MPI ranks on one Aurora node and completed
without nonfinite history values.

### Frozen residual gauge

Jobs:

- Undamped control: `8714178`
- Radial sponge: `8714184`

At `t = 24 M`:

| Diagnostic | Control | Sponge | Sponge/control |
| --- | ---: | ---: | ---: |
| `Theta-max` | `7.5673070e-9` | `1.6881893e-9` | `0.2231` |
| `res-outer` | `2.9185362e-7` | `6.5156010e-8` | `0.2233` |
| `res-inner` | `8.1268325e-14` | `8.0269125e-14` | `0.9877` |

The expected full-strength attenuation is
`exp(-24/16) = 0.22313`. The measured pulse and all-field exterior maxima agree
with this value. The inner source is identically zero by construction; the
small history difference is evolved roundoff/truncation response, not a sponge
source inside `r <= 512 M`.

### Evolved residual gauge

Jobs:

- Undamped control: `8714158`
- Radial sponge: `8714169`

The seeded Theta pulse is damped correctly at early times: at `t = 7.2 M`, the
ratio is `0.6376`, consistent with `exp(-7.2/16) = 0.6376`. However, a coupled
residual-shift mode grows faster with the all-field sponge:

| Diagnostic at `t = 24 M` | Control | Sponge |
| --- | ---: | ---: |
| `beta-res` / `res-outer` | `6.1341287e-1` | `1.7770187` |

Moving the pgen source before versus after the Sommerfeld boundary RHS did not
change this result. The Sommerfeld implementation itself operates only on
residual `Khat`, `Theta`, `Gamma^i`, and `A_ij`; it does not extrapolate or
couple the analytic background. It does not provide a characteristic boundary
treatment for the remaining gauge and metric residual fields.

The physical ghost-fill path also extrapolates `u0`, which is the residual
array. The analytic Kerr-Schild background is generated independently and is
only added when reconstructing the full state. Therefore background
extrapolation is not a direct error source in the present Sommerfeld path.
The more likely boundary defect is that the placeholder RHS condition is not
the characteristic decomposition of the complete Z4c plus gauge principal
part.

This result means that the sink operator is implemented with the correct sign
and timescale, but uniform damping of all evolved residuals is not by itself a
complete cure for the known evolved residual-gauge instability. A
background-consistent characteristic/constraint-preserving boundary condition
or a correction to the residual gauge subsystem remains a separate follow-up.

A complete follow-up should derive the face-normal characteristic fields and
speeds for the exact Z4c, lapse, and shift choices used here. Outgoing modes
should be taken from the interior, while incoming physical, gauge, and
constraint modes should be assigned residual targets consistent with the
analytic background. The implementation should fuse all fields in one device
kernel per face orientation (with explicit edge/corner handling), making the
cost proportional to boundary area and negligible compared with the volume
RHS kernel. Plane-wave and spherical-pulse reflection tests, followed by
Schwarzschild and spinning-background tests, are required before replacing
the current placeholder.

## Production smoke

Aurora job `8714194` completed successfully.

- Nodes/ranks: `32 / 384`
- Wall-time request: `01:00:00`
- Scope: fresh production initialization plus one RK cycle
- PBS result: exit status `0`, used wall time `00:00:15`
- Source/executable identity matched the configuration recorded above
- MeshBlocks: `1184`; 352 ranks held 3 blocks and 32 ranks held 4 blocks
- Maximum per-rank load: `4`, below `max_nmb_per_rank = 16`
- TOV initialization completed and star tracking was valid after the cycle
- Cycle: `t = 5.859375e-4 M`, with `dt = 5.859375e-4 M`
- Tracked density maximum changed from `3.0736352e-3` to `3.0756602e-3`
  during initialization and the first cycle
- All MHD, problem, and Z4c history values were finite; `bad-metric = 0`
- No MPI, SYCL, Level Zero, allocation, or out-of-memory errors were reported

The radial diagnostics at the end of the cycle were
`res-ramp = 8.6282825e-10` and `res-outer = 6.8168637e-10`. The source
definition is exactly zero at every cell with Kerr-Schild radius
`r <= 512 M`; the `res-inner` history is a residual-field maximum, not a
measurement of the source itself.

## Verdict

The radial pgen source, parameter validation, AMR layout, SYCL/MPI build,
profile, isolated damping rate, and 32-node initialization path all pass their
targeted checks. The source has the intended sign and produces the measured
`exp(-t/16)` attenuation without timestep stiffness.

The broader claim that damping all 25 residual fields is sufficient to make
the existing boundary treatment safe is **not established**. In the evolved
residual-gauge pulse test, the sponge increased an exterior shift mode by a
factor of about 2.9 at `t = 24 M`, despite correctly damping Theta. The
one-cycle production smoke is too short to measure a material central-density
or trajectory difference. This implementation is therefore suitable for
controlled follow-up runs, but should not yet be described as a complete
production boundary-condition cure.
