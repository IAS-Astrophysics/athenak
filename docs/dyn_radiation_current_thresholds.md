# Dyn Radiation Current Regression Thresholds

These thresholds summarize the current `dyn_radiation` solver baselines recorded
in `docs/dyn_radiation_tests.md`.  They are intended as regression guards for
performance and correctness work, not as final physics tolerances.

## Solver-Quality Smoke Tests

The stress harness must pass all non-optional executable checks for:

- CKS and ADM tetrad construction.
- CKS and ADM flat beam transport.
- ADM black-hole beam and Z4c ADM wave smoke tests.
- Linear wave and SMR linear wave tests.
- CKS and ADM positivity limiter tests.
- Nonlinear source-iteration test.
- FLRW redshift, lapse-gradient, and momentum-source formal tests.
- Restart/resume where restart support is built.

The harness rejects non-finite stdout/stderr or text outputs.  Current guard
floors are:

- ADM black-hole beam minimum parsed timestep: `dt >= 1.0e-4`.
- MPI ADM black-hole beam minimum parsed timestep: `dt >= 1.0e-4`.

## Current Numeric Baselines

Linear-wave last-three-point convergence rates:

| mode | RMS | rho | ux | Pgas | Rtt | Rtx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CKS | 2.071 | 1.849 | 2.129 | 2.111 | 2.121 | 2.247 |
| ADM | 2.070 | 1.849 | 2.129 | 2.105 | 2.121 | 2.247 |

Crossing-beams angular convergence:

| quantity | current range |
| --- | ---: |
| ADM-CKS normalized Linf difference | `3.757567e-4` at `Nang=12` to `2.443248e-5` at `Nang=362` |
| CKS relative L1 error | `1.956930e-2` to `1.504625e-2` |
| ADM relative L1 error | `1.956859e-2` to `1.504618e-2` |

Kerr photon-orbit beam:

| comparison | normalized Linf | relative L1 |
| --- | ---: | ---: |
| dyn CKS vs legacy | `7.298162e-2` | `1.595626e-2` |
| dyn ADM vs dyn CKS | `4.497598e-2` | `7.477185e-2` |

Gas-radiation equilibration:

| quantity | current value |
| --- | ---: |
| 100-step max `|Delta T|` | `1.712653e-2` |
| 100-step max `|Delta u|` | `2.568980e-2` |
| final `Tgas` | `1.216323` |
| final `Trad` | `1.214481` |
| final `utot` | `4.000000` |

ADM formal checks:

| check | current value |
| --- | ---: |
| FLRW final relative E error | `1.18618e-4` |
| FLRW final relative `sqrt(gamma) E` error | `1.18618e-4` |
| lapse-gradient correlation | `0.999307` |
| lapse-gradient relative RMS mismatch | `4.68375e-2` |
| momentum-source absolute residual, `Nx=32` | `1.80196e-5` |
| momentum-source absolute residual, `Nx=128` | `1.14669e-6` |
| momentum-source max relative residual, `Nx=128` | `8.47848e-4` |

## Performance Guard Used For This Audit

On the Intel Arc B580 SYCL/Level Zero build, the optimized
Valencia+`dyn_radiation` ADM and CKS `gr_torus` cases should stay within a
factor of two of the HARM GRMHD+legacy-radiation case for the same `64^3`,
50-step, output-disabled benchmark configuration.
