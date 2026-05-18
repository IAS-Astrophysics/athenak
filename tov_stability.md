# TOV Stability Diagnostics

## Job Status

- Earlier long CUDA+MPI job `7597051`: cancelled before allocation; no data.
- Earlier no-damping GPU job `7597203`: completed to `t=5`; all cases failed without Z4c dissipation/damping.
- New damping sweep job `7634711`: completed on `della-l02g12`, `Elapsed=00:13:51`, exit code `0:0`.
- Long dissipation-only job `7637267`: completed the uniform `L=5` unboosted and boosted cases to `t=200`, then failed on the first AMR case from an oversized AMR allocation.
- AMR retry job `7637821`: completed the expanded-boundary `L=10` AMR unboosted and boosted cases to `t=200`, `Elapsed=00:20:49`, exit code `0:0`.

The new sweep used the centered Minkowski setup (`bh_mass = 0`, `coord/minkowski = true`, `lapse = 1`, `shift = 0`), `nghost = 4`, `wenoz`, `hlle`, and `user_hist = true`. All cases were run to `t = 20`.

## Diagnostic Plots

Updated damping-sweep plots:

- `plots/tov_damping_sweep_7634711/rho_max_damping_sweep.png`
- `plots/tov_damping_sweep_7634711/z4c_constraint_damping_sweep.png`
- `plots/tov_damping_sweep_7634711/rho_collapse_time_damping_sweep.png`
- `plots/tov_damping_sweep_7634711/damping_sweep_summary.csv`

Older no-damping-only plots remain in `plots/tov_stability_7597203/`.

New 200M dissipation-only plots:

- `plots/tov_200m_stability_7637267/rho_max_200m_with_boundary_contact.png`
- `plots/tov_200m_stability_7637267/rho_max_200m_log_with_boundary_contact.png`
- `plots/tov_200m_stability_7637267/z4c_constraint_200m_with_boundary_contact.png`
- `plots/tov_200m_stability_7637267/mass_drift_200m_with_boundary_contact.png`
- `plots/tov_200m_stability_7637267/tov_200m_summary.csv`

## Damping Sweep

Tested cases:

- `unboosted_L3_n48`: `L=3`, `nx=48`, 18.4 cells across star.
- `boosted_L3_n48`: same, with `v_y=0.2`.
- `unboosted_L5_n80`: `L=5`, `nx=80`, same 18.4 cells across star.
- `boosted_L5_n80`: same, with `v_y=0.2`.

Tested Z4c settings:

| label | diss | damp_kappa1 | damp_kappa2 |
|---|---:|---:|---:|
| no_diss_no_damp | 0.0 | 0.0 | 0.0 |
| diss_0p1_no_damp | 0.1 | 0.0 | 0.0 |
| diss_0p5_no_damp | 0.5 | 0.0 | 0.0 |
| diss_0p5_k1_0p02_k2_0 | 0.5 | 0.02 | 0.0 |
| diss_0p5_k1_0p02_k2_0p02 | 0.5 | 0.02 | 0.02 |

## Results

Key result: `diss = 0.5` is the important stabilizer in this test. Constraint damping with `damp_kappa1 = 0.02` did not materially improve the central-density behavior relative to `diss = 0.5` alone.

| setting | unboosted L3 | boosted L3 | unboosted L5 | boosted L5 |
|---|---|---|---|---|
| diss=0, k1=0, k2=0 | fail at t=3.0 | fail at t=2.9 | fail at t=3.0 | fail at t=2.9 |
| diss=0.1, k1=0, k2=0 | fail at t=8.8 | fail at t=8.6 | fail at t=9.9 | fail at t=9.6 |
| diss=0.5, k1=0, k2=0 | stable to t=20 | fail late, rho/rho0=0.053 | stable to t=20 | stable to t=20 |
| diss=0.5, k1=0.02, k2=0 | stable to t=20 | fail late, rho/rho0=0.053 | stable to t=20 | stable to t=20 |
| diss=0.5, k1=0.02, k2=0.02 | stable to t=20 | fail late, rho/rho0=0.053 | stable to t=20 | stable to t=20 |

For `diss = 0.5`, the Z4c constraint norm stayed finite and did not grow above its initial value in all four cases. The central-density failure in the boosted `L=3` case therefore looks more like a domain/boundary/boost interaction than an undamped constraint-growth problem. At matched resolution, moving the boundary from `L=3` to `L=5` stabilized the boosted case.

Representative final central-density ratios at `t=20`:

| setting | case | rho_max(t=20)/rho_max(0) | tail std/mean | tail drift |
|---|---|---:|---:|---:|
| diss=0.5 | unboosted_L3_n48 | 1.00083 | 3.1e-5 | 1.0e-4 |
| diss=0.5 | unboosted_L5_n80 | 1.00083 | 3.1e-5 | 1.1e-4 |
| diss=0.5 | boosted_L5_n80 | 1.00748 | 3.3e-3 | 4.2e-3 |
| diss=0.5, k1=0.02, k2=0 | boosted_L5_n80 | 1.00761 | 3.3e-3 | 4.3e-3 |

## 200M Dissipation-Only Test

This test used `diss = 0.5`, `damp_kappa1 = 0.0`, `damp_kappa2 = 0.0`, `nghost = 4`, `wenoz`, and `hlle`. The boost velocity was `v_y = 0.2`. The `L=10` runs used one AMR level beyond the `L=5` root grid, so the outer boundary moved out by a factor of 2 while the finest star resolution stayed fixed at `dx = 0.125`, or 18.4 cells across the stellar diameter. The AMR criterion used the problem generator's density-gradient refinement plus a star-region refinement guard.

The plots mark boundary timing with vertical lines:

- dotted: earliest surface-to-boundary causal contact, `(L - R)/(1 + |v|)`;
- dashed: center-to-boundary causal contact, `L/(1 + |v|)`;
- dash-dot: boosted stellar surface physically reaches the `+y` boundary, `(L - R)/|v|`.

| case | L | AMR | v_y | final rho_max/rho0 | final-third std/mean | final-third drift | rho<0.9 time | surface reaches +y boundary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unboosted_L5_n80 | 5 | no | 0.0 | 1.00325 | 1.11e-4 | 3.13e-4 | none | n/a |
| unboosted_L10_n80_amr1 | 10 | yes | 0.0 | 1.13228 | 1.55e-2 | 5.29e-2 | none | n/a |
| boosted_L5_n80 | 5 | no | 0.2 | 7.17e-10 | 1.59e-1 | -3.59e-1 | 27 | 19.238 |
| boosted_L10_n80_amr1 | 10 | yes | 0.2 | 7.02e-10 | 7.44e-2 | 1.73e-1 | 52 | 44.238 |

Interpretation:

- The unboosted `L=5` uniform run is stable by the density criterion through `t=200`; `rho_max` finishes within 0.4% of its initial value and the final-third drift is much less than 5%.
- The expanded `L=10` AMR unboosted run remains finite and does not collapse, but it settles to a higher-density state, about 13% above the initial `rho_max` by `t=200`. I would treat this as stable enough for a qualitative isolated-star check, but not as matching the stricter `<5%` production criterion.
- The boosted `L=5` and `L=10` results are dominated by the star leaving the box, not by an intrinsic pre-boundary instability. Doubling the boundary moves the `rho_max < 0.9 rho0` time from `t=27` to `t=52`, consistent with the physical surface-boundary times moving from `t=19.238` to `t=44.238`.
- For a `v_y = 0.2` run to `t=200` without the stellar surface reaching the boundary, the half-width must satisfy `L > v t + R`, so `L` should be larger than about `41.2M`. If the center must also remain far from the boundary, use a larger margin or a star-centered/comoving box.

## Boosted-Star Resolution Requirement

The current boosted tests should be interpreted as a lower-bound resolution check, not as a full `t=200` boosted-star convergence study. At `v_y = 0.2`, the Lorentz factor is only `gamma = 1.0206`, so contraction along the boost direction changes the effective diameter resolution by about 2%. The existing finest spacing `dx = 0.125` therefore gives:

- transverse diameter resolution: `2R/dx = 18.44` cells;
- boost-direction diameter resolution: `2R/(gamma dx) = 18.06` cells.

This is sufficient for the star to remain stable before boundary/box-exit effects dominate in the `L=5` and `L=10` boosted runs, so `dx = 0.125` is the coarsest spacing I would currently accept for a `v = 0.2` production preflight. For production, I would use a margin above this because a boosted star is advected across the mesh and is more sensitive to prolongation/restriction, refinement-trigger lag, and atmosphere interaction than the unboosted star.

Recommended boosted-star targets:

| target | finest dx | cells across unboosted diameter | cells across boosted-direction diameter at v=0.2 |
|---|---:|---:|---:|
| minimum tested | 0.125 | 18.4 | 18.1 |
| preferred | 0.096 | 24.0 | 23.5 |
| conservative | 0.072 | 32.0 | 31.4 |

For AMR runs, the relevant resolution is the finest spacing actually covering the star, not the root-grid spacing. The refinement criterion should keep the whole star, including the leading and trailing density gradients, on the finest level throughout the physical evolution window. In practice, require the density-gradient AMR to refine before the surface reaches a coarse-fine boundary, keep at least one buffer of refined blocks around the stellar surface, and reject a setup if `rho-max` changes when adding one extra refinement level at fixed outer boundary by more than the same `<5%` tail-drift criterion used above.

## Production Recommendation

For this simple geodesic-gauge TOV setup, use at least:

- `diss = 0.5`
- `damp_kappa1 = 0.0`
- `damp_kappa2 = 0.0`
- `nghost = 4`
- `reconstruct = wenoz`
- `rsolver = hlle`
- at least 18 cells across the boosted-direction stellar diameter as an absolute lower bound; prefer 24-32 cells across the diameter for production boosted runs
- for a boosted star with `v_y ~= 0.2`, choose the boundary from the runtime: require `L > v t_end + R` before applying any safety margin. For `t_end = 200`, this means `L > 41.2M`; `L=5` and `L=10` are only useful as short pre-boundary tests.
- use gradient-based AMR or an equivalent star-tracking refinement guard for boosted runs so the star remains on the finest level until it leaves the intended high-resolution region.

Constraint damping at `damp_kappa1 = 0.02` is safe in this short sweep, but it did not fix the unstable boosted `L=3` case or improve the stable `L=5` case. I would not count on constraint damping as the primary cure here; the KO dissipation and boundary location matter more.

Before production, rerun the isolated-star preflight with the exact production box, boost, AMR, and runtime. Accept it only if:

1. `rho-max` remains finite and does not drop below `0.9*rho-max(0)` after the initial transient.
2. Final-third `std(rho-max)/mean(rho-max) < 0.05`.
3. Final-third linear drift magnitude `< 0.05` of the mean.
4. Z4c histories contain no NaNs.
5. `C-norm2` and `H-norm2` do not grow above their initial values by more than a small factor; in the stable `diss=0.5` runs here, they actually decayed.
6. A larger-boundary repeat at matched cells across the star changes the central-density tail metrics by less than about 10%; the `L=10` AMR unboosted run did not meet this stricter threshold relative to the `L=5` uniform run.
