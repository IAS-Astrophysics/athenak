# Cost of relaxing the Bondi-fed binary

Planning estimate only; no new evolution. These estimates apply to the current
a=32 M, M_total=1, r_B=161.227 M mesh on one Intel Max 1550 root GPU.

P_binary=2 pi a^(3/2)=1137.378 M, so 100 periods=113737.8 M.
With r_B=2/c_s,infinity^2, t_B=r_B/c_s,infinity=1447.57 M=1.273 periods.
The global clock advances during coarse stages while stored interior patches
are held. Finest-stage active time is a separate diagnostic; other gas outside
the saved patches can continue evolving during coarse stages.

The reference prescriptions in [Guo et al. (2025)](https://arxiv.org/html/2504.16802v2)
are Eq. 17 (hydro: 100 M at stage 0, skip 1/2, then 10 t_dyn), Eq. 20
(magnetized Bondi: 500 M at stage 0, skip 1/2, then min[50 t_dyn, 0.15 t_B]),
and Eq. 24 (large Bondi radius: replace 50 with 20). Here
t_dyn=R^(3/2)/sqrt(M). Their stage indices are specific to their hierarchy;
copying skipped stages to our off-center patches needs interface validation.

Our nominal radii are 32, 64, 128, 256 M, with dynamical times
181.0, 512.0, 1448.2, 4096.0 M. Only stored cells inside these radii are held,
not entire spheres. The final nominal radius exceeds r_B; a conservative
physical experiment could stop at stage 2 (128 M). The existing interface
geometry has only short infrastructure validation.

An illustrative schedule is an initial 10-20 fully active periods, then fine
visits lasting 1-3 periods with outer visits capped at
0.15 t_B=217.14 M=0.191 period. The initial duration is a proposed starting
point, not demonstrated convergence. Assess individual-hole fluxes, gas mass
near the binary, energy/angular-momentum fluxes, and independence of seed and
dwell times. The three-stage outer sweep still holds the inner patches for
0.57 period; resolving orbital variability requires shorter gaps or a continuous
reference.

The current adiabatic hydro controls have B=0 and no prescribed viscosity or
cooling. An alpha-disk thermal/viscous time is therefore not a defined target.
For a future disk with effective stress alpha_eff, the standard estimates are

    t_thermal ~ t_dyn / alpha_eff
    t_viscous ~ t_dyn / [alpha_eff (H/R)^2].

See [Pringle (1981), Sections 6-7](https://courses.physics.ucsd.edu/2020/Winter/physics116/Pringle_%20Accretion%20Disks.pdf).
At R=2a and alpha_eff=0.1, t_thermal~4.5 binary periods; t_viscous~50
periods for H/R=0.3, ~450 for H/R=0.1, and ~5000 for H/R=0.03.
A 100-period target cannot guarantee viscous relaxation of a thin disk.

## Cost model

Aurora job 8924219 recorded 154 binary steps, 219464 MeshBlock-cycles,
135.9838 seconds of evolution, and 146.835 seconds end to end.
Use k=135.9838/219464=0.00061962 seconds per block-step and
wall=sum_z k N_z tau_z/dt_z. Coarse cost per block is modeled, not separately
benchmarked. Metric caching is already included.

| Stage | Blocks | Approximate dt/M | Model wall seconds per M |
| --- | ---: | ---: | ---: |
| 0 | 1520 | 0.01451 | 64.9 |
| 1 | 1352 | 0.029 | 28.9 |
| 2 | 1184 | 0.057 | 12.9 |
| 3 | 1016 | 0.11154 | 5.64 |

These finite schedules start at stage 0 and stop at the target, including
partial cycles. Costs are per configuration on one GPU at current resolution.

| Schedule | Global orbits | Finest-stage active orbits | GPU-days |
| --- | ---: | ---: | ---: |
| Continuous finest grid | 100 | 100 | 85.5 |
| Four stages; 10 t_dyn per visit, uncapped | 100 | 3.18 | 17.7 |
| Three stages; 10 t_dyn per visit, uncapped | 100 | 7.96 | 30.9 |
| Three stages; 1 period fine, 0.15 t_B outer | 100 | 63.9 | 65.8 |
| Three stages; 3 periods fine, 0.15 t_B outer | 100 | 84.0 | 76.8 |
| Three stages; 3 periods fine, 0.15 t_B outer | 118.9 | 100 | 91.3 |

The uncapped four-stage cycle lasts 72.1 periods: a global 100-period target
completes only one return and part of the next sweep. The capped three-stage
schedules yield about 28-64 cycles per 100 global periods.

An earlier fine-grid measurement of 0.69 seconds per step instead predicts
62.6 GPU-days for continuous 100-period binary evolution. A planning range
of roughly 60-100 GPU-days per case is sensible for substantial inner
relaxation; allow factor-two uncertainty until longer stage benchmarks exist.
If additional to the target, 10-20 active startup periods cost about 6-17
GPU-days. A matched single/binary pair doubles GPU time. Multi-GPU scaling
has not been measured. These are production-scale costs; debug is suitable
for calibration.

The current rotating outer-grid bound is approximately
dt_outer=0.25*8/(1+Omega*256)=0.828 M, above the finest timestep. In a much
larger domain with fixed relative resolution, rotation gives
dt<=CFL (Delta x/R)/Omega and can impose a floor on steps per orbit.
These a=32 estimates must not be carried unchanged to a parsec-separation
horizon-resolved hierarchy.

[Machine-readable calculations](bondi_100_orbit_cost.json) and
[measured validation records](bondi_boundary_aurora_validation.json).
