# Dynamical-Spacetime Radiation Design Notes

This note summarizes the current `src/radiation` solver and sketches a
Valencia/ADM radiation transport path for dynamical spacetimes.  The goal is the
radiation analogue of the HARM GRMHD versus `dyn_grmhd` split: keep the existing
stationary Cartesian Kerr-Schild path, and add a path that takes geometry from
the ADM variables without assuming a stationary metric or a particular gauge
driver.

## Current Solver

The current solver is a grey discrete-ordinates (`S_N`) transport scheme on a
geodesic angular grid.  Each angular bin stores a null direction in a local
orthonormal tetrad,

```text
q^(a)_n = (1, l_n, m_n, n_n),        l_n^2 + m_n^2 + n_n^2 = 1.
```

The tetrad is built analytically for Cartesian Kerr-Schild coordinates by
`ComputeTetrad()` in `radiation_tetrad.hpp`.  For each cell it computes

```text
e_(a)^mu,        e_(a)_mu = g_munu e_(a)^nu,
omega_(a)(b)(c) = Ricci rotation coefficients.
```

For the current CKS tetrad, the spatial tetrad legs have zero coordinate time
component, so the coordinate null-vector time component is independent of angle,

```text
n^mu_n = e_(a)^mu q^(a)_n,
n^0_n = e_(0)^0.
```

The stored radiation variable is

```text
U_n = n^0_n n_{0,n} I_n,
```

where `I_n` is the grey intensity measured in the tetrad frame and
`n_{0,n} = e_(a)_0 q^(a)_n`.  Flux reconstruction uses the primitive quantity

```text
n_{0,n} I_n = U_n / n^0_n.
```

The spatial transport update is

```text
partial_t U_n + partial_i F^i_n + D_a A^a_n = S_n,
F^i_n = n^i_n n_{0,n} I_n.
```

`CalculateFluxes()` reconstructs `n_0 I` in space and upwinds with the face
coordinate component `n^i`.  `RKUpdate()` subtracts the spatial flux divergence
and, if enabled, the angular flux divergence.

Angular fluxes describe geodesic bending on the momentum sphere.  The code
precomputes edge angular velocities `na(m,n,k,j,i,nb)` from the Ricci rotation
coefficients.  Schematically,

```text
d q^(a)/d lambda =
  -[omega^(a)_(b)(c) - q^(a) omega^(0)_(b)(c)] q^(b) q^(c),
```

then projects this drift onto each edge of the geodesic angular cell.  This is
performed in `SetOrthonormalTetrad()`, and therefore only happens once for the
static CKS metric.

The implicit radiation-fluid coupling in `RadFluidCoupling()` is local.  It:

1. converts the fluid conserved state back to primitives,
2. computes opacities,
3. transforms the fluid velocity into the radiation tetrad,
4. uses Lorentz invariance of grey intensity to get comoving intensities,
5. solves a local implicit absorption/emission/scattering update,
6. optionally applies Comptonization,
7. adds the lost radiation coordinate moments back to the fluid.

The key transformations are

```text
n^0_cm = u_(a) q^(a),
d Omega_cm = d Omega_tet / (n^0_cm)^2,
I_cm = I_tet (n^0_cm)^4.
```

The present implementation computes the coordinate radiation moments directly
from `U_n` and `n_mu/n_0` and applies `m_old - m_new` to the HARM-style fluid
conserved variables.

## Why This Is Not Enough For Dynamical Spacetime

The current solver assumes:

1. The metric is analytic Cartesian Kerr-Schild and stationary.
2. Tetrads, face direction maps, and angular drift coefficients can be built
   once in the constructor.
3. `n_0` is effectively the photon Killing energy associated with stationarity.
4. The coupling code uses HARM primitive/conserved conventions.

For a dynamical ADM spacetime, `n_0` is not conserved and the tetrad/connection
changes every RK stage.  Rebuilding a full analytic tetrad and Ricci-rotation
tensor per angle would be too expensive and, more importantly, would still not
give a Valencia source structure.

## ADM/Valencia Radiation Moments

Let the ADM fields be

```text
ds^2 = -alpha^2 dt^2 + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt),
n^mu = (1/alpha, -beta^i/alpha).
```

Use an Eulerian orthonormal tetrad:

```text
e_(0)^mu = n^mu,
e_(a)^mu = (0, E_(a)^i),
gamma_ij E_(a)^i E_(b)^j = delta_ab.
```

For direction `l^(a)` on the local unit sphere,

```text
s^i = E_(a)^i l^(a),
k^mu = epsilon (n^mu + s^mu),
dx^i/dt = alpha s^i - beta^i.
```

The radiation stress-energy tensor decomposes as

```text
R^munu = E n^mu n^nu + F^mu n^nu + F^nu n^mu + P^munu,
E      = int I dOmega,
F^i    = int I s^i dOmega,
P^ij   = int I s^i s^j dOmega.
```

The Valencia moment equations, ignoring matter coupling, are

```text
partial_t(sqrt(gamma) E)
  + partial_i[sqrt(gamma) (alpha F^i - beta^i E)]
  = sqrt(gamma) (alpha P^ij K_ij - F^i partial_i alpha),

partial_t(sqrt(gamma) S_j)
  + partial_i[sqrt(gamma) (alpha P^i_j - beta^i S_j)]
  = sqrt(gamma) [
      0.5 alpha P^ik partial_j gamma_ik
    + S_i partial_j beta^i
    - E partial_j alpha ].
```

Matter coupling adds `-alpha sqrt(gamma) G_n` to the radiation energy equation
and `+alpha sqrt(gamma) G_j` to the radiation momentum equation if `G_mu` is the
four-force density exerted on the fluid; flip signs if the opposite convention
is used.  These equations are the radiation counterpart of
`dyn_grmhd::AddCoordTermsEOS()`: all geometric terms use ADM fields, spatial
derivatives, and `K_ij`.  No explicit gauge-driver time derivatives are needed.
The normal-time metric evolution is represented by `K_ij`.

## Angle-Resolved ADM Transport

A minimal `S_N` ADM update can evolve

```text
U_n = sqrt(gamma) w_n I_n,
F^i_n = sqrt(gamma) w_n (alpha s_n^i - beta^i) I_n.
```

The per-angle energy-shift source should be chosen so the angular sum exactly
recovers the Valencia energy source:

```text
S^E_n = sqrt(gamma) w_n I_n kappa_n,
kappa_n = alpha K_ij s_n^i s_n^j - s_n^i partial_i alpha.
```

Then

```text
sum_n S^E_n = sqrt(gamma) (alpha P^ij K_ij - F^i partial_i alpha).
```

Angular advection updates the local direction `l^(a)`.  The cleanest route is to
use the Eulerian tetrad connection:

```text
d l^(a)/dt =
  -alpha [omega^(a)_(b)(c)
        - l^(a) omega^(0)_(b)(c)] q^(b) q^(c),
q^(a) = (1, l^(a)).
```

For the Eulerian tetrad, the connection coefficients are built from ADM data:

```text
a_i = D_i ln alpha,
omega_(0)(a)(0) = a_(a),
omega_(a)(0)(b) = -K_(a)(b),
omega_(a)(b)(c) = 3D spin connection of E_(a)^i.
```

This uses `alpha`, `gamma_ij`, `K_ij`, and spatial derivatives of `alpha` and
`gamma_ij`.  Shift appears in the spatial coordinate flux
`alpha s^i - beta^i` and in the Valencia momentum source through
`partial_j beta^i`, not as a gauge-driver time derivative.

The angular drift must be discretized so angular sums reproduce the Valencia
momentum source to truncation error.  A useful invariant test is:

```text
sum_n l_j [angular_flux_divergence_n + S^E_n]
```

should reproduce the geometric momentum source in the moment equation above.

## Efficient Geometry Algorithm

Do not rebuild a full 4x4 tetrad and 4x4x4 connection per angle.  Use an
ADM radiation-geometry cache updated once per RK stage.

Cell-centered cache:

```text
sqrt_gamma,
gamma_ij, gamma^ij,
alpha, beta^i,
E_(a)^i and E^(a)_i,
K_(a)(b),
a_(a) = E_(a)^i partial_i ln alpha,
3D spin connection coefficients omega_(a)(b)(c).
```

Face cache:

```text
alpha_f,
beta^dir_f,
E_(a)^dir_f
```

For spatial fluxes, each angle only needs

```text
v^dir_n = alpha_f E_(a)^dir_f l_n^(a) - beta^dir_f.
```

This matches the structure of the existing `tet_d1_x1f`, `tet_d2_x2f`,
`tet_d3_x3f` arrays, but replaces analytic CKS columns with ADM face geometry.

For angular fluxes, compute connection coefficients once per cell and loop over
all angular neighbors inside that cell.  This is `O(N_cell N_angle)` for the
necessary angular drift, but avoids `O(N_cell N_angle)` tetrad construction and
avoids repeated metric derivative work.

Possible optimization levels:

1. First implementation: compute cell triad and connection in the angular-flux
   kernel, then loop over all angles.  This is simple and device-safe.
2. Cached implementation: store the cell connection cache above, then angular
   flux kernels only evaluate contractions with `q^(a)`.
3. Hybrid: cache only `E_(a)^i`, `K_(a)(b)`, and `a_(a)`; recompute the 3D spin
   connection on demand.  This saves memory but costs more flops.

The right default is probably level 1 for correctness, then level 2 for
production BBH radiation.

## Implementation Sketch

1. Add a radiation geometry mode:

```ini
<radiation>
geometry = cks   # default, current behavior
geometry = adm   # requires <adm> / dynamical relativity
```

`auto` could select `adm` when `pcoord->is_dynamical_relativistic`.

2. Split geometry setup:

```text
SetOrthonormalTetradCKS()   # current code path
UpdateRadiationADMGeometry()
```

The ADM path must run at initialization and before every radiation flux/source
stage after the ADM variables have been filled or advanced.

3. Add a task before `CalculateFluxes()`:

```text
rad_geom = Radiation::UpdateGeometry
rad_flux depends on rad_geom
```

For stationary CKS, `UpdateGeometry` is a no-op after construction.  For ADM,
it rebuilds face geometry, center triads, and angular drifts.

4. Modify fluxes behind the geometry interface:

```text
cks: current n^i = tet_d?_x?f dot q, U = n^0 n_0 I
adm: v^i = alpha s^i - beta^i, U = sqrt(gamma) w I
```

This suggests an internal `RadiationGeometry` helper rather than sprinkling
`if (adm)` throughout every kernel.

5. Add ADM coupling branch:

The current `RadFluidCoupling()` assumes HARM primitives and CKS metric calls.
For dyngr it should:

```text
read ADM geometry from pmbp->padm->adm,
read dyngr primitive pressure from IPR, not IEN,
construct fluid W and Eulerian velocity using gamma_ij,
transform to the Eulerian radiation tetrad,
update intensities with the existing implicit local solve,
compute radiation moment changes:
  Delta E = sum_n w_n Delta I_n,
  Delta S_i = sum_n w_n Delta I_n s_i,
apply to fluid:
  u0(IEN) += -sqrt(gamma) Delta E,
  u0(IMi) += -sqrt(gamma) Delta S_i,
with sign matched to the four-force convention.
```

6. Use moment diagnostics from the angular solution:

```text
E_rad, F_rad^i, P_rad^ij, S_rad_i
```

These are needed both for tests and for eventual radiation stress-energy
coupling to Z4c.

7. Time step:

The radiation CFL speed in ADM mode is

```text
max_n |alpha E_(a)^dir l_n^(a) - beta^dir|.
```

The angular CFL uses the freshly updated angular drift.  Unlike the current
comment in `radiation_newdt.cpp`, this must be recomputed whenever the metric
changes.

## Test Plan

1. Flat spacetime: ADM mode with `alpha=1`, `beta=0`, `gamma_ij=delta_ij`,
   `K_ij=0` must reproduce current radiation beam, hohlraum, relaxation, and
   linear-wave tests.
2. Stationary Kerr-Schild: initialize ADM from analytic KS and compare ADM mode
   against the existing CKS mode for beam bending and torus radiation over a
   fixed background.
3. Gauge-wave metric, no matter coupling: verify that the Valencia moment sums
   are stable and that no source proportional to gauge-driver time derivatives
   appears.
4. Expanding homogeneous metric: with `K_ij = -H gamma_ij`, check the expected
   redshift of Eulerian radiation energy.
5. Coupled optically thick diffusion: compare dyngr radiation coupling against
   current static-metric diffusion in the stationary limit.
6. BBH smoke test: run with `geometry=adm`, angular fluxes on, radiation
   feedback off first; then enable coupling and compare global radiation energy
   plus fluid energy exchange.

