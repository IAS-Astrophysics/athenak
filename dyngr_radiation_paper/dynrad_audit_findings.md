# `src/dyn_radiation/` audit — findings since `d3c6ee0a`

Status note: this file is retained as an audit trail.  The current solver has
addressed two items that were previously called out here: the ADM geometric
metric source is now applied with a closed-form exponential stage increment,
and `gm1` is initialized before being captured by the source-coupling device
lambda.  Fixed-fluid radiation coupling is also guarded: `fixed_fluid=true`
defaults to `affect_fluid=false`, and the contradictory explicit setting
`fixed_fluid=true, affect_fluid=true` is rejected at startup.

This report enumerates concrete bugs, latent bugs, and stability/portability concerns introduced (or left in place) in the dynamical-spacetime radiation module added on top of commit `d3c6ee0a20e3a115bb89be9dde6a53ecd47baac1`. References are file:line. Severity: **CRITICAL** = produces wrong answers or crashes; **HIGH** = wrong answers in plausible regimes / silent correctness loss; **MEDIUM** = stability or precision degradation; **LOW** = code quality / future-proofing.

---

## 1. Math & equation correctness

### 1.1 Densitized intensity / Valencia source structure — OK
`dyn_radiation_update.cpp:69–108`. The ADM update solves
`∂_t U + ∂_i F^i = U(α K_{ij} s^i s^j − s^i ∂_i α)` with `U = √γ I_Eul`.
The kernel uses the *current-stage* `i_stage = i0_(m,n,k,j,i)` for the source, which is the correct Strang-style placement consistent with the paper Eqs. 311–349. The face-divergence is `(F_{i+1/2} − F_{i−1/2})/Δx` with `F = √γ (α s^i − β^i) I` baked into `tet_d1_x1f * sqrt_detg_x1f` × upwinded primitive `I = U/√γ`. This is consistent.

### 1.2 Cholesky cotriad time-derivative — OK, but fragile
`dyn_radiation_tetrad.cpp:90–121, 422–460`. `dgdt[a][b] = −2 α K_{ab} + ∇_a β_b + ∇_b β_a` is built using `grad_beta_d` and Christoffel-corrected covariant derivatives (`gamma_cab` lowered with `g_uu`). The Cholesky derivative formulas (`dl00 = ½ dgxx/l00`, `dl11 = (½ dgyy − l10 dl10)/l11`, …) match the paper Eqs. 449–486 line-for-line. Fragile points:
- **Floor `adm_metric_floor = 1.0e-30`**. Any cell whose `gxx`, `gyy − l10²`, or `gzz − l20² − l21²` underflows to zero produces `l00 = 1e-15`, then `dl00 = 0.5 dgxx / 1e-15`. The triad and its derivatives become numerically meaningless but not NaN — these cells will silently inject huge spurious source terms unless excised. Suggestion: tighten the floor (e.g., `1e-12`) and either zero the affected cells' source or assert in DEBUG.

### 1.3 Geodesic angular drift — math correct, two roundoff guards required
`dyn_radiation_tetrad.cpp:178–237`.
- `pdot[i] = −∂_iα + p_j ∂_iβ^j − ½ α p_j p_k ∂_iγ^{jk}` (paper Eq. 397).
- `frame_adv = ∑_b (dt_cotriad[b][i] + v^d ∂_d cotriad[b][i]) ell_b` (paper Eq. 433).
- The radial-drift cancellation `elldot -= (ell·elldot) ell` (line 228–231) is essential to keep ℓ on S². Confirmed.
- `theta_dot = −elldot[2]/sin(θ)` and `sin²ψ_dot = ell[0] elldot[1] − ell[1] elldot[0]` follow paper Eq. 502.
- **One concern**: `sin2 = fmax(1 − ell[2]², 1e-300)`. At a pole (ℓ = ±ẑ), `theta_dot` is ill-defined and the floor merely converts NaN into a huge number. The `unit_flux` weights at the pole edges should already be zero (geodesic mesh property), so `theta_dot * unit_flux[0]` should be 0×∞ = NaN unless `unit_flux[0]=0` is honoured. **Recommendation**: skip explicitly when the polar pentagon edge is touched by setting `theta_dot = 0` if `sin2 < 1e-12`.

### 1.4 Momentum coupling index — OK
`dyn_radiation_source.cpp:283–296, 421–434`. ADM branch sets `mom[a] = γ_{ab} s^b = s_a` (lower spatial index). Then `m_old[1..3] += s_a U Ω`, which integrates to `√γ F_{rad,a}` — the correct Eulerian 3-momentum density to add/subtract from the conserved fluid momentum `S_i` in dyngrmhd.

### 1.5 Comoving-frame transform — OK
`dyn_radiation_source.cpp:240–252, 327–336`. `n0_cm = u^{(0)} − u^{(i)} ℓ^{(i)}` is the Doppler factor between the Eulerian tetrad and the fluid rest frame. `intensity_cm = 4π·I_Eul·(n0_cm)⁴` correctly transports `I/ν³`-invariance into the rest frame. The CKS branch keeps the legacy `i0/(n0 n_0)` primitive convention, which is internally consistent.

---

## 2. Numerical algorithm / stability

### 2.1 **RESOLVED** — ADM geometric source no longer uses a linear multiplier
`dyn_radiation_update.cpp:103` now applies the local metric source with
`i_stage*(exp(beta_dt*geom) - 1)`, where
`geom = α K_{ab} s^a s^b − s^a ∂_a α`.  For the pure source equation this is
the exact multiplicative update.  The geometric-source timestep estimate in
`dyn_radiation_newdt.cpp:181–202` remains useful as a limiter on large
single-stage amplification factors, but it is no longer compensating for a
forward-Euler positivity defect.

### 2.2 **RESOLVED** — Conservative angular positivity limiter replaces per-bin clipping
`dyn_radiation_update.cpp` and `dyn_radiation_source.cpp` now call a conservative
angular redistribution helper instead of independently applying `fmax(.,0)` to
each angular bin.  In ADM mode the limiter acts on `U=sqrt(gamma) I`; in CKS
compatibility mode it acts on the primitive legacy intensity `I=U/(k^0 k_0)` so
the stationary tetrad sign convention is handled correctly.  Negative angular
bins are zeroed and positive bins in the same cell are rescaled to preserve the
local angular zeroth moment whenever the pre-limited angular integral is
nonnegative.  `inputs/tests/dynrad_positivity_floor.athinput` checks this path
in both CKS mode and ADM mode through the stress-suite geometry override.

### 2.3 **RESOLVED** — Source step now iterates nonlinear opacity/temperature coupling
`dyn_radiation_source.cpp` iterates the local gas-temperature solve up to
`dyn_radiation/source_max_iter` times, using `source_tolerance` for convergence.
Each iteration rebuilds the opacity-weighted angular coefficients before solving
the quartic temperature equation.  Temperature-dependent opacity tests use mild
under-relaxation to avoid the known fixed-point two-cycle for Kramers-like
opacities.  `inputs/tests/dynrad_source_iteration.athinput` exercises this path.

### 2.4 **RESOLVED** — `FourthPolyRoot` replaced by monotone Newton-bisection solve
`dyn_radiation_source.cpp` now solves `coef4*x^4 + x + tconst = 0` with a
bracketed Newton-bisection method.  For `coef4 >= 0` the polynomial is monotone
on `x >= 0`, so the new solver avoids the closed-form cancellation branch that
rejected valid weak-coupling roots.

### 2.5 **LOW** — Compton block reuses partially-updated `tgas`
`dyn_radiation_source.cpp:367–502`. The Compton step uses `tgas = tgasnew` from the absorption/emission solve. If the absorption step set `badcell = true`, `tgasnew == tgas` (line 270), so Compton operates on the pre-coupling temperature. This is at least self-consistent. But the moment-tally arrays `m_old/m_new` are reset between the two blocks — a Compton-only update doesn't double-count momentum because the previous block already pushed `(m_old − m_new)` into the fluid. Confirmed. No fix needed.

### 2.6 **MEDIUM** — `n_0_floor` excision logic differs by branch
`dyn_radiation_update.cpp:141`, `dyn_radiation_source.cpp:351, 471, 497`: in the **ADM** branch `n_0_floor` is *not* applied (`!use_adm_geometry_` guards each test). The justification is that ADM's `n_0` is replaced by `1/α` which is well-behaved away from the puncture; inside the excision mask the `rad_mask_` test handles it. This is fine for ADM, but check that the excision mask is conservatively chosen — without the n_0 backup, a thin shell just outside the mask where `α → 0` gets no protection.

### 2.7 `cmax3` initialization in 2D ADM — OK
`dyn_radiation_newdt.cpp:120–121`. `cmax3 = tiny` for 2D in ADM mode. `dt3 = dx3 / tiny = HUGE`, then dt3 isn't used in `dtnew` (line 209 guards with `three_d`). No bug.

---

## 3. GPU / Kokkos compatibility

### 3.1 **HIGH** — Reference-binding to host scalars before lambda capture
`dyn_radiation_update.cpp:42–63`, `dyn_radiation_source.cpp:60–114`, `dyn_radiation_fluxes.cpp:43`, `dyn_radiation_newdt.cpp:60–72`. Many idioms like
```
bool &fixed_fluid_ = fixed_fluid;
Real &n_0_floor_ = n_0_floor;
auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
```
declare *references* to data that lives on the host (member fields of host classes). The KOKKOS_LAMBDA captures `[=]`. Per the C++ standard, capture-by-copy of a name bound to a reference captures the **referent's value**, so the `bool`/`Real` is captured by value and copied into the device lambda closure — this is **safe**.

The risky pattern is when the right-hand side is an *array view* through a host member: `auto &rad_mask_ = pmy_pack->pcoord->excision_floor`. Here `rad_mask_` is a reference to a `DvceArray4D<bool>` (Kokkos View) stored on the host class. The lambda copies the View by value — Views are reference-counted device handles, so this is also fine. Confirmed by direct inspection of `rad_mask_(m,k,j,i)` working on device.

**However**: the current code uses both forms inconsistently within the same TU, which is brittle. Recommend always using `auto x_ = obj->member` (copy) rather than `auto &x_ = obj->member`. There is one unambiguously dangerous case below.

### 3.2 **RESOLVED** — `gm1` is initialized before device capture
`dyn_radiation_source.cpp:93–98, 260–261`:
```
Real gm1 = 0.0;
if (is_hydro_enabled_) { gm1 = … - 1.0; }
else if (is_mhd_enabled_) { gm1 = … - 1.0; }
// captured into KOKKOS_LAMBDA, used in source coefficients
```
This removes the previous latent uninitialized capture if future refactors let
the source path run without a hydro/MHD EOS.

### 3.3 **MEDIUM** — `nh_f.h_view(n,5,...) = FLT_MAX` sentinel may bleed into device computations
`dyn_radiation_tetrad.cpp:276–282`. For pentagonal angular cells (`num_neighbors == 5`), the unused 6th edge slot is filled with `FLT_MAX`. Both the CKS `na` kernel (lines 710–727) and the ADM `na_` kernel (lines 519–530) loop only up to `num_neighbors_.d_view(n)`, so they don't touch slot 5 — confirmed safe. **However**, the angular-flux divergence in `dyn_radiation_fluxes.cpp:265–283` and the angular-CFL in `dyn_radiation_newdt.cpp:95–113` also loop only to `numn.d_view(n)`. So the sentinel never escapes. This is fine for now; just keep it in mind if anyone refactors `nh_f` to a fixed-shape iteration.

### 3.4 **HIGH** — Stencil read of grad-cache uses the *same view that was just written* in the previous par_for, with no `Kokkos::fence()`
`dyn_radiation_tetrad.cpp:300–392, 463–530`. The kernel `dynrad_adm_tet_c` writes `adm_alpha_c_, adm_g_dd_c_, adm_g_uu_c_, adm_cotriad_c_`. The next kernel `dynrad_adm_grad_cache` reads them with stencils `(m,k,j,i±1)`. AthenaK's `par_for` enqueues onto `DevExeSpace()` (default execution space, single stream on CUDA/HIP). On a single-stream backend, kernels execute in order, so the read-after-write hazard is satisfied implicitly. **On hosts using OpenMP `parallel_for` with task semantics**, or if anyone changes `par_for` to launch on a non-default stream, this hazard becomes real. Defensive `Kokkos::fence()` between the two `par_for`s — or replacing them with a single `par_for` that recomputes everything per thread — would be safer.

The same observation applies to `dynrad_adm_grad_cache → dynrad_adm_norm_to_tet → dynrad_adm_na`, all of which read previously-written ADM cache arrays. Currently safe by single-stream semantics.

### 3.5 **MEDIUM** — One-sided differences at MeshBlock boundaries break stencil consistency between MPI ranks
`dyn_radiation_tetrad.cpp:351–392`. The gradient cache uses
```
int im[3] = {(i > 0) ? i-1 : i, j, k};
int ip[3] = {(i < n1-1) ? i+1 : i, j, k};
... inv_dx[0] = 1/(dx1 * (ip[0]==im[0] ? 1 : ip[0]-im[0]))
```
At the **outermost** ghost cell on each side, the stencil collapses to one-sided, halving the effective spacing. In a multi-MeshBlock run, two MeshBlocks that share a boundary **disagree** on the gradient at their respective outermost ghost cells, because each one-sided differencing uses different points. This is OK *inside* the gradient cache — each MB only ever uses its own version of the cache.

But the **angular drift `na`** is computed from this gradient cache and then used in flux computations *across* boundaries: when MB A computes the flux on its right face using its own `na`, and MB B computes the flux on its left face (which is A's right face) using its own `na`, the two values can differ by O(Δx) because A used a one-sided gradient (interior cell `i = is`) while B used a one-sided gradient (ghost cell). This breaks **flux conservation** at AMR fine/coarse boundaries (since the flux-correction step assumes both sides agree to truncation order).

The conservative fix is to ensure ADM ghost zones are populated to enough depth that **two** layers of ghost cells exist on each side, then use a centered stencil at every ghost cell that is one cell from the outermost. Z4cToADM already populates `[isg, ieg]`; verify `ng ≥ 2` (true by default). With `ng ≥ 2`, the gradient at `i = is` (active interior) uses `(is−1, is+1)` — both valid — and disagrees with neighbor at `i = is+ng−1` (ghost) which uses one-sided. So the **interior-active gradients agree across MBs** and only the outermost-ghost gradients differ. The angular drift `na` at face `is` (i.e., at A's left boundary) in MB A uses **interior** gradients on both sides — fine. At face `is`, MB A and the upstream neighbor MB ag-ree because both compute `na` at the cell whose gradient is centered. **Verdict**: OK as long as `ng ≥ 2` and the actual face flux uses cell-centered (not face-centered) `na`. Confirmed at `dyn_radiation_fluxes.cpp:262` (`par_for("rflux_angular", ..., ks, ke, js, je, is, ie)` — strictly active region).

But: `dynrad_adm_x1f` etc. (lines 535–574) recompute `tet_d1_x1f`, `sqrt_detg_x1f` *on faces* via `Face1Metric`, which uses cell-centered ADM data on the two cells flanking the face. At the boundary face `i = is` (left edge of active region), this reads `adm_.g_dd(m, ..., is−1)` — a ghost cell value. **If ADM ghost zones were updated by Z4cToADM, this is the same as the neighbor MB's value at `i = ie+1` (its rightmost ghost cell)**. Consistency depends on Z4cToADM running over `[isg, ieg]` *and* the Z4c MPI ghost-zone exchange having completed. The task graph (z4c_tasks.cpp:63–70) does Z4c_RecvU → Z4c_BCS → Z4c_Prolong → Z4c_AlgC → Z4c_Z4c2ADM, so by the time Z4c2ADM runs, ghost zones are filled. ✓

### 3.6 **LOW** — Kokkos::Min reduction with early return in parallel_reduce
`dyn_radiation_newdt.cpp:84–86`. Returning from the lambda before writing `min_dt1, min_dt2, min_dt3, min_dta` leaves them at the reducer's identity (`+inf`), which is the correct semantics. Confirmed.

---

## 4. MPI / AMR compatibility

### 4.1 **HIGH** — `PrepareGeometryTask` runs at the start of every stage but is **not** in the `before_stagen` list
`dyn_radiation_tasks.cpp:139, 178`. The geometry refresh sits inside `stagen` with dependency `Rad_CopyI` only. There is **no** dependency `Z4c_Z4c2ADM`. In the NumericalRelativity dispatcher (`AssembleNumericalRelativityTasks` does topological sort), `Rad_PrepareGeom` will be scheduled as soon as `Rad_CopyI` is done — typically before `Z4c_Z4c2ADM` of the same stage but after `Z4c_Z4c2ADM` of the previous stage.

Per stage, this means the radiation tetrad/grad cache is rebuilt from the **previous-stage Z4c2ADM result**. The Z4c update of the same stage hasn't happened yet. So radiation evolves with stage-locked ADM data corresponding to the time of `i1` (the begin-of-stage state). For SSP-RK this matches the Z4c CalcRHS, which also uses begin-of-stage ADM for its source terms (because Z4c CalcRHS is called before Z4c2ADM). **Verdict**: OK — both physics see the same ADM time slice each stage.

But this places a **strong correctness requirement** that future contributors might break: nothing in the source code documents it. Add a comment near `Rad_PrepareGeom` queueing.

### 4.2 **CRITICAL** — `Rad_Newdt` depends on `Z4c_Z4c2ADM` *of the same stage* but its `PrepareADMGeometry()` call clobbers the cache used by **next stage's flux computation**
`dyn_radiation_newdt.cpp:34–37`:
```
TaskStatus DynRadiation::NewTimeStep(...) {
  if (use_adm_geometry) { PrepareADMGeometry(); }
  ...
}
```
And `dyn_radiation_tasks.cpp:203–204`:
```
pnr->QueueTask(&DynRadiation::NewTimeStep, this, Rad_Newdt, ...,
               {Rad_Prolong}, {Z4c_Z4c2ADM});
```
The optional `Z4c_Z4c2ADM` dependency makes `Rad_Newdt` wait until *this stage's* Z4c2ADM is done. So PrepareADMGeometry inside Newdt rebuilds `tet_c, tet_d1_x1f, sqrt_detg_*, na, adm_*_c` from the **end-of-stage** ADM data.

Now, the *next stage's* `Rad_PrepareGeom` will rebuild the same arrays from begin-of-next-stage = end-of-this-stage ADM data — i.e., the same data Newdt just used. So functionally, the next stage's PrepareADMGeometry is a redundant rebuild. Wasteful but not wrong.

**However**, the redundant rebuild also means `na_` (the angular drift) is computed using `dt_cotriad`, which depends on `dgdt = -2αK + 2D_(aβ_b)`. This `dgdt` is the **time derivative of γ_ij at the end of the stage**. In a true RK integrator, the time derivative is stage-dependent. Newdt's `dgdt` is computed at end-of-stage, but the *next* stage's PrepareADMGeometry uses the same end-of-stage data and recomputes the same `dgdt`. They agree — but neither one is the "right" `dgdt` for the next stage's flux midpoint. This is a known limitation: `dt_cotriad` is a frozen-coefficient approximation that Hwang the paper acknowledges (Eq. 471 footnote). No bug, but a documentation gap.

The **CRITICAL** part: if anyone moves `Rad_Newdt` to before stage's Z4c2ADM (e.g., to remove the Z4c dependency), `PrepareADMGeometry` inside Newdt would clobber the in-stage cache with end-of-previous-stage data. This pattern is brittle. **Recommendation**: either (a) drop the `PrepareADMGeometry()` call from `NewTimeStep` (the cache is already fresh from `Rad_PrepareGeom`); or (b) add a short comment block explaining why both calls are needed and what invariant the dependency chain protects.

### 4.3 **HIGH** — AMR re-tetrad-only path is incomplete for non-Z4c ADM runs
`mesh_refinement.cpp:640–653`:
```
if (pz4c != nullptr)      pz4c->Z4cToADM(pm->pmb_pack);
else if (padm != nullptr) padm->SetADMVariables(pm->pmb_pack);
if (pdynrad != nullptr)   pdynrad->SetOrthonormalTetrad();
```
For Z4c runs: `Z4cToADM` populates ADM over `[isg, ieg]`. ✓
For ADM-only runs (analytic spacetime): `padm->SetADMVariables` populates ADM. ✓
For dyn_radiation: `SetOrthonormalTetrad` rebuilds tetrad and the grad cache by reading `padm->adm.*` directly — fine on new MBs created by AMR.

**Concern**: After AMR, the **MPI ghost-zone exchange of ADM** has not yet been triggered for the new mesh layout. If a new MB is on a different rank than its old neighbors, its ghost-zone ADM data is stale (or zero). The grad cache built immediately from this stale data is wrong at the outermost ghost cells. The next stage's `Rad_PrepareGeom` will call `SetOrthonormalTetrad` again, but only after Z4c (or analytic ADM) has done its own ghost-zone exchange. So one stage of "wrong-ghost-zone" tetrad data exists.

Z4c, MHD, etc. don't suffer from this because they use Z4c boundary conditions / Sommerfeld BC at the active-boundary first, then re-derive any cache. Radiation uses the cache directly in the very next flux computation.

**Recommendation**: After AMR, defer the `pdynrad->SetOrthonormalTetrad()` call until after the first MPI ghost-zone exchange of Z4c/ADM, or accept one stage of degraded accuracy at AMR-newly-refined boundaries.

### 4.4 Restart I/O — OK
`outputs/restart.cpp:71–120, 159–174, 588–600`. The restart writer dumps `pdynrad->i0` over `[nmb, nrad, nout3, nout2, nout1]` including ghost zones. The reader allocates `outarray_rad` and pushes back into `pdynrad->i0`. **None of the cached tetrad/ADM arrays are checkpointed** — they are rebuilt by the constructor's `PrepareADMGeometry()` call (`dyn_radiation.cpp:195`). For this to work correctly, the constructor must run *after* ADM/Z4c data is loaded.

Trace through `meshblock_pack.cpp:209–255`:
1. `pz4c = new z4c::Z4c(...)` — allocates Z4c.
2. `padm = new adm::ADM(...)` — allocates ADM.
3. `pdynrad = new dyn_radiation::DynRadiation(...)` — runs `PrepareADMGeometry()` at line 195.

But ADM data isn't loaded from the restart file at this point — that happens later in `Driver::Initialize` via `RestartOutput::LoadOutputData` ... wait, restart is read in the problem generator. The constructor's `PrepareADMGeometry()` will operate on **uninitialized ADM data** (zero-filled allocations), which yields a degenerate Cholesky tetrad and meaningless gradient cache. *This is rebuilt by the first stage's `Rad_PrepareGeom`*, so by the time data is consumed, it's correct. **No bug**, but the constructor-time call is wasted work and could mask bugs by populating arrays with garbage that *looks* valid.

### 4.5 Load balancing — OK
`mesh/load_balance.cpp:149–156, 406–413, 540–561, 826–851`. `pdynrad->i0` and `pdynrad->coarse_i0` are packed/unpacked through the standard CC AMR machinery exactly as `prad->i0`. The `else if (pdynrad != nullptr)` chain explicitly forbids both `prad` and `pdynrad` simultaneously, which is enforced again by `meshblock_pack.cpp:173–186`. ✓

### 4.6 Boundary buffers — OK
`dyn_radiation.cpp:219`: `pbval_i = new MeshBoundaryValuesCC(ppack, pin, false)` followed by `InitializeBuffers(prgeo->nangles)`. The Send/Recv/Init/Clear pattern in `dyn_radiation_tasks.cpp:212–423` matches the legacy radiation module. The `false` argument disables coarse-buffer flux correction... actually, looking at `MeshBoundaryValuesCC` constructor — confirm by reading. Best practice is to verify that flag matches what `legacy radiation` passes. Looking at `radiation/radiation.cpp` if it exists:

Verified that the same pattern is used in legacy `radiation/`. ✓

### 4.7 **MEDIUM** — `coarse_i0` dimensions silently relied on `cnx*` matching `nx*/2`
`dyn_radiation.cpp:210–216` allocates `coarse_i0` only if `multilevel`. Standard. No bug.

---

## 5. Cross-cutting / minor

### 5.1 **LOW** — Potential operator-precedence ambiguity in legacy CKS angular drift (pre-existing, copied unchanged)
`dyn_radiation_tetrad.cpp:723`:
```
na_(m,n,k,j,i,nb) = iszetaf*na1*uflux.d_view(n,nb,0)+na2*uflux.d_view(n,nb,1);
```
Reads as `(iszetaf*na1*uflux[0]) + (na2*uflux[1])`. The `iszetaf = 1/sin(θ)` factor multiplies *only* `na1` (the θ-component) and not `na2` (the ψ-component). Comparing with paper Eq. 502, this is *correct*: only `θ_dot` carries the `1/sin θ`, while `sin²ψ_dot` does not. So the un-parenthesized form is intentional — but it took 5 minutes to verify. Add parentheses for clarity:
```
na_(m,n,k,j,i,nb) = (iszetaf*na1)*uflux.d_view(n,nb,0) + na2*uflux.d_view(n,nb,1);
```

### 5.2 **LOW** — `is_compton_enabled` unused in pure-radiation runs but read on every cell anyway
`dyn_radiation_source.cpp:63, 351, 366`. Tiny perf hit on GPU; skipping the load on disabled-Compton runs is a one-line fix.

### 5.3 **LOW** — Dead `AddTmunu` task slot
`dyn_radiation_source.cpp:37–39`. `AddTmunu` returns `complete` immediately, but `Rad_SetTmunu` is declared in `numerical_relativity.hpp:95` and never queued. Either implement or drop both.

### 5.4 **LOW** — Inconsistent `cnx2 > 1` test using `nx2 > 1`
`dyn_radiation.cpp:260, 213` and similar. The convention used elsewhere in AthenaK is `multi_d`/`three_d` vs the local nx2/nx3 — both work but mix freely here. Cosmetic.

---

## 6. Summary table

| # | Severity | File:line | Brief |
|---|----------|-----------|-------|
| 2.1 | RESOLVED | `dyn_radiation_update.cpp:103` | Geometric source uses exponential stage increment |
| 2.2 | RESOLVED | `dyn_radiation_update.cpp`, `dyn_radiation_source.cpp` | Conservative angular positivity limiter |
| 2.3 | RESOLVED | `dyn_radiation_source.cpp` | Iterated nonlinear source solve |
| 2.4 | RESOLVED | `dyn_radiation_source.cpp` | Bracketed quartic root solve |
| 3.2 | RESOLVED | `dyn_radiation_source.cpp:93–98` | `gm1` initialized before device capture |
| 3.4 | HIGH | `dyn_radiation_tetrad.cpp:300–392` | Implicit single-stream assumption; no `Kokkos::fence()` |
| 3.5 | MEDIUM | `dyn_radiation_tetrad.cpp:351–392` | One-sided ghost-cell gradients; OK only if `ng ≥ 2` |
| 4.1 | HIGH | `dyn_radiation_tasks.cpp:178` | Undocumented stage-locked ADM-time invariant |
| 4.2 | CRITICAL (brittle) | `dyn_radiation_newdt.cpp:34–37` | Newdt rebuilds tetrad cache; depends on Z4c2ADM ordering |
| 4.3 | HIGH | `mesh_refinement.cpp:651–653` | AMR re-tetrad runs before MPI ADM ghost-zone exchange |

## 7. Recommended top-priority fixes

1. ~~Initialize `gm1 = 0.0` in `dyn_radiation_source.cpp:93`.~~ Done.
2. **Add `Kokkos::fence()`** between the four sequential `par_for`s in `SetOrthonormalTetrad` (or merge them).
3. **Document the stage-locked invariant** that `Rad_PrepareGeom` and `Z4c_CalcRHS` both consume the previous-stage Z4c2ADM result.
4. **Drop the `PrepareADMGeometry()` call from `NewTimeStep`** (the cache is fresh; the call is wasteful and brittle to task-graph reorderings).
5. **Tighten `adm_metric_floor`** from `1e-30` to `1e-12` and emit a one-shot warning the first time it is triggered.
6. ~~Make the `α K_{ab} s^a s^b` source term implicit in `dyn_radiation_update.cpp` via the closed-form factor `exp(Δt geom)`.~~ Done.
7. ~~Replace per-bin positivity clipping with a conservative angular limiter.~~ Done.
8. ~~Iterate the nonlinear source solve and replace the brittle closed-form quartic.~~ Done.
9. **Audit AMR re-tetrad sequence**: move `pdynrad->SetOrthonormalTetrad()` to after the first ghost-zone exchange of the next stage, or equivalently, perform it inside the next stage's `Rad_PrepareGeom` (which already happens — the AMR-time call is then redundant and safe to drop).
