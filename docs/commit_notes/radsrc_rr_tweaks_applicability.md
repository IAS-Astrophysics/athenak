# R&R Radiation Tweaks Applicability

## 1. Commit summary

Commit `d3d284ef0389b0bdcd4c53e274b6ff23552b6524` adds three non-WENO changes that are relevant here.

First, the EOS/C2P path gains density-dependent entropy-floor inputs `sfloor1`, `sfloor2`, `rho1`, and `rho2`. The source commit uses log-log interpolation in density to replace the constant entropy floor in the ideal MHD and SRMHD conservative-to-primitive paths.

Second, the fixed-metric radiation solver gains default-off radiation source correction parameters: `correct_radsrc_velocity`, `correct_radsrc_opacity`, `dfloor_opacity`, `dens_trunc_max`, `tau_truncation`, and `sigmoid_residual`. These control optional source-step stabilizations and should preserve old behavior when disabled.

Third, `radiation_source.cpp` gains two optional corrections. The opacity correction computes an effective opacity density `wdn_opacity`, optionally sets it to `dfloor_opacity` in flux-excised cells, truncates it smoothly in low-density/high-magnetization regions, and rescales `sigma_a`, `sigma_s`, and `sigma_p` by `wdn_opacity / wdn`. The velocity correction activates in radiation-dominated cells, estimates a radiation-frame velocity from tetrad-frame moments, limits it by `gamma_max`, estimates gas momentum change over the source step, and changes only the local tetrad velocity used by the source solve.

All WENO-Z/WENO-MZ reconstruction changes are intentionally excluded.

## 2. Static radiation solver applicability

The EOS/C2P entropy-floor change is accepted with modifications. The source commit's repeated interpolation is device-safe but lacks validation and can take `log10` of nonpositive values or divide by zero when `rho1 == rho2`. This branch should instead validate inputs at EOS construction and use one helper. Density will be clamped to `[min(rho1,rho2), max(rho1,rho2)]` before interpolation to avoid unconstrained extrapolation outside the calibrated interval.

The static radiation parameter plumbing is accepted with modifications. The flags remain default-off, but invalid correction parameters should fail clearly instead of silently clamping `sigmoid_residual`.

The static opacity correction is accepted with safety guards. It maps directly onto the current fixed-metric source solve: `wdn` is the rest-mass density primitive, `OpacityFunction` returns density-multiplied opacities, `pmhd->bcc0` provides cell-centered magnetic fields, and `pcoord->excision_flux` exists as the flux-excision mask. If any required denominator or log argument is invalid in a cell, the opacities should be left unchanged.

When `kappa_s <= 0` or `tau_truncation <= 0`, the magnetization-based truncation scale is skipped and the correction falls back to the floor/excision density regularization only. This avoids division by zero while preserving the intended excision behavior.

The static velocity correction is accepted with modifications. It should remain a local source-solve stabilization, changing only `u_tet` used inside `RadFluidCoupling`. It must not write corrected velocities back to primitives or conserved variables.

## 3. Dynamical-spacetime radiation solver applicability

The EOS/C2P change affects the legacy ideal MHD/SRMHD C2P path used by the fixed-background hydro/MHD systems. The dynamical GRMHD primitive solver uses the separate primitive-solver framework, so no analogous EOS entropy-floor port is made there in this patch.

The dynamic radiation parameter plumbing is accepted and adapted to the `<dyn_radiation>` block. The same parameter names are used in each solver's own input block: `<radiation>` for `src/radiation` and `<dyn_radiation>` for `src/dyn_radiation`.

The dynamic opacity correction is accepted with adaptations. In CKS mode it can follow the static metric/tetrad conventions. In ADM mode the source solve uses cached lapse, shift, spatial metric, tetrads, and `i0 = sqrt(gamma) I`; therefore metric-dependent quantities must use the ADM cache for the current source stage. `excision_flux` exists through the shared coordinate object and is the correct flux-excision mask when excision is enabled.

The dynamic velocity correction is accepted only with frame-aware normalization. CKS mode uses `i0/(n0*n_0)` as in the static solver. ADM mode uses `i0/sqrt_detg` for radiation moments and keeps the existing ADM tetrad convention. The correction changes only the local `u_tet` used by the source solve and does not modify dynamic metric, Z4c, or hydrodynamic conserved variables directly.

## 4. Physical correctness checks

The opacity correction remains dimensionally consistent for constant-opacity laws because `OpacityFunction` already multiplies by density; multiplying by `wdn_opacity / wdn` replaces that density factor. For `power_opacity`, this is an approximation because it rescales the final opacity rather than recomputing all density and temperature dependent terms.

The ratio `wdn_opacity / wdn` is meaningful only when `wdn > 0`. Both solvers should guard this explicitly and skip the correction otherwise.

For MHD, `sigma_cold = b^2 / rho` should use the magnetic four-vector built from the cell-centered magnetic field and the fluid four-velocity in the solver's metric convention. In the fixed CKS solver, the source commit's construction using `glower`, `gupper`, and `bcc0` is applicable with finiteness checks. In ADM dynamic radiation, the four-metric must be reconstructed from lapse, shift, and spatial metric before lowering indices.

The dynamic solver uses the same `nh_c`, `tc`, `tt`, and `norm_to_tet` names, but their meaning depends on geometry mode. CKS mode follows the fixed-metric solver. ADM mode uses an Eulerian ADM tetrad, cached `sqrt_detg`, and `i0/sqrt_detg` intensity normalization; formulas containing `n0*n_0` must not be copied into ADM paths.

For `delta_l`, coordinate spacing is consistent with the source commit and the static solver. In ADM dynamic radiation, a cheap proper-length estimate from the diagonal spatial metric and coordinate spacings is preferred; if unavailable or non-finite, fall back to the coordinate maximum and document that the optical-depth truncation remains a stabilization heuristic.

`excision_flux` exists in both solver paths through `pcoord->excision_flux`. It marks cells near flux excision and is a better match to the source commit's opacity-density suppression than `excision_floor`, which is used to zero radiation in fully excised cells.

The velocity correction should alter only the velocity used inside the local radiation source solve. It should not feed corrected velocities into primitive or conserved fluid variables; any fluid feedback remains the existing radiation moment exchange already present in the solver.

Radiation energy and moment transformations must respect sign and normalization conventions. Static and dynamic CKS use `n0*n_0` to convert conserved intensity to physical intensity. Dynamic ADM uses `sqrt_detg`. Tetrad-frame moment signs follow the existing source terms with spatial components contracted as `n0_cm = u_tet[0] nh0 - u_tet[i] nhi`.

## 5. Decision table

| Component | Static radiation decision | Dyn radiation decision | Reason | Files to edit |
|---|---|---|---|---|
| EOS density-dependent entropy floor | Accepted with validation/helper | Affects only shared legacy EOS paths, not dyn GRMHD primitive solver | Useful floor extension; source needs guards and non-extrapolating interpolation | `src/eos/eos.cpp`, `src/eos/eos.hpp`, `src/eos/ideal_c2p_mhd.hpp` |
| Radiation correction parameters | Accepted with validation | Applied identically in `<dyn_radiation>` block | Default-off switches preserve old behavior | `src/radiation/radiation.*`, `src/dyn_radiation/dyn_radiation.*` |
| Opacity-density reduction | Accepted with guards | Adapted for ADM/CKS metrics and intensity normalization | Dimensionally consistent for constant opacities; approximate for power opacities | `src/radiation/radiation_source.cpp`, `src/dyn_radiation/dyn_radiation_source.cpp` |
| Excision opacity density | Accepted using `excision_flux` | Applied using shared `excision_flux` | Flux-excision mask exists in both paths | `src/radiation/radiation_source.cpp`, `src/dyn_radiation/dyn_radiation_source.cpp` |
| Radiation-dominated velocity correction | Accepted as local `u_tet` correction | Adapted with ADM `i0/sqrt_detg` normalization | Stabilization only; no primitive/conserved velocity feedback | `src/radiation/radiation_source.cpp`, `src/dyn_radiation/dyn_radiation_source.cpp` |
| WENO/WENO-Z/WENO-MZ changes | Rejected | Rejected | Explicitly outside requested scope | None |
