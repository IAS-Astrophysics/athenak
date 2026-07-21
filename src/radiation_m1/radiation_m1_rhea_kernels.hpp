#ifndef RADIATION_M1_RHEA_KERNELS_HPP
#define RADIATION_M1_RHEA_KERNELS_HPP
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_rhea_kernels.hpp
//! \brief Pure-Kokkos physics kernels for Rhea-based neutrino flavor mixing.
//!
//! Everything in this header is a KOKKOS_INLINE_FUNCTION free function with zero Torch
//! types and zero backend-conditional compilation (rhea_athenak_port_design.md §5.9) --
//! it operates only on plain Real/float scalars and small fixed-size C arrays, so it is
//! identically testable whether the inputs came from a real Rhea inference call or a
//! synthetic fixture. Ported faithfully from THC_M1 (thc_M1_flavor_mix.cc:255-711) and
//! from Rhea's own reference C++ helper (Rhea/cpp_interface/FFISubgridModel.h:61-86) --
//! see rhea_athenak_port_design.md §2b for the physics contract.

#include "athena.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::RestrictToPhysical
//! \brief Ensure Rhea's raw per-cell 4-current prediction is timelike (or null), in place.
//!
//! Ports Rhea/cpp_interface/FFISubgridModel.h:61-86 (`restrict_to_physical`) from its
//! batched-tensor form down to a single cell. Input/output layout matches Rhea's raw
//! `F4_out` (design doc §2a, *before* the mu/tau -> x/y fold): `F4[m][f][mu]` with
//! `m = 0 (nu), 1 (nubar)`, `f = 0 (e), 1 (mu), 2 (tau)`, and `mu` a Minkowski 4-vector
//! index where `mu = 3` is the (negative-signature) time leg = lab-frame number density
//! and `mu = 0,1,2` are the (positive-signature) spatial number-flux components -- this
//! sign convention is read directly off `FFISubgridModel::dot4_Minkowski`.
//!
//! Algorithm, per THC/Rhea's reference: within each nu/nubar sector, average the NF=3
//! flavor 4-vectors to get a reference vector `avgF4[m]` that IS timelike by construction
//! (an incoherent mix of physical states); for every flavor's 4-vector, solve for the
//! smallest boost `alpha >= 0` towards `avgF4[m]` that makes `F4[m][f] + alpha*avgF4[m]`
//! timelike; take the single largest `alpha` needed across all six (m,f) 4-vectors in the
//! cell and apply it uniformly to every one of them (matches `torch.amax(alpha, {1,2})`
//! taking the max over both the nu/nubar and flavor axes before boosting).
//!
//! NOTE(observed while implementing this package -- see final report): this only
//! mathematically guarantees the Minkowski norm is <= 0 (timelike-or-null); it does NOT
//! separately guarantee non-negative density (a future-pointing vector). A 4-vector that
//! is already timelike but *past*-pointing (e.g. `{0,0,0,-N}`, N>0 -- negative density,
//! zero flux) has no alpha>0 that helps: boosting towards an also-timelike average cannot
//! flip a vector's time-orientation. FFISubgridModel.h's own docstring claims the result
//! is "time-like and have positive density", but the algorithm itself (ported here
//! faithfully, matching Rhea's own C++ reference bit-for-logic) only delivers the former
//! in general. Confirmed empirically by this package's standalone test. This package's
//! stated acceptance criterion (design doc §10 Package 3) is timelike-ness only, which
//! this function does satisfy; do not assume non-negative density downstream without an
//! explicit additional floor (ApplyRheaMixing already floors N_post_* separately).
//!
//! NOTE(sherwood): FFISubgridModel.h's C++ `restrict_to_physical` has a dead-code bug --
//! the fallback branch for the near-degenerate `a ~= 0` case is computed
//! (`torch::where(torch::abs(a/b)<1e-6, (-c/b), alpha)`) but its result is never assigned
//! back to `alpha` (the statement's return value is simply discarded), so that branch is
//! actually inert in the reference implementation. Porting the bug verbatim would produce
//! NaN/Inf whenever an averaged 4-vector `avgF4[m]` is itself null-or-near-null (exactly
//! the adversarial inputs this port's acceptance criteria require to stay finite/timelike,
//! rhea_athenak_port_design.md §10 Package 3), so this port applies the evidently-intended
//! fix (actually use the linear solution `-c/b` when `a` is negligible relative to `b`),
//! with an additional `b ~= 0` guard (alpha = 0, i.e. no correction) that neither the
//! Python (`ml_tools.py:56-83`) nor the C++ reference handles explicitly. Confirm with
//! Sherwood before relying on this in a regime where it matters physically.
KOKKOS_INLINE_FUNCTION
void RestrictToPhysical(Real F4_final[2][3][4]) {
  constexpr int NF = 3;
  constexpr Real eps = 1e-6;

  // Minkowski inner product with the same signature FFISubgridModel::dot4_Minkowski uses:
  // index 3 (time) is negative, indices 0-2 (space) are positive.
  auto dot4 = [](const Real v1[4], const Real v2[4]) -> Real {
    Real result = -v1[3] * v2[3];
    for (int mu = 0; mu < 3; ++mu) {
      result += v1[mu] * v2[mu];
    }
    return result;
  };

  // Per-sector (nu / nubar) average 4-vector over the NF flavors.
  Real avgF4[2][4];
  for (int m = 0; m < 2; ++m) {
    for (int mu = 0; mu < 4; ++mu) {
      Real s = 0.0;
      for (int f = 0; f < NF; ++f) {
        s += F4_final[m][f][mu];
      }
      avgF4[m][mu] = s / NF;
    }
  }

  // Largest boost needed across all (m,f) 4-vectors in this cell.
  Real maxalpha = 0.0;
  for (int m = 0; m < 2; ++m) {
    const Real a = dot4(avgF4[m], avgF4[m]);
    for (int f = 0; f < NF; ++f) {
      const Real b = 2.0 * dot4(avgF4[m], F4_final[m][f]);
      const Real c = dot4(F4_final[m][f], F4_final[m][f]);
      Real radical = b * b - 4.0 * a * c;
      radical = Kokkos::max(radical, Real(0.0));

      Real alpha;
      if (Kokkos::abs(a) > eps * Kokkos::abs(b)) {
        const Real sign_a = (a > Real(0.0)) ? Real(1.0)
                           : (a < Real(0.0)) ? Real(-1.0) : Real(0.0);
        alpha = (-b + sign_a * Kokkos::sqrt(radical)) / (2.0 * a);
      } else if (Kokkos::abs(b) > Real(0.0)) {
        alpha = -c / b;  // linear solution; see NOTE(sherwood) above
      } else {
        alpha = 0.0;  // fully degenerate; no information to correct with
      }
      maxalpha = Kokkos::max(maxalpha, alpha);
    }
  }
  maxalpha += eps;
  maxalpha = Kokkos::max(maxalpha, Real(0.0));

  // Boost every flavor's 4-vector towards its sector's average by the same maxalpha.
  for (int m = 0; m < 2; ++m) {
    for (int f = 0; f < NF; ++f) {
      for (int mu = 0; mu < 4; ++mu) {
        F4_final[m][f][mu] =
            (F4_final[m][f][mu] + maxalpha * avgF4[m][mu]) / (maxalpha + 1.0);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::ReconstructMixingMatrix
//! \brief Build the 4x4 block-diagonal, column-stochastic flavor transition matrix
//! `Y[f][g] = P(g -> f)` from pre/post-mix number densities.
//!
//! Ports thc_M1_flavor_mix.cc:465-503. Species order matches the rest of radiation_m1
//! (radiation_m1_flavor_mix.cpp's own convention): `0 = nu_e, 1 = nu_ebar, 2 = nu_x,
//! 3 = nu_xbar`. `nn_pre` must already be floored (rad_N_floor) by the caller, exactly as
//! THC's `nn_pt` is before this reconstruction runs.
//!
//! For each of the two sectors {e,x} and {a,y} independently, solves for a survival
//! probability `p` from the pre/post density difference, clamped to [0,1]; falls back to
//! `p = 1` (identity, no mixing) if the pre-mix denominator is degenerate -- there is no
//! information in the density difference to pin down a mixing angle in that case. All
//! cross-sector entries of Y are exactly 0 (FFI cannot exchange between the neutrino and
//! antineutrino sectors). Y is column-stochastic (each column sums to exactly 1) by
//! construction, so applying it to N and to E within each sector conserves the sector
//! total exactly (rhea_athenak_port_design.md §2b, §9 Stage A acceptance criterion).
KOKKOS_INLINE_FUNCTION
void ReconstructMixingMatrix(Real const nn_pre[4], Real const N_post_e, Real const N_post_x,
                              Real const N_post_a, Real const N_post_y, Real Y[4][4]) {
  constexpr int ie = 0, ia = 1, ix = 2, iy = 3;
  constexpr Real p_eps = 1e-10;

  for (int fa = 0; fa < 4; ++fa) {
    for (int fb = 0; fb < 4; ++fb) {
      Y[fa][fb] = 0.0;
    }
  }

  // Neutrino sector: nu_e <-> nu_x
  const Real sum_pre_nu = nn_pre[ie] + nn_pre[ix];
  const Real denom_nu = nn_pre[ie] - nn_pre[ix];
  Real p_nu;
  if (Kokkos::abs(denom_nu) < p_eps * sum_pre_nu) {
    p_nu = 1.0;
  } else {
    p_nu = 0.5 + (N_post_e - N_post_x) / (2.0 * denom_nu);
    p_nu = Kokkos::min(Kokkos::max(p_nu, Real(0.0)), Real(1.0));
  }

  // Antineutrino sector: nu_ebar <-> nu_xbar
  const Real sum_pre_nubar = nn_pre[ia] + nn_pre[iy];
  const Real denom_nubar = nn_pre[ia] - nn_pre[iy];
  Real p_nubar;
  if (Kokkos::abs(denom_nubar) < p_eps * sum_pre_nubar) {
    p_nubar = 1.0;
  } else {
    p_nubar = 0.5 + (N_post_a - N_post_y) / (2.0 * denom_nubar);
    p_nubar = Kokkos::min(Kokkos::max(p_nubar, Real(0.0)), Real(1.0));
  }

  Y[ie][ie] = p_nu;           Y[ix][ie] = 1.0 - p_nu;
  Y[ie][ix] = 1.0 - p_nu;     Y[ix][ix] = p_nu;
  Y[ia][ia] = p_nubar;        Y[iy][ia] = 1.0 - p_nubar;
  Y[ia][iy] = 1.0 - p_nubar;  Y[iy][iy] = p_nubar;
}

//----------------------------------------------------------------------------------------
//! \fn int radiationm1::RheaBatchIndex
//! \brief Linear index into the rank-wide Rhea batch (design doc §6.1: `n_batch =
//! nmb_thispack*nx1*nx2*nx3`) for interior zone `(m,k,j,i)`.
//!
//! NOT one of the two functions the design doc's §4 illustrative kernel-signature list
//! mandates (`RestrictToPhysical`, `ReconstructMixingMatrix`) -- added here only so
//! Package 2's `PackRheaInputs` (which fills `rhea_f4_in_scratch` at this same index) and
//! Package 3's `ApplyRheaMixing` (which reads `F4_out`/`growthrate`/`stability` at this
//! index) share a single, unambiguous definition of the pack/unpack correspondence. The
//! design doc does not pin this mapping down explicitly; a silent mismatch here would be
//! exactly the kind of "no crash, just silently wrong physics" bug §5.4 warns about for
//! stream ordering, just on the indexing side instead. Row-major over `(m, k-ks, j-js,
//! i-is)` with `i` fastest, matching AthenaK's own `LayoutRight` `(m,n,k,j,i)` convention
//! (`athena.hpp:100`). Confirm this against whatever Package 2 actually implements.
KOKKOS_INLINE_FUNCTION
int RheaBatchIndex(int m, int k, int j, int i, int ks, int js, int is,
                    int nx3, int nx2, int nx1) {
  return ((m * nx3 + (k - ks)) * nx2 + (j - js)) * nx1 + (i - is);
}

}  // namespace radiationm1

#endif  // RADIATION_M1_RHEA_KERNELS_HPP
