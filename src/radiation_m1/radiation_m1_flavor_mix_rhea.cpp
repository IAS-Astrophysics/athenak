//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_flavor_mix_rhea.cpp
//! \brief Rhea-based neutrino flavor mixing: pack Rhea's input tensor from M1/ADM/EOS
//! state, and unpack/apply its prediction via BGK relaxation.
//!
//! ============================================================================
//! Implementation notes, please read before extending:
//!
//! `RestrictToPhysical` and `ReconstructMixingMatrix` (the physics kernels this mixing
//! scheme needs) live in radiation_m1_rhea_kernels.hpp as pure KOKKOS_INLINE_FUNCTION
//! free functions with zero Torch types, so they are unit-testable directly against
//! synthetic fixtures independent of a real Rhea model. Everything below composes those
//! two kernels into the per-cell BGK relaxation update.
//!
//! Two design choices worth flagging for whoever next touches this file:
//!
//! 1. SIGNATURE. `ApplyRheaMixing` does not call `RheaModel::Predict` itself: it takes
//!    the three prediction Views (F4_out, growthrate, stability) as explicit parameters
//!    beyond (Driver*, int). `RadiationM1::FlavorMix` calls PackRheaInputs ->
//!    prhea->Predict() -> ApplyRheaMixing sequentially instead, so Predict()'s result
//!    reaches ApplyRheaMixing as data. This also keeps this function genuinely free of
//!    Torch types and independent of radiation_m1_rhea.hpp.
//!
//! 2. UNITS. Rhea is scale-invariant in number density: `predict_all` normalizes
//!    `F4_in` by the total density before evaluating the network, then rescales both
//!    `F4_out` and `growthrate` back by the same factor on the way out
//!    (Rhea/model_training/ml_neuralnet.py:264-270, 311, 313-315). So `F4_in` can be
//!    packed straight from code-unit densities with no conversion, and `F4_out` comes
//!    back already in code units -- no EOS object, no unit-system query, needed for
//!    either direction. The only surviving conversion is `growthrate`, which Rhea
//!    returns in the same (now code-unit) number-density units as its input, not in
//!    1/s; ApplyRheaMixing below converts it with one fixed factor built from the two
//!    free-function unit tables `Primitive::MakeGeometricSolar()`/`MakeCGS()` (no EOS
//!    object, no dynamic_cast). This assumes code units are always
//!    G = c = Msun = 1 with number densities in fm^-3 -- see the ASSUMPTION comment at
//!    its point of use in ApplyRheaMixing for the full rationale.
//! ============================================================================

#include <algorithm>
#include <cassert>
#include <iostream>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "globals.hpp"
#include "mhd/mhd.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_rhea_kernels.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"

namespace radiationm1 {

//! ============================================================================
//! PackRheaInputs implementation notes, please read before extending:
//!
//! Implemented as a single per-cell Kokkos kernel -- no EOS-templated dynamic_cast
//! dispatch is needed here (unlike RadiationM1::CalcOpacityNurates/CalcOpacityNurates_,
//! radiation_m1_calc_opacities_nurates.cpp:18-93); see file-level NOTE 2 above for why.
//! Two points worth flagging for whoever next touches this file:
//!
//! 1. TETRAD. This kernel projects the lab-frame number 4-current onto a per-cell
//!    "Eulerian tetrad": leg 0 is the Eulerian normal n^mu itself; legs 1-3 are the
//!    coordinate x/y/z axes Gram-Schmidt-orthonormalized against all previous legs under
//!    the full 4-metric g_dd. Worked through by hand (not guessed): in that Gram-Schmidt
//!    loop, every leg-0 (time-leg) projection is identically zero for every spatial seed
//!    vector, because it reduces to an inner product against n_d's *spatial* components,
//!    which are exactly 0 by the ADM 3+1 definition n_d = (-alpha,0,0,0)
//!    (radiation_m1_tensors.hpp's pack_n_d). So the spatial legs never pick up a nonzero
//!    time component, and the full 4D tetrad construction reduces exactly to an ordinary
//!    3D Gram-Schmidt of the coordinate axes under the ADM 3-metric gamma_ij alone.
//!    BuildSpatialTriad below implements exactly that reduced 3D construction; the
//!    tetrad's leg-0 (time) projection of a covector is then just -n^mu N_mu (n^mu
//!    obtained from n_d via tensor_contract with g_uu) -- no separate 4D tetrad object is
//!    ever materialized.
//!
//! 2. FLUID-FRAME DECOMPOSITION. This pack kernel recomputes the per-species fluid-frame
//!    number density, energy density, and flux covector inline, every call (no persistent
//!    fluid-frame grid functions are stored), reusing exactly the same closure/projection
//!    machinery RadiationM1::CalcOpacityNurates_ already uses for J[]/rnnu[]/Gamma[]
//!    (radiation_m1_calc_opacities_nurates.cpp:167-199) -- plus
//!    radiation_m1_helpers.hpp's calc_H_from_rT for the flux covector H_d, which
//!    CalcOpacityNurates_ does not need but this pack kernel does. n/J/H_d are all
//!    divided by volform (sqrt(det gamma)) to match CalcOpacityNurates_'s "local
//!    undensitized neutrino quantities" convention (that file's own comment, line 193)
//!    before entering the N_mu = n(u_mu + H_mu/J) formula. Also note: scaling n_pt AND
//!    J_pt AND H_d by the per-(m,f)-slot flv_fac before forming the ratio H_d/J_pt would
//!    cancel exactly in the ratio, since H and J share the same factor -- so only n_pt
//!    actually needs the flv_fac multiplier in the final formula (see [E] below); this
//!    avoids double-applying flv_fac by accident.
//!
//! Unlike CalcOpacityNurates_, this kernel does NOT early-return for stage>1: the
//! FlavMixRhea dispatch branch that calls PackRheaInputs is required to fire every RK
//! stage, not once per timestep -- copying CalcOpacityNurates_'s stage>1 skip here would
//! silently freeze Rhea's input on stale substage-1 data every subsequent stage.
//!
//! Indexing: writes rhea_f4_in_scratch using RheaBatchIndex(m,k,j,i,ks,js,is,nx3,nx2,nx1)
//! (radiation_m1_rhea_kernels.hpp) -- the exact same index ApplyRheaMixing uses to read
//! F4_out/growthrate/stability back. Do not derive an independent formula here.
//! ============================================================================

namespace {

//----------------------------------------------------------------------------------------
//! \fn void BuildSpatialTriad
//! \brief Gram-Schmidt-orthonormalize the coordinate axes under the ADM 3-metric
//! gamma_ij, producing the purely-spatial legs of the per-cell Eulerian tetrad (see note
//! [1] above for why the full 4D construction reduces to this 3D one). `E[a][p]`: tetrad
//! leg a=0,1,2 (legs 1,2,3 of the full 4D tetrad, i.e. x,y,z), coordinate component
//! p=0,1,2.
KOKKOS_INLINE_FUNCTION
void BuildSpatialTriad(const Real gamma_dd[3][3], Real E[3][3]) {
  for (int a = 0; a < 3; ++a) {
    Real s[3] = {0.0, 0.0, 0.0};
    s[a] = 1.0;
    for (int b = 0; b < a; ++b) {
      Real inner = 0.0;
      for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 3; ++q) {
          inner += gamma_dd[p][q] * E[b][p] * s[q];
        }
      }
      for (int p = 0; p < 3; ++p) {
        s[p] -= inner * E[b][p];
      }
    }
    Real norm2 = 0.0;
    for (int p = 0; p < 3; ++p) {
      for (int q = 0; q < 3; ++q) {
        norm2 += gamma_dd[p][q] * s[p] * s[q];
      }
    }
    // std::sqrt(std::abs(...)) is a defensive guard against a numerically negative norm2.
    const Real norm = Kokkos::sqrt(Kokkos::abs(norm2));
    for (int p = 0; p < 3; ++p) {
      E[a][p] = s[p] / norm;
    }
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationM1::PackRheaInputs
//! \brief Fill rhea_f4_in_scratch from u0_/w0_/ADM state via an Eulerian-tetrad
//! projection of the lab-frame number 4-current, folding in the i_flv_map/flv_fac species
//! mapping in the same pass. No EOS/unit-system query needed (see file-level NOTE 2
//! above): F4_in is packed directly from code-unit densities, since Rhea is
//! scale-invariant in number density and does not need cgs inputs.
//! See the implementation notes above for the
//! tetrad-reduction and fluid-frame-decomposition details.
//!
//! Runs every RK stage (no stage>1 early exit -- see note above); requires nspecies == 4
//! (Rhea mixing's hard prerequisite, not relaxed elsewhere).
TaskStatus RadiationM1::PackRheaInputs(Driver *pdrive, int stage) {
  assert(nspecies == 4);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &u0_ = u0;
  auto &nvars_ = nvars;
  auto &params_ = params;
  auto &chi_ = chi;
  auto &f4_in_ = rhea_f4_in_scratch;

  DvceArray5D<Real> w0_ = w0;
  if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
  }

  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  // Species -> Rhea (m,f) mapping, lin = NF*m + f. Fixed by Rhea's own contract;
  // do not re-derive. AthenaK species order: 0=e, 1=a(bar-e), 2=x, 3=y(bar-x), matching
  // ReconstructMixingMatrix's convention (radiation_m1_rhea_kernels.hpp).
  constexpr int NF = 3;
  constexpr int i_flv_map[6] = {0, 2, 2, 1, 3, 3};
  constexpr Real flv_fac[6] = {1.0, 0.5, 0.5, 1.0, 0.5, 0.5};

  par_for(
      "radiation_m1_pack_rhea_inputs", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        // -------------------------------------------------------------------
        // [A] Metric, lapse, shift, volume form -- identical pattern to
        //     CalcOpacityNurates_/ApplyRheaMixing.
        // -------------------------------------------------------------------
        Real garr_dd[16], garr_uu[16];
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        adm::SpacetimeMetric(
            adm.alpha(m, k, j, i),
            adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
            adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
            garr_dd);
        adm::SpacetimeUpperMetric(
            adm.alpha(m, k, j, i),
            adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
            adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
            garr_uu);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            g_dd(a, b) = garr_dd[a + b * 4];
            g_uu(a, b) = garr_uu[a + b * 4];
          }
        }
        const Real alp = adm.alpha(m, k, j, i);
        const Real betax = adm.beta_u(m, 0, k, j, i);
        const Real betay = adm.beta_u(m, 1, k, j, i);
        const Real betaz = adm.beta_u(m, 2, k, j, i);

        const Real gam = adm::SpatialDet(
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
        const Real volform = Kokkos::sqrt(gam);

        // -------------------------------------------------------------------
        // [B] Eulerian normal (n_d, n_u) and fluid 4-velocity (u_u, u_d, v_u,
        //     v_d, proj_ud) -- identical pattern to CalcOpacityNurates_.
        // -------------------------------------------------------------------
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
        pack_n_d(alp, n_d);
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_u{};
        tensor_contract(g_uu, n_d, n_u);

        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};

        const Real w_lorentz = get_w_lorentz(
            w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i), w0_(m, IVZ, k, j, i), g_dd);
        pack_u_u(
            w_lorentz / alp,
            w0_(m, IVX, k, j, i) - w_lorentz * betax / alp,
            w0_(m, IVY, k, j, i) - w_lorentz * betay / alp,
            w0_(m, IVZ, k, j, i) - w_lorentz * betaz / alp,
            u_u);
        pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), alp, betax, betay, betaz, v_u);
        tensor_contract(g_dd, u_u, u_d);
        tensor_contract(g_dd, v_u, v_d);
        calc_proj(u_d, u_u, proj_ud);

        // -------------------------------------------------------------------
        // [C] Purely-spatial legs of the Eulerian tetrad (note [1]
        //     above).
        // -------------------------------------------------------------------
        Real gamma3[3][3];
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            gamma3[a][b] = adm.g_dd(m, a, b, k, j, i);
          }
        }
        Real triad[3][3];
        BuildSpatialTriad(gamma3, triad);

        // -------------------------------------------------------------------
        // [D] Per-species undensitized fluid-frame decomposition: number
        //     density n, energy density J, flux covector H_d. Mirrors
        //     CalcOpacityNurates_'s J[]/rnnu[]/Gamma[] construction
        //     (radiation_m1_calc_opacities_nurates.cpp:167-199) plus
        //     calc_H_from_rT for H_d (not needed there, needed here).
        // -------------------------------------------------------------------
        Real n_fluid[4], J_fluid[4];
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_fluid[4];
        for (int s = 0; s < 4; ++s) {
          const Real E_s = u0_(m, CombinedIdx(s, M1_E_IDX, nvars_), k, j, i);
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          pack_F_d(betax, betay, betaz,
                   u0_(m, CombinedIdx(s, M1_FX_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(s, M1_FY_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(s, M1_FZ_IDX, nvars_), k, j, i),
                   F_d);
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
          apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E_s, F_d,
                        chi_(m, s, k, j, i), P_dd, params_);
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
          assemble_rT(n_d, E_s, F_d, P_dd, T_dd);

          const Real J_s = calc_J_from_rT(T_dd, u_u);
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
          calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
          const Real Gamma_s = compute_Gamma(w_lorentz, v_u, J_s, E_s, F_d, params_);
          const Real N_s = u0_(m, CombinedIdx(s, M1_N_IDX, nvars_), k, j, i);

          n_fluid[s] = (N_s / Gamma_s) / volform;
          J_fluid[s] = J_s / volform;
          for (int a = 0; a < 4; ++a) {
            H_fluid[s](a) = H_d(a) / volform;
          }
        }

        // -------------------------------------------------------------------
        // [E] Per Rhea (mm,f) slot: apply the species mapping, build the
        //     covariant lab-frame number 4-current N_mu = n_pt*(u_mu + H_mu/J)
        //     with the lab-frame floor enforced before projection, project onto the
        //     Eulerian tetrad, and write into rhea_f4_in_scratch at RheaBatchIndex
        //     (no unit conversion -- code-unit densities are written straight through,
        //     see file-level NOTE 2).
        //
        //     NOTE: only n_pt carries the flv_fac multiplier here -- H_fluid
        //     and J_fluid (the fluid-frame quantities computed in [D]) are
        //     species-level, unscaled; scaling n_pt, J_pt, AND H_d all by
        //     flv_fac before forming H_d/J_pt would cancel exactly in the ratio
        //     (note [2] above), so applying it
        //     to the unscaled H_fluid/J_fluid ratio directly is algebraically
        //     identical and avoids double-applying it by accident.
        // -------------------------------------------------------------------
        const int idx = RheaBatchIndex(m, k, j, i, ks, js, is, nx3, nx2, nx1);
        for (int mm = 0; mm < 2; ++mm) {
          for (int f = 0; f < NF; ++f) {
            const int lin = NF * mm + f;
            const int s = i_flv_map[lin];
            const Real fac = flv_fac[lin];

            const Real n_pt = Kokkos::max(params_.rad_N_floor, n_fluid[s]) * fac;
            const Real J_raw = Kokkos::max(params_.rad_E_floor, J_fluid[s]);

            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> N_d{};
            for (int a = 0; a < 4; ++a) {
              N_d(a) = n_pt * (u_d(a) + H_fluid[s](a) / J_raw);
            }

            // Enforce the lab-frame number-density floor by adding a multiple
            // of u_d.
            const Real ndens = -tensor_dot(g_uu, N_d, u_d);
            const Real floor_prefactor =
                Kokkos::max(Real(0.0), params_.rad_N_floor - ndens);
            for (int a = 0; a < 4; ++a) {
              N_d(a) += floor_prefactor * u_d(a);
            }

            // Project onto the Eulerian tetrad: time leg = -n^mu N_mu (lab-
            // frame number density); spatial legs = the Gram-Schmidt triad
            // contracted with N_d's spatial components (note [1]).
            const Real N_time = -tensor_dot(n_u, N_d);
            Real N_space[3] = {0.0, 0.0, 0.0};
            for (int a = 0; a < 3; ++a) {
              for (int p = 0; p < 3; ++p) {
                N_space[a] += triad[a][p] * N_d(p + 1);
              }
            }

            f4_in_(idx, mm, f, 3) = static_cast<float>(N_time);
            for (int sp = 0; sp < 3; ++sp) {
              f4_in_(idx, mm, f, sp) = static_cast<float>(N_space[sp]);
            }
            // f4_in_ is packed directly from code-unit densities (no eos_units -> cgs
            // multiply -- see file-level NOTE 2), so these are plain NaN guards, not an
            // overflow check: a realistic neutrino code-unit density (~1e-5-1e-7, cf.
            // inputs/tests/rad_m1_rhea_singlezone.athinput's rad_N*) sits ~45 orders of
            // magnitude below float32's max (~3.4e38) and ~31 above its smallest normal
            // (~1.2e-38), so there is no realistic over/underflow risk left in this cast.
            // The old ~1e-1 overflow ceiling was entirely an artifact of the ~1e39 cgs
            // multiply this change removes. Kept (compiled out under
            // -DNDEBUG, matching CalcOpacityNurates_'s own isfinite asserts) purely to
            // catch a genuinely NaN/inf upstream state in debug/test builds instead of
            // an unexplained NaN much later.
            assert(Kokkos::isfinite(f4_in_(idx, mm, f, 3)));
            assert(Kokkos::isfinite(f4_in_(idx, mm, f, 0)));
            assert(Kokkos::isfinite(f4_in_(idx, mm, f, 1)));
            assert(Kokkos::isfinite(f4_in_(idx, mm, f, 2)));
          }
        }
      });  // par_for

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationM1::ApplyRheaMixing
//! \brief Unpack Rhea's per-cell prediction, reconstruct the flavor mixing matrix, and
//! relax the M1 moments towards the mixed state via BGK relaxation. See the file-level
//! NOTE above for the two design choices (signature, units) this implementation makes.
//!
//! Does NOT write the mixed state directly into u0 -- it relaxes towards it
//! exponentially, and does NOT apply any spatial smoothing to inv_tau across neighboring
//! cells (a 5x5x5 stencil was considered and explicitly deferred); gamma/lambda are
//! computed inline, per cell, with no persistent inv_tau_0/inv_tau_1 grid-function
//! arrays.
//!
//! \param rhea_f4_out      [n_batch, 2, NF=3, 4] float32, lab-frame code-unit number
//!                         density; n_batch/index convention: RheaBatchIndex
//!                         (radiation_m1_rhea_kernels.hpp).
//! \param rhea_growthrate  [n_batch] float32, linear FFI growth-rate proxy in the SAME
//!                         code-unit number-density units as rhea_f4_out (Rhea rescales
//!                         growthrate by the same normalization it applies to F4_out --
//!                         ml_neuralnet.py:313-315), *not* log-scaled and *not* 1/s
//!                         (predates Rhea commit 7bd7b98, which changed the model from
//!                         predicting log(growthrate) to growthrate). Converted to
//!                         1/code_time below via growthrate_to_code.
//! \param rhea_stability   [n_batch] float32, exactly 0.0 (unstable) or 1.0 (stable).
TaskStatus RadiationM1::ApplyRheaMixing(
    Driver *pdrive, int stage,
    const DvceArray4D<const float> &rhea_f4_out,
    const DvceArray1D<const float> &rhea_growthrate,
    const DvceArray1D<const float> &rhea_stability) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &u0_ = u0;
  auto &nvars_ = nvars;
  auto &params_ = params;

  Real dt = pmy_pack->pmesh->dt;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  Real const stability_threshold = params_.rhea_stability_threshold;
  Real const tau0_factor = params_.rhea_tau_0_factor;
  Real const tau1_factor = params_.rhea_tau_1_factor;

  // ASSUMPTION (this branch, PI decision): code units are G = c = Msun = 1 with
  // number densities in fm^-3, i.e. Primitive::MakeGeometricSolar() -- the only unit
  // system radiation_m1's Rhea path supports. Any other <mhd> units setting
  // (geometric_kilometer, nuclear, cgs) is already broken elsewhere in AthenaK and is
  // out of scope for this branch; no runtime guard is added here for it.
  //
  // rhea_f4_out/rhea_growthrate above are already in code-unit number density (Rhea
  // is scale-invariant in density -- ml_neuralnet.py:264-270, 311, 313-315 -- so
  // F4_in/F4_out need no eos_units <-> cgs conversion at all). growthrate is the one
  // output whose UNITS change from "number density" to "1/time", so it alone needs
  // converting to 1/code_time. Chain: [code n] -> [cm^-3] -> [1/s] -> [1/code_time].
  const Primitive::UnitSystem code_units = Primitive::MakeGeometricSolar();
  const Primitive::UnitSystem cgs_units  = Primitive::MakeCGS();
  // ndens_to_invsec = sqrt(2)*G_F/hbar (Rhea/model_training/ml_constants.py:4-9),
  // converting growthrate's number-density-unit proxy into a genuine rate [1/s].
  constexpr Real ndens_to_invsec = 1.9255158167467008e-22;
  const Real growthrate_to_code = code_units.NumberDensityConversion(cgs_units)
                                 * ndens_to_invsec
                                 * code_units.TimeConversion(cgs_units);

  // Counts cells where the Rhea prediction itself is non-finite (NaN/Inf) and mixing was
  // skipped there -- see the finiteness screen in block [C] below. Newer checkpoints (e.g.
  // model7278_cuda.pt) return NaN rather than throwing on the same Box3D edge cases the old
  // checkpoint crashed on (exact-zero density, near-luminal flux factor -- see
  // runs/rhea_box3d_bug_report/), and `stability` alone does not flag these cells safely:
  // it comes back a well-defined 0.0 ("unstable"), which would otherwise route them into
  // the mixing branch below with NaN inputs.
  Kokkos::View<int, DevMemSpace> nan_count_dev("rhea_nan_count");
  Kokkos::deep_copy(nan_count_dev, 0);

  par_for(
      "radiation_m1_apply_rhea_mixing", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const int idx = RheaBatchIndex(m, k, j, i, ks, js, is, nx3, nx2, nx1);

        // -------------------------------------------------------------------
        // [A] Metric (needed for apply_floor and the BGK lapse factor) --
        //     identical to the existing FlavorMix task.
        // -------------------------------------------------------------------
        Real garr_dd[16], garr_uu[16];
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        adm::SpacetimeMetric(
            adm.alpha(m, k, j, i),
            adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
            adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
            garr_dd);
        adm::SpacetimeUpperMetric(
            adm.alpha(m, k, j, i),
            adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
            adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
            garr_uu);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            g_dd(a, b) = garr_dd[a + b * 4];
            g_uu(a, b) = garr_uu[a + b * 4];
          }
        }
        const Real alp   = adm.alpha(m, k, j, i);
        const Real betax = adm.beta_u(m, 0, k, j, i);
        const Real betay = adm.beta_u(m, 1, k, j, i);
        const Real betaz = adm.beta_u(m, 2, k, j, i);

        // volform = sqrt(det gamma_ij), the ADM spatial-metric determinant's square
        // root -- the same purely-geometric densitization factor PackRheaInputs's
        // block [A] computes (radiation_m1_flavor_mix_rhea.cpp's pack half). u0_'s
        // stored N is densitized (carries this factor); Rhea's N_post_* (below) is
        // undensitized, since PackRheaInputs already divided by volform before
        // feeding Rhea (F4_out/F4_in carry no unit-system conversion at all -- see
        // file-level NOTE 2 -- so volform is the only factor at play here).
        // IMPORTANT: without dividing nn[] by volform before
        // ReconstructMixingMatrix, the reconstructed mixing angle picks up a spurious
        // 1/volform factor everywhere volform != 1 (i.e. essentially everywhere in a real
        // dynamical-GR run). Does not affect Y's column-stochastic conservation
        // property or the gamma=0 BGK no-op (both hold for any Y) -- only the
        // numeric value of the mixing angle itself.
        const Real gam_here = adm::SpatialDet(
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
        const Real volform = Kokkos::sqrt(gam_here);

        // -------------------------------------------------------------------
        // [B] Pre-mix N/E/F for the 4 species (0=e,1=a,2=x,3=y); causal floor.
        //     Rhea mixing requires nspecies == 4 (a hard prerequisite, not relaxed
        //     elsewhere).
        // -------------------------------------------------------------------
        Real nn[4], E[4], Fx[4], Fy[4], Fz[4];
        for (int s = 0; s < 4; ++s) {
          nn[s] = Kokkos::max(params_.rad_N_floor,
                              u0_(m, CombinedIdx(s, M1_N_IDX, nvars_), k, j, i));
          E[s]  = u0_(m, CombinedIdx(s, M1_E_IDX,  nvars_), k, j, i);
          Fx[s] = u0_(m, CombinedIdx(s, M1_FX_IDX, nvars_), k, j, i);
          Fy[s] = u0_(m, CombinedIdx(s, M1_FY_IDX, nvars_), k, j, i);
          Fz[s] = u0_(m, CombinedIdx(s, M1_FZ_IDX, nvars_), k, j, i);
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          pack_F_d(betax, betay, betaz, Fx[s], Fy[s], Fz[s], F_d);
          apply_floor(g_uu, E[s], F_d, params_);
          Fx[s] = F_d(1); Fy[s] = F_d(2); Fz[s] = F_d(3);
        }

        // -------------------------------------------------------------------
        // [C] RestrictToPhysical on Rhea's raw per-cell prediction (a deliberate
        //     safety step not present in the reference implementation this was ported
        //     from), then fold Rhea's mu/tau flavor slots back into AthenaK's x/y
        //     species. No unit conversion needed here: rhea_f4_out is already in
        //     code-unit number density (file-level NOTE 2).
        // -------------------------------------------------------------------
        Real F4_cell[2][3][4];
        bool finite = true;
        for (int mm = 0; mm < 2; ++mm) {
          for (int f = 0; f < 3; ++f) {
            for (int mu = 0; mu < 4; ++mu) {
              const Real val = static_cast<Real>(rhea_f4_out(idx, mm, f, mu));
              F4_cell[mm][f][mu] = val;
              finite = finite && Kokkos::isfinite(val);
            }
          }
        }
        finite = finite && Kokkos::isfinite(static_cast<Real>(rhea_growthrate(idx)))
                         && Kokkos::isfinite(static_cast<Real>(rhea_stability(idx)));
        if (!finite) {
          // The model's prediction for this cell is unusable (NaN/Inf) -- most likely a
          // Box3D edge case (see runs/rhea_box3d_bug_report/). Leave u0_ untouched, i.e.
          // treat this cell as "no mixing" this stage, rather than letting a NaN propagate
          // through RestrictToPhysical/ReconstructMixingMatrix/the BGK relaxation below:
          // that relaxation writes u0_ = lambda*u0_ + (1-lambda)*N_mix even when lambda=1
          // (the "stable" no-op case), and (1-1)*NaN = NaN in IEEE754, so the existing
          // stability-threshold branch alone would not have caught this.
          Kokkos::atomic_increment(&nan_count_dev());
          return;
        }
        RestrictToPhysical(F4_cell);

        const Real N_post_e = F4_cell[0][0][3];
        const Real N_post_x = F4_cell[0][1][3] + F4_cell[0][2][3];
        const Real N_post_a = F4_cell[1][0][3];
        const Real N_post_y = F4_cell[1][1][3] + F4_cell[1][2][3];

        // -------------------------------------------------------------------
        // [D] Reconstruct the block-diagonal column-stochastic mixing matrix.
        //     nn_pre_undens: nn[] (u0_'s densitized N, floored) undensitized by
        //     volform so it is on the same footing as N_post_* above (see the
        //     volform computation in [A] above). nn[] itself (still
        //     densitized) is deliberately left untouched for use in [E] below --
        //     Y is dimensionless, so applying it to the densitized nn[]/E[]/F[] is
        //     correct and preserves the sector-conservation property regardless of
        //     densitization.
        // -------------------------------------------------------------------
        Real const nn_pre_undens[4] = {nn[0] / volform, nn[1] / volform,
                                        nn[2] / volform, nn[3] / volform};
        Real Y[4][4];
        ReconstructMixingMatrix(nn_pre_undens, N_post_e, N_post_x, N_post_a, N_post_y, Y);

        // -------------------------------------------------------------------
        // [E] Apply Y to all five M1 moments (N, E, Fx, Fy, Fz); re-floor.
        // -------------------------------------------------------------------
        Real N_mix[4] = {}, E_mix[4] = {}, Fx_mix[4] = {}, Fy_mix[4] = {}, Fz_mix[4] = {};
        for (int f = 0; f < 4; ++f) {
          for (int g = 0; g < 4; ++g) {
            N_mix[f]  += Y[f][g] * nn[g];
            E_mix[f]  += Y[f][g] * E[g];
            Fx_mix[f] += Y[f][g] * Fx[g];
            Fy_mix[f] += Y[f][g] * Fy[g];
            Fz_mix[f] += Y[f][g] * Fz[g];
          }
          N_mix[f] = Kokkos::max(params_.rad_N_floor, N_mix[f]);
          E_mix[f] = Kokkos::max(params_.rad_E_floor, E_mix[f]);
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d_mix{};
          pack_F_d(betax, betay, betaz, Fx_mix[f], Fy_mix[f], Fz_mix[f], F_d_mix);
          apply_floor(g_uu, E_mix[f], F_d_mix, params_);
          Fx_mix[f] = F_d_mix(1); Fy_mix[f] = F_d_mix(2); Fz_mix[f] = F_d_mix(3);
        }

        // -------------------------------------------------------------------
        // [F] Per-cell BGK relaxation towards the mixed state (exact
        //     formula -- computed inline, no persistent inv_tau_0/1 fields,
        //     no 5x5x5 spatial smoothing).
        //
        // TODO(sherwood): confirm predict_all's growthrate output is
        //     linear (cm^-3), not log-scaled, for whichever Rhea checkpoint
        //     is actually deployed. This differs from earlier reference code, which
        //     applied exp() and predates Rhea commit 7bd7b98
        //     (2025-11-27), which changed the model from predicting
        //     log(growthrate) to growthrate.
        // -------------------------------------------------------------------
        const bool unstable = rhea_stability(idx) < stability_threshold;
        // Clamp the rate to >= 0. TODO(sherwood): Rhea's growthrate output can be
        // NEGATIVE even for a cell flagged unstable -- the returned growthrate is
        // growthrate_box3d + y_growthrate (ml_neuralnet.py:308), where the NN correction
        // y_growthrate is a bare 1x0e scalar with unconstrained sign, while stability is
        // derived from growthrate_box3d ALONE (ml_neuralnet.py:322), so the two are not
        // guaranteed sign-consistent. Without this clamp a negative rate gives inv_tau <
        // 0 -> lambda = exp(+x) > 1, i.e. BGK *anti-damping*: the update moves away from
        // the mix target each stage, an unphysical exponential blow-up of N/E/F.
        // Earlier reference code never hit this because it applied exp(lgr) >= 0; the
        // switch to the linear growthrate interpretation silently dropped that positivity
        // guarantee, so restore it here. Confirm the intended semantics with Sherwood.
        const Real gamma_code = unstable
            ? Kokkos::max(Real(0),
                          static_cast<Real>(rhea_growthrate(idx)) * growthrate_to_code)
            : Real(0);
        const Real inv_tau_0 = gamma_code * tau0_factor;
        const Real inv_tau_1 = gamma_code * tau1_factor;
        const Real lambda_0 = Kokkos::exp(-dt * alp * inv_tau_0);
        const Real lambda_1 = Kokkos::exp(-dt * alp * inv_tau_1);

        // NOTE: the "old" term below intentionally re-reads u0_ directly
        // (the raw, un-floored stored value), not the floored local nn[]/
        // E[]/Fx[]/Fy[]/Fz[] computed in [B] -- this is what makes the
        // gamma=0 => lambda=1 no-op bit-identical to u0_, matching
        // the existing equilibrium/maximal FlavorMix task
        // (radiation_m1_flavor_mix.cpp:245-259), which likewise re-reads the raw stored
        // value rather than a floored local copy.
        for (int s = 0; s < 4; ++s) {
          u0_(m, CombinedIdx(s, M1_N_IDX,  nvars_), k, j, i) =
              lambda_0 * u0_(m, CombinedIdx(s, M1_N_IDX,  nvars_), k, j, i)
              + (1.0 - lambda_0) * N_mix[s];
          u0_(m, CombinedIdx(s, M1_E_IDX,  nvars_), k, j, i) =
              lambda_1 * u0_(m, CombinedIdx(s, M1_E_IDX,  nvars_), k, j, i)
              + (1.0 - lambda_1) * E_mix[s];
          u0_(m, CombinedIdx(s, M1_FX_IDX, nvars_), k, j, i) =
              lambda_1 * u0_(m, CombinedIdx(s, M1_FX_IDX, nvars_), k, j, i)
              + (1.0 - lambda_1) * Fx_mix[s];
          u0_(m, CombinedIdx(s, M1_FY_IDX, nvars_), k, j, i) =
              lambda_1 * u0_(m, CombinedIdx(s, M1_FY_IDX, nvars_), k, j, i)
              + (1.0 - lambda_1) * Fy_mix[s];
          u0_(m, CombinedIdx(s, M1_FZ_IDX, nvars_), k, j, i) =
              lambda_1 * u0_(m, CombinedIdx(s, M1_FZ_IDX, nvars_), k, j, i)
              + (1.0 - lambda_1) * Fz_mix[s];
        }
      });  // par_for

  int nan_count_host = 0;
  Kokkos::deep_copy(nan_count_host, nan_count_dev);
  if (nan_count_host > 0 && global_variable::my_rank == 0) {
    std::cout << "RadiationM1::ApplyRheaMixing: WARNING - Rhea prediction non-finite at "
              << nan_count_host << " cell(s) this stage; mixing skipped there." << std::endl;
  }

  return TaskStatus::complete;
}

}  // namespace radiationm1
