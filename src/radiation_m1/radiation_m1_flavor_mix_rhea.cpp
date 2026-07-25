//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_flavor_mix_rhea.cpp
//! \brief Rhea-based neutrino flavor mixing: pack Rhea's input tensor from M1/ADM/EOS
//! state, and unpack/apply its prediction via BGK relaxation.
//!
//! NOTE(package split): this file is shared by two independently-scoped work packages
//! (rhea_athenak_port_design.md §4, §10): Package 2 owns `RadiationM1::PackRheaInputs`/
//! `PackRheaInputs_` (the pack half); Package 3 owns `RadiationM1::ApplyRheaMixing` and
//! everything below (the unpack/mix half). Package 2's pack half has since been added,
//! immediately after the `namespace radiationm1 {` opening and before Package 3's
//! implementation notes/`ApplyRheaMixing` below (Package 2 was not running concurrently
//! with Package 3, so nothing below this point was overwritten).
//!
//! ============================================================================
//! Package 3 (unpack/mix half) -- implementation notes, please read before extending:
//!
//! `RestrictToPhysical` and `ReconstructMixingMatrix` (the physics kernels the design doc
//! §4 explicitly mandates) live in radiation_m1_rhea_kernels.hpp, are pure
//! KOKKOS_INLINE_FUNCTION free functions with zero Torch types, and are unit-tested
//! directly against synthetic fixtures (see the accompanying standalone Kokkos-only test,
//! not part of this repo's CMake build -- rhea_athenak_port_design.md §10 Package 3's
//! "tested via"). Everything below composes those two kernels into the per-cell BGK
//! relaxation update, ported from thc_M1_flavor_mix.cc:255-711 (§2b).
//!
//! Two genuine open points this file could not resolve without touching files outside
//! this package's scope (radiation_m1.hpp/radiation_m1_params.hpp, owned by Package 4)
//! are flagged explicitly where they matter, and repeated in the implementing agent's
//! final report:
//!
//! 1. SIGNATURE. Design doc §4 shows `TaskStatus RadiationM1::ApplyRheaMixing(Driver*,
//!    int)` with no further parameters, and its own prose says it "calls prhea->Predict(),
//!    then in one Kokkos kernel...". But §10 Package 3 says this package "does not call
//!    RheaModel::Predict itself" and should take "a RheaModel::Prediction ... or ... any
//!    Views of the matching shape/type", and §10 Package 4 independently describes
//!    `RadiationM1::FlavorMix` "calling PackRheaInputs -> RheaModel::Predict ->
//!    ApplyRheaMixing sequentially" -- three separate top-level calls, which only makes
//!    sense if Predict()'s result reaches ApplyRheaMixing as data, not via an internal
//!    Predict() call. This implementation follows the latter (§10) reading: it does not
//!    call Predict() itself, and takes the three prediction Views as explicit parameters
//!    beyond (Driver*, int). This also lets this function stay genuinely free of Torch
//!    types (design doc §5.9) and independent of radiation_m1_rhea.hpp, which does not
//!    exist in this working tree as of this writing (Package 1 runs concurrently with,
//!    and this package does not depend on, Package 1). Package 4, which owns
//!    radiation_m1.hpp, should treat this signature choice as a proposal to confirm/adjust
//!    when it adds the actual class declaration, not as settled fact.
//!
//! 2. UNITS. §6.1 says the reverse conversion of F4_out's post-mix densities back to code
//!    units ("the same is true in reverse for ApplyRheaMixing") uses the same
//!    `eos.GetEOSUnitSystem().NumberDensityConversion(Primitive::MakeCGS())` factor
//!    PackRheaInputs computes going the other way -- which requires the EOS-templated
//!    dynamic_cast dispatch pattern (radiation_m1_calc_opacities_nurates.cpp:18-46). That
//!    dispatch needs `pmy_pack`, a *private* RadiationM1 member, so it can only be done in
//!    a genuine RadiationM1 member function template -- which would need a new
//!    `ApplyRheaMixing_<EOSPolicy,ErrorPolicy>` declaration in radiation_m1.hpp, again
//!    outside this package's file scope (§10 Package 3's own "May assume" list does not
//!    mention EOS/unit access at all, unlike Package 2's). To avoid duplicating or
//!    guessing at Package 2's EOS-dispatch implementation, `ApplyRheaMixing` below takes
//!    the conversion factor as a caller-supplied scalar (`unit_num_dens`) instead of
//!    computing it itself. Whoever wires Package 4's FlavMixRhea dispatch branch should
//!    pass the same `unit_num_dens` PackRheaInputs already computed for the forward
//!    conversion (or recompute it identically) -- do not let the two drift apart.
//! ============================================================================

#include <cassert>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "mhd/mhd.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_rhea_kernels.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"
#include "units/units.hpp"

namespace radiationm1 {

//! ============================================================================
//! Package 2 (pack half) -- implementation notes, please read before extending:
//!
//! Ported from THC's thc_M1_rhea.cc:65-174 (design doc §2a/§6.1), restructured into a
//! per-cell Kokkos kernel, mirroring RadiationM1::CalcOpacityNurates/CalcOpacityNurates_'s
//! dynamic_cast-dispatch-to-templated-helper pattern exactly
//! (radiation_m1_calc_opacities_nurates.cpp:18-93). Two points worth flagging for whoever
//! next touches this file:
//!
//! 1. TETRAD. THC projects the lab-frame number 4-current onto a per-cell "Eulerian
//!    tetrad" built by CPPUtils/src/utils_tetrad.hh's build_tetrad(n^mu, g_dd, &e) (CPPUtils
//!    is THC's own shared tensor-utilities dependency, outside both read-only reference
//!    codebases this port is scoped to -- Rhea/, THCode/ -- so it was read for background
//!    only, not ported from directly): leg 0 is the Eulerian normal n^mu itself; legs 1-3
//!    are the coordinate x/y/z axes Gram-Schmidt-orthonormalized against all previous legs
//!    under the full 4-metric g_dd. Worked through by hand (not guessed): in that
//!    Gram-Schmidt loop, every leg-0 (time-leg) projection is identically zero for every
//!    spatial seed vector, because it reduces to an inner product against n_d's *spatial*
//!    components, which are exactly 0 by the ADM 3+1 definition n_d = (-alpha,0,0,0)
//!    (radiation_m1_tensors.hpp's pack_n_d). So the spatial legs never pick up a nonzero
//!    time component, and THC's full 4D construction reduces exactly to an ordinary 3D
//!    Gram-Schmidt of the coordinate axes under the ADM 3-metric gamma_ij alone.
//!    BuildSpatialTriad below implements exactly that reduced 3D construction; the tetrad's
//!    leg-0 (time) projection of a covector is then just -n^mu N_mu (n^mu obtained from n_d
//!    via tensor_contract with g_uu) -- no separate 4D tetrad object is ever materialized.
//!
//! 2. FLUID-FRAME DECOMPOSITION. AthenaK's radiation_m1 does not persist THC's precomputed
//!    fluid-frame grid functions (rnnu/rJ/rHt,x,y,z); this pack kernel recomputes the
//!    per-species fluid-frame number density, energy density, and flux covector inline,
//!    every call, reusing exactly the same closure/projection machinery
//!    RadiationM1::CalcOpacityNurates_ already uses for J[]/rnnu[]/Gamma[]
//!    (radiation_m1_calc_opacities_nurates.cpp:167-199) -- plus radiation_m1_helpers.hpp's
//!    calc_H_from_rT for the flux covector H_d, which CalcOpacityNurates_ does not need but
//!    this pack kernel does. n/J/H_d are all divided by volform (sqrt(det gamma)) to match
//!    CalcOpacityNurates_'s "local undensitized neutrino quantities" convention (that file's
//!    own comment, line 193) before entering THC's N_mu = n(u_mu + H_mu/J) formula. Also
//!    note: THC scales n_pt AND J_pt AND H_d by the per-(m,f)-slot flv_fac before forming
//!    the ratio H_d/J_pt (thc_M1_rhea.cc:122-131) -- since H and J are scaled by the SAME
//!    factor, it cancels exactly in the ratio, so only n_pt actually needs the flv_fac
//!    multiplier in the final formula (see [E] below); ported algebraically simplified,
//!    not verbatim, to avoid double-applying flv_fac by accident.
//!
//! Unlike CalcOpacityNurates_, this kernel does NOT early-return for stage>1: the
//! FlavMixRhea dispatch branch (Package 4) that will call PackRheaInputs is required to fire
//! every RK stage, not once per timestep (design doc §3, PI-confirmed) -- copying
//! CalcOpacityNurates_'s stage>1 skip here would silently freeze Rhea's input on stale
//! substage-1 data every subsequent stage.
//!
//! Indexing: writes rhea_f4_in_scratch using RheaBatchIndex(m,k,j,i,ks,js,is,nx3,nx2,nx1)
//! (radiation_m1_rhea_kernels.hpp) -- the exact same index Package 3's ApplyRheaMixing uses
//! to read F4_out/growthrate/stability back. Do not derive an independent formula here.
//! ============================================================================

namespace {

//----------------------------------------------------------------------------------------
//! \fn void BuildSpatialTriad
//! \brief Gram-Schmidt-orthonormalize the coordinate axes under the ADM 3-metric gamma_ij,
//! producing the purely-spatial legs of THC's Eulerian tetrad (see the Package 2 note [1]
//! above for why the full 4D construction reduces to this 3D one). `E[a][p]`: tetrad leg
//! a=0,1,2 (THC's legs 1,2,3, i.e. x,y,z), coordinate component p=0,1,2. Ported from
//! CPPUtils/src/utils_tetrad.hh's build_tetrad, restricted to its spatial legs.
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
    // std::sqrt(std::abs(...)) matches build_tetrad's own defensive norm, verbatim.
    const Real norm = Kokkos::sqrt(Kokkos::abs(norm2));
    for (int p = 0; p < 3; ++p) {
      E[a][p] = s[p] / norm;
    }
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationM1::PackRheaInputs
//! \brief Dispatch to the EOS-templated PackRheaInputs_ (needed for eos.GetEOSUnitSystem(),
//! §6.1), exactly mirroring RadiationM1::CalcOpacityNurates's dynamic_cast dispatch pattern
//! (radiation_m1_calc_opacities_nurates.cpp:18-46). Requires pmy_pack->pdyngr != nullptr
//! (dynamical-GR MHD active, §6.1/§11.9) -- if it is null, both dynamic_casts below fail and
//! this falls through to the same "Unsupported EOS type" abort CalcOpacityNurates uses.
TaskStatus RadiationM1::PackRheaInputs(Driver *pdrive, int stage) {
  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(
          pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    return PackRheaInputs_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor>(pdrive, stage);
  }

  auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
      pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    return PackRheaInputs_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor>(pdrive, stage);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationM1::PackRheaInputs_
//! \brief Fill rhea_f4_in_scratch from u0_/w0_/ADM state via THC's Eulerian-tetrad
//! projection of the lab-frame number 4-current (thc_M1_rhea.cc:65-174, design doc §2a),
//! folding in the i_flv_map/flv_fac species mapping and the eos_units -> cgs number-density
//! conversion (§6.1) in the same pass. See the Package 2 implementation notes above for the
//! tetrad-reduction and fluid-frame-decomposition details.
//!
//! Runs every RK stage (no stage>1 early exit -- see note above); requires nspecies == 4
//! (Rhea mixing's hard, non-goal-to-relax prerequisite, §1).
template <class EOSPolicy, class ErrorPolicy>
TaskStatus RadiationM1::PackRheaInputs_(Driver *pdrive, int stage) {
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

  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();

  // eos_units -> cgs number-density conversion factor (§6.1); this is the ONLY conversion
  // factor needed -- F4_in's spatial (flux) components share the same number-density units
  // as its time component (§2a/§6.1), so there is no separate EnergyDensityConversion call,
  // unlike CalcOpacityNurates_. Deliberately a plain local (not stashed anywhere) so
  // Package 4's FlavMixRhea dispatch branch can cheaply recompute the identical one-line
  // expression for ApplyRheaMixing's reverse conversion (design doc §6.1's "SETTLED
  // (Package 3)" note) -- no shared-state mechanism needed for a cheap host-side scalar.
  Real const unit_num_dens =
      eos.GetEOSUnitSystem().NumberDensityConversion(Primitive::MakeCGS());

  // Species -> Rhea (m,f) mapping, lin = NF*m + f (§2a, thc_M1_rhea.cc:79-85). Port exactly;
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
        // [C] Purely-spatial legs of THC's Eulerian tetrad (Package 2 note [1]
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
        //     with the lab-frame floor enforced before projection (thc_M1_
        //     rhea.cc:120-141), project onto the Eulerian tetrad, convert to
        //     cgs, and write into rhea_f4_in_scratch at RheaBatchIndex.
        //
        //     NOTE: only n_pt carries the flv_fac multiplier here -- H_fluid
        //     and J_fluid (the fluid-frame quantities computed in [D]) are
        //     species-level, unscaled; THC scales n_pt, J_pt, AND H_d all by
        //     flv_fac before forming H_d/J_pt, but that factor cancels
        //     exactly in the ratio (Package 2 note [2] above), so applying it
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
            // of u_d (thc_M1_rhea.cc:136-141).
            const Real ndens = -tensor_dot(g_uu, N_d, u_d);
            const Real floor_prefactor =
                Kokkos::max(Real(0.0), params_.rad_N_floor - ndens);
            for (int a = 0; a < 4; ++a) {
              N_d(a) += floor_prefactor * u_d(a);
            }

            // Project onto the Eulerian tetrad: time leg = -n^mu N_mu (lab-
            // frame number density); spatial legs = the Gram-Schmidt triad
            // contracted with N_d's spatial components (Package 2 note [1]).
            const Real N_time = -tensor_dot(n_u, N_d);
            Real N_space[3] = {0.0, 0.0, 0.0};
            for (int a = 0; a < 3; ++a) {
              for (int p = 0; p < 3; ++p) {
                N_space[a] += triad[a][p] * N_d(p + 1);
              }
            }

            f4_in_(idx, mm, f, 3) = static_cast<float>(N_time * unit_num_dens);
            for (int sp = 0; sp < 3; ++sp) {
              f4_in_(idx, mm, f, sp) = static_cast<float>(N_space[sp] * unit_num_dens);
            }
            // Guard the eos_units -> cgs conversion: unit_num_dens ~ 1e39 for a typical EOS
            // table, and F4_in is float32 (max ~3.4e38), so a code-unit density above ~1e-1
            // overflows to inf here -- which would then silently poison N_post/ntot/growthrate
            // downstream and break the gamma=0 stable-zone bit-identity (1*N_old + 0*nan =
            // nan). Real neutrino densities (~1e-5 code units) sit ~4 orders below that
            // ceiling, so this only fires on a pathological/overflowing input; the assert
            // (compiled out under -DNDEBUG, matching CalcOpacityNurates_'s own isfinite
            // asserts) catches it in debug/test builds instead of as an unexplained NaN much
            // later. See design doc §11 (float32 magnitude constraint, found in Package 5).
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
//! relax the M1 moments towards the mixed state via BGK relaxation (rhea_athenak_port_
//! design.md §2b). See the file-level NOTE above for the two open integration points
//! (signature, units) this implementation had to make a documented choice on.
//!
//! Does NOT write the mixed state directly into u0 -- it relaxes towards it exponentially
//! (§2b), and does NOT implement THC's 5x5x5 spatial smoothing of inv_tau (explicitly
//! deferred, §2b/§11.1); gamma/lambda are computed inline, per cell, with no persistent
//! inv_tau_0/inv_tau_1 grid-function arrays.
//!
//! \param rhea_f4_out      [n_batch, 2, NF=3, 4] float32, lab-frame cgs number density
//!                         (§2a); n_batch/index convention: RheaBatchIndex (radiation_m1_
//!                         rhea_kernels.hpp).
//! \param rhea_growthrate  [n_batch] float32, linear FFI growth-rate proxy in cm^-3, *not*
//!                         log-scaled and *not* 1/s (§2a's PI-confirmed bug fix -- THC's
//!                         C++ applies exp() here and predates Rhea commit 7bd7b98).
//! \param rhea_stability   [n_batch] float32, exactly 0.0 (unstable) or 1.0 (stable).
//! \param unit_num_dens    eos_units -> cgs number-density conversion factor (§6.1); see
//!                         NOTE 2 above for why this is a parameter instead of being
//!                         computed here.
TaskStatus RadiationM1::ApplyRheaMixing(
    Driver *pdrive, int stage,
    const DvceArray4D<const float> &rhea_f4_out,
    const DvceArray1D<const float> &rhea_growthrate,
    const DvceArray1D<const float> &rhea_stability,
    Real unit_num_dens) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &u0_ = u0;
  auto &nvars_ = nvars;
  auto &params_ = params;

  bool isunits_ = isunits;
  Real time_cgs = isunits ? pmy_pack->punit->time_cgs() : Real(1.0);
  Real const unit_num_dens_ = unit_num_dens;

  Real dt = pmy_pack->pmesh->dt;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  Real const stability_threshold = params_.rhea_stability_threshold;
  Real const tau0_factor = params_.rhea_tau_0_factor;
  Real const tau1_factor = params_.rhea_tau_1_factor;

  // ndens_to_invsec = sqrt(2)*G_F/hbar (Rhea/model_training/ml_constants.py:4-9),
  // converting growthrate's number-density-unit proxy into a genuine rate [1/s].
  constexpr Real ndens_to_invsec = 1.9255158167467008e-22;

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
        // root -- the same purely-geometric densitization factor PackRheaInputs_'s
        // block [A] computes (radiation_m1_flavor_mix_rhea.cpp's pack half). u0_'s
        // stored N is densitized (carries this factor); Rhea's N_post_* (below) is
        // undensitized, since PackRheaInputs_ already divided by volform before
        // feeding Rhea (and unit_num_dens is a pure unit-system conversion, not a
        // densitization). BLOCKING FIX (design doc §10 Package 4, review-found):
        // without dividing nn[] by volform before ReconstructMixingMatrix, the
        // reconstructed mixing angle picks up a spurious 1/volform factor
        // everywhere volform != 1 (i.e. essentially everywhere in a real
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
        //     Rhea mixing requires nspecies == 4 (§1, non-goal to relax).
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
        // [C] RestrictToPhysical on Rhea's raw per-cell prediction (§2a PI
        //     decision -- THC does not call this, we do), then fold Rhea's
        //     mu/tau flavor slots back into AthenaK's x/y species (§2a) and
        //     convert the post-mix densities back to code units (§6.1).
        // -------------------------------------------------------------------
        Real F4_cell[2][3][4];
        for (int mm = 0; mm < 2; ++mm) {
          for (int f = 0; f < 3; ++f) {
            for (int mu = 0; mu < 4; ++mu) {
              F4_cell[mm][f][mu] = static_cast<Real>(rhea_f4_out(idx, mm, f, mu));
            }
          }
        }
        RestrictToPhysical(F4_cell);

        const Real N_post_e = F4_cell[0][0][3] / unit_num_dens_;
        const Real N_post_x = (F4_cell[0][1][3] + F4_cell[0][2][3]) / unit_num_dens_;
        const Real N_post_a = F4_cell[1][0][3] / unit_num_dens_;
        const Real N_post_y = (F4_cell[1][1][3] + F4_cell[1][2][3]) / unit_num_dens_;

        // -------------------------------------------------------------------
        // [D] Reconstruct the block-diagonal column-stochastic mixing matrix.
        //     nn_pre_undens: nn[] (u0_'s densitized N, floored) undensitized by
        //     volform so it is on the same footing as N_post_* above (BLOCKING FIX,
        //     see the volform computation in [A] above). nn[] itself (still
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
        // [F] Per-cell BGK relaxation towards the mixed state (§2b, exact
        //     formula -- computed inline, no persistent inv_tau_0/1 fields,
        //     no 5x5x5 spatial smoothing, per the PI decision).
        //
        //     TODO(sherwood): confirm predict_all's growthrate output is
        //     linear (cm^-3), not log-scaled, for whichever Rhea checkpoint
        //     is actually deployed. This differs from THC's thc_M1_flavor_
        //     mix.cc, which applies exp() and predates Rhea commit 7bd7b98
        //     (2025-11-27), which changed the model from predicting
        //     log(growthrate) to growthrate.
        // -------------------------------------------------------------------
        const bool unstable = rhea_stability(idx) < stability_threshold;
        // Clamp the rate to >= 0. TODO(sherwood): Rhea's growthrate output can be NEGATIVE
        // even for a cell flagged unstable -- the returned growthrate is growthrate_box3d +
        // y_growthrate (ml_neuralnet.py:308), where the NN correction y_growthrate is a bare
        // 1x0e scalar with unconstrained sign, while stability is derived from
        // growthrate_box3d ALONE (ml_neuralnet.py:322), so the two are not guaranteed
        // sign-consistent (design doc §2a). Without this clamp a negative rate gives
        // inv_tau < 0 -> lambda = exp(+x) > 1, i.e. BGK *anti-damping*: the update moves
        // away from the mix target each stage, an unphysical exponential blow-up of N/E/F.
        // THC never hit this because it applied exp(lgr) >= 0; the (correct, PI-confirmed)
        // switch to the linear growthrate interpretation silently dropped that positivity
        // guarantee, so restore it here. Confirm the intended semantics with Sherwood.
        const Real gamma_persec = unstable
            ? Kokkos::max(Real(0),
                          static_cast<Real>(rhea_growthrate(idx)) * ndens_to_invsec)
            : Real(0);
        const Real gamma_code = isunits_ ? gamma_persec * time_cgs : gamma_persec;
        const Real inv_tau_0 = gamma_code * tau0_factor;
        const Real inv_tau_1 = gamma_code * tau1_factor;
        const Real lambda_0 = Kokkos::exp(-dt * alp * inv_tau_0);
        const Real lambda_1 = Kokkos::exp(-dt * alp * inv_tau_1);

        // NOTE: the "old" term below intentionally re-reads u0_ directly
        // (the raw, un-floored stored value), not the floored local nn[]/
        // E[]/Fx[]/Fy[]/Fz[] computed in [B] -- this is what makes the
        // gamma=0 => lambda=1 no-op bit-identical to u0_ (§2b), and matches
        // both THC (thc_M1_flavor_mix.cc:672, using rN[i4D] not nn_pt[f])
        // and the existing equilibrium/maximal FlavorMix task
        // (radiation_m1_flavor_mix.cpp:245-259).
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

  return TaskStatus::complete;
}

}  // namespace radiationm1
