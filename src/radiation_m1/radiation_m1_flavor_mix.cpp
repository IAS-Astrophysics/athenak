//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_flavor_mix.cpp
//! \brief Neutrino flavor mixing for grey M1 transport.
//!
//! Implements equilibrium and maximal flavor mixing with BGK relaxation,
//! following D. Radice, arXiv:2307.16793.
//!
//! Algorithm (per cell):
//!   1. Load N, E, F for all species; apply causal floor.
//!   2. Compute mixing invariants N_tot, Ne, Nx.
//!   3. Solve for target mixed number densities (equilibrium or maximal).
//!   4. Enforce positivity via alpha-blending with the original state.
//!   5. Build 4x4 column-stochastic transition matrix Y[f][g] = P(g->f).
//!   6. Apply Y to energy densities and fluxes; re-apply causal floor.
//!   7. Apply BGK relaxation: Q_new = exp(-dt*alp*inv_tau)*Q_old
//!                                  + (1 - exp(-dt*alp*inv_tau))*Q_mix
//!
//! Species convention: 0=nu_e, 1=nu_ebar, 2=nu_x, 3=nu_xbar
//! Mixing channels: nu_e <-> nu_x,  nu_ebar <-> nu_xbar
//!
//! Input parameters (block [radiation_m1]):
//!   flavor_mix      = none | equilibrium | maximal  (default: none)
//!   bgk_inv_tau_0   = BGK rate for number density   (default: 0 = no mixing)
//!                     [1/s if <units> block present; 1/code_time otherwise]
//!   bgk_inv_tau_1   = BGK rate for energy density   (default: 0 = no mixing)
//!                     [1/s if <units> block present; 1/code_time otherwise]
//!   flavor_mix_rho  = density threshold below which mixing is skipped (default: -1 = everywhere)
//!                     [g/cm^3 if <units> block present; code density otherwise]

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"

#if ENABLE_TORCH
#include <type_traits>
#include <utility>

#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#endif

namespace radiationm1 {

#if ENABLE_TORCH
namespace {
//! \fn Real RheaUnitNumDens
//! \brief eos_units -> cgs number-density conversion factor, computed the same way
//! PackRheaInputs_ computes it for the forward (pack) conversion -- needed here too since
//! ApplyRheaMixing takes it as an explicit parameter rather than computing it itself (see
//! radiation_m1_flavor_mix_rhea.cpp's file-level NOTE 2).
template <class EOSPolicy, class ErrorPolicy>
Real RheaUnitNumDens(MeshBlockPack *pmy_pack) {
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  return eos.GetEOSUnitSystem().NumberDensityConversion(Primitive::MakeCGS());
}
}  // namespace
#endif

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationM1::FlavorMix
//! \brief Apply neutrino flavor mixing after each time update, before ghost sync.
TaskStatus RadiationM1::FlavorMix(Driver *pdrive, int stage) {
  // Skip if mixing is disabled or only one species (photon transport)
  if (params.flavor_mix_type == FlavMixNone || nspecies <= 1) {
    return TaskStatus::complete;
  }

#if ENABLE_TORCH
  // Rhea ML flavor mixing: structurally different from the equilibrium/maximal branches
  // below (a batched Torch call sandwiched between two Kokkos kernels, not one
  // self-contained par_for), so it dispatches to its own functions rather than being
  // inlined into the KOKKOS_LAMBDA further down. No new TaskIDs -- PackRheaInputs ->
  // prhea->Predict() -> ApplyRheaMixing run sequentially inside this one task-function
  // invocation, precedented by CalculateFluxes's sequential par_for calls in one task
  // body (radiation_m1_fluxes.cpp).
  if (params.flavor_mix_type == FlavMixRhea) {
    // Same dynamic_cast-dispatch pattern as RadiationM1::CalcOpacityNurates/PackRheaInputs
    // (radiation_m1_calc_opacities_nurates.cpp:18-46). NOTE: PackRheaInputs below performs
    // this same dispatch internally (PackRheaInputs_ computes its own unit_num_dens rather
    // than taking it as a parameter), so this is a second, harmless dispatch of the same
    // cheap host-side scalar computation, not a duplicated device kernel.
    Real unit_num_dens;
    auto *ptest_nqt =
        dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                       Primitive::ResetFloor> *>(pmy_pack->pdyngr);
    if (ptest_nqt != nullptr) {
      unit_num_dens = RheaUnitNumDens<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                      Primitive::ResetFloor>(pmy_pack);
    } else {
      auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
          Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
          pmy_pack->pdyngr);
      if (ptest_nlog != nullptr) {
        unit_num_dens = RheaUnitNumDens<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                        Primitive::ResetFloor>(pmy_pack);
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl;
        std::cout << "Unsupported EOS type!\n";
        abort();
      }
    }

    TaskStatus tstat = PackRheaInputs(pdrive, stage);
    if (tstat != TaskStatus::complete) return tstat;

    // rhea_f4_in_scratch is allocated at rank CAPACITY (nmb =
    // std::max(nmb_thispack, nmb_maxperrank), radiation_m1.cpp) so AMR regrids never
    // require reallocation, which can exceed the LIVE nmb_thispack for this call (e.g.
    // right after an AMR regrid shrinks the block count on this rank). PackRheaInputs_
    // only fills the leading n_active rows (its
    // par_for ranges m over 0..nmb_thispack-1, and RheaBatchIndex's m-major ordering
    // means those are exactly rows [0, n_active) of the scratch buffer's leading index)
    // -- slice down to that active extent before handing the buffer to Predict(), which
    // accepts any extent(0) <= its capacity (RheaModel::n_capacity_).
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int n_active = pmy_pack->nmb_thispack * indcs.nx1 * indcs.nx2 * indcs.nx3;

    // CONTIGUITY ARGUMENT (do not remove without re-verifying):
    // rhea_f4_in_scratch is Kokkos::View<float****, LayoutRight, ...>. Slicing a
    // Kokkos::pair RANGE on the LEADING index while keeping Kokkos::ALL on every other
    // index carves out a contiguous run of whole "rows" of a LayoutRight (row-major)
    // view's linear storage, since the leading index has the LARGEST stride. Kokkos's
    // subview type-deduction is conservative, though (rank/pattern-based, not a runtime
    // contiguity check), and in general could deduce LayoutStride for a subview built
    // from a Kokkos::pair argument even when the actual strides stay contiguous -- which
    // would make RheaModel::Predict's torch::from_blob (given only `sizes`, no explicit
    // strides) silently wrong. This was checked empirically against the exact Kokkos
    // version vendored in this repo (its subview-type-deduction rule lives in
    // View/Kokkos_ViewMapping.hpp), via a standalone probe compiled and linked against
    // this repo's own build's libkokkoscore.a: a leading-range/ALL/ALL/ALL subview of a
    // View<float****,LayoutRight,DevMemSpace> deduces LayoutRight, not LayoutStride, and
    // is directly assignable to Kokkos::View<const float****, LayoutWrapper,
    // DevMemSpace>. The static_assert below pins that fact so a future Kokkos upgrade
    // that changes the deduction rule fails to COMPILE here instead of silently handing
    // Predict() a mis-strided view.
    auto f4_in_active = Kokkos::subview(rhea_f4_in_scratch, std::make_pair(0, n_active),
                                        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    static_assert(
        std::is_same<decltype(f4_in_active)::array_layout, LayoutWrapper>::value,
        "Kokkos::subview over a leading-index range of rhea_f4_in_scratch no longer "
        "deduces LayoutRight -- RheaModel::Predict's torch::from_blob call assumes "
        "contiguous row-major strides from shape alone; either restore contiguity or "
        "construct the argument View explicitly from a raw pointer + extents instead of "
        "relying on Kokkos::subview's deduced type here (see the comment above).");

    // pred must stay alive as a local for the duration of ApplyRheaMixing's kernel
    // (output-side lifetime hazard) -- do not take the Views out of it and let it go out
    // of scope before ApplyRheaMixing runs.
    RheaModel::Prediction pred = prhea->Predict(f4_in_active);
    TaskStatus astat = ApplyRheaMixing(pdrive, stage, pred.F4_out, pred.growthrate,
                                       pred.stability, unit_num_dens);
    // Defensive device fence before `pred` (which owns the Torch output buffers the unpack
    // kernel reads) is destroyed at end of scope. ApplyRheaMixing only ENQUEUES its par_for
    // on DevExeSpace(), so on a device backend pred's destructor could otherwise return
    // those buffers to Torch's caching allocator before the kernel has executed. Same-stream
    // caching-allocator ordering makes this very likely safe already (buffers are freed to,
    // and any reuse is serialized on, the single DevExeSpace() stream) -- this fence removes
    // the remaining doubt and future-proofs against a second stream ever being introduced.
    // No-op on CPU (Serial/OpenMP block already). If GPU profiling later shows this fence
    // matters, re-evaluate it against the same-stream-ordering argument before removing.
    Kokkos::fence();
    return astat;
  }
#endif

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &u0_      = u0;
  auto &nvars_   = nvars;
  auto &nspecies_ = nspecies;
  auto &params_  = params;

  // Access fluid density (for density threshold guard)
  DvceArray5D<Real> w0_ = w0;
  if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
  }
  bool ismhd_    = ismhd;
  bool ishydro_  = ishydro;

  Real dt = pmy_pack->pmesh->dt;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  // Capture mixing type as int for Kokkos lambda
  int mix_type = static_cast<int>(params.flavor_mix_type);

  par_for(
      "radiation_m1_flavor_mix", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {

        // Optional density threshold: skip cells below rho floor
        if (params_.flavor_mix_rho > 0.0 && (ismhd_ || ishydro_)) {
          if (w0_(m, IDN, k, j, i) > params_.flavor_mix_rho) return;
        }

        // -----------------------------------------------------------------------
        // [A] Build metric (needed for apply_floor and BGK lapse factor)
        // -----------------------------------------------------------------------
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

        // -----------------------------------------------------------------------
        // [B] Load radiation variables for all species; apply causal floor
        // -----------------------------------------------------------------------
        Real nn[4], E[4], Fx[4], Fy[4], Fz[4];
        for (int s = 0; s < nspecies_; ++s) {
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

        // -----------------------------------------------------------------------
        // [C] Compute mixing invariants
        // -----------------------------------------------------------------------
        const Real N_tot = nn[0] + nn[1] + nn[2] + nn[3];
        const Real Ne    = nn[0] - nn[1];  // electron lepton number asymmetry
        const Real Nx    = nn[2] - nn[3];  // x-flavor asymmetry

        // -----------------------------------------------------------------------
        // [D] Compute target mixed number densities
        // -----------------------------------------------------------------------
        Real nn_mm[4];
        if (mix_type == static_cast<int>(FlavMixEquilibrium)) {
          // Equilibrium solution: maximizes entropy subject to N, Ne, Nx conservation
          // See arXiv:2307.16793 eq. (A1)
          const Real disc = Kokkos::sqrt(4.0*N_tot*N_tot + 12.0*Ne*Ne - 3.0*Nx*Nx);
          nn_mm[0] = -N_tot/6.0 + Ne/2.0 + disc/6.0;
          nn_mm[1] =  nn_mm[0] - Ne;
          nn_mm[2] =  0.5*(N_tot + Ne + Nx) - nn_mm[0];
          nn_mm[3] =  nn_mm[2] - Nx;
        } else {
          // Maximal mixing: equal partition within each lepton-number sector
          nn_mm[0] = N_tot/6.0 + Ne/2.0;
          nn_mm[1] = N_tot/6.0 - Ne/2.0;
          nn_mm[2] = N_tot/3.0 + Nx/2.0;
          nn_mm[3] = N_tot/3.0 - Nx/2.0;
        }

        // -----------------------------------------------------------------------
        // [E] Enforce positivity via alpha blending
        //     Find the smallest alpha in [0,1] such that
        //     nn_mix[s] = alpha*nn[s] + (1-alpha)*nn_mm[s] >= 0 for all s.
        //     Two-pass: first handle nu_e/nu_x sector, then nu_ebar/nu_xbar.
        // -----------------------------------------------------------------------
        Real alpha_blend = 0.0;

        // Pass 1: neutrino sector (nu_e and nu_x)
        if (nn_mm[0] < 0.0)
          alpha_blend = Kokkos::max(alpha_blend, -nn_mm[0] / (nn[0] - nn_mm[0]));
        if (nn_mm[2] < 0.0)
          alpha_blend = Kokkos::max(alpha_blend, -nn_mm[2] / (nn[2] - nn_mm[2]));
        alpha_blend = Kokkos::min(1.0, alpha_blend);

        Real nn_mix[4];
        nn_mix[0] = alpha_blend * nn[0] + (1.0 - alpha_blend) * nn_mm[0];
        nn_mix[2] = alpha_blend * nn[2] + (1.0 - alpha_blend) * nn_mm[2];

        // Pass 2: antineutrino sector (nu_ebar and nu_xbar) — may increase alpha
        if (nn_mm[1] < 0.0)
          alpha_blend = Kokkos::max(alpha_blend, -nn_mm[1] / (nn[1] - nn_mm[1]));
        if (nn_mm[3] < 0.0)
          alpha_blend = Kokkos::max(alpha_blend, -nn_mm[3] / (nn[3] - nn_mm[3]));
        alpha_blend = Kokkos::min(1.0, alpha_blend);

        nn_mix[1] = alpha_blend * nn[1] + (1.0 - alpha_blend) * nn_mm[1];
        nn_mix[3] = alpha_blend * nn[3] + (1.0 - alpha_blend) * nn_mm[3];

        for (int s = 0; s < nspecies_; ++s)
          nn_mix[s] = Kokkos::max(params_.rad_N_floor, nn_mix[s]);

        // -----------------------------------------------------------------------
        // [F] Build column-stochastic transition matrix Y[f][g] = P(g -> f)
        //     Mixing channels: nu_e <-> nu_x,  nu_ebar <-> nu_xbar
        //     Each column sums to 1, so total number and energy are conserved.
        // -----------------------------------------------------------------------
        Real Y[4][4] = {};
        Y[0][0] = Kokkos::min(1.0, nn_mix[0] / nn[0]);  // nu_e   stays nu_e
        Y[1][1] = Kokkos::min(1.0, nn_mix[1] / nn[1]);  // nu_ebar stays nu_ebar
        Y[2][2] = Kokkos::min(1.0, nn_mix[2] / nn[2]);  // nu_x   stays nu_x
        Y[3][3] = Kokkos::min(1.0, nn_mix[3] / nn[3]);  // nu_xbar stays nu_xbar

        Y[2][0] = 1.0 - Y[0][0];  // nu_e   -> nu_x
        Y[0][2] = 1.0 - Y[2][2];  // nu_x   -> nu_e
        Y[3][1] = 1.0 - Y[1][1];  // nu_ebar -> nu_xbar
        Y[1][3] = 1.0 - Y[3][3];  // nu_xbar -> nu_ebar
        // All other off-diagonals remain 0 (no nu <-> nubar mixing)

        // -----------------------------------------------------------------------
        // [G] Apply transition matrix to energy densities and fluxes
        // -----------------------------------------------------------------------
        Real E_mix[4]  = {}, Fx_mix[4] = {}, Fy_mix[4] = {}, Fz_mix[4] = {};
        for (int f = 0; f < nspecies_; ++f) {
          for (int g = 0; g < nspecies_; ++g) {
            E_mix[f]  += Y[f][g] * E[g];
            Fx_mix[f] += Y[f][g] * Fx[g];
            Fy_mix[f] += Y[f][g] * Fy[g];
            Fz_mix[f] += Y[f][g] * Fz[g];
          }
          E_mix[f] = Kokkos::max(params_.rad_E_floor, E_mix[f]);
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          pack_F_d(betax, betay, betaz, Fx_mix[f], Fy_mix[f], Fz_mix[f], F_d);
          apply_floor(g_uu, E_mix[f], F_d, params_);
          Fx_mix[f] = F_d(1); Fy_mix[f] = F_d(2); Fz_mix[f] = F_d(3);
        }

        // -----------------------------------------------------------------------
        // [H] Apply BGK relaxation and write back
        //     Q_new = lam * Q_old + (1 - lam) * Q_mix
        //     lam = exp(-dt * alp * inv_tau)
        //     inv_tau = 0 => lam = 1 (no change)
        //     inv_tau -> inf => lam = 0 (instant mixing)
        // -----------------------------------------------------------------------
        const Real lam0 = Kokkos::exp(-dt * alp * params_.bgk_inv_tau_0);
        const Real lam1 = Kokkos::exp(-dt * alp * params_.bgk_inv_tau_1);

        for (int s = 0; s < nspecies_; ++s) {
          u0_(m, CombinedIdx(s, M1_N_IDX,  nvars_), k, j, i) =
              lam0 * u0_(m, CombinedIdx(s, M1_N_IDX,  nvars_), k, j, i)
              + (1.0 - lam0) * nn_mix[s];
          u0_(m, CombinedIdx(s, M1_E_IDX,  nvars_), k, j, i) =
              lam1 * u0_(m, CombinedIdx(s, M1_E_IDX,  nvars_), k, j, i)
              + (1.0 - lam1) * E_mix[s];
          u0_(m, CombinedIdx(s, M1_FX_IDX, nvars_), k, j, i) =
              lam1 * u0_(m, CombinedIdx(s, M1_FX_IDX, nvars_), k, j, i)
              + (1.0 - lam1) * Fx_mix[s];
          u0_(m, CombinedIdx(s, M1_FY_IDX, nvars_), k, j, i) =
              lam1 * u0_(m, CombinedIdx(s, M1_FY_IDX, nvars_), k, j, i)
              + (1.0 - lam1) * Fy_mix[s];
          u0_(m, CombinedIdx(s, M1_FZ_IDX, nvars_), k, j, i) =
              lam1 * u0_(m, CombinedIdx(s, M1_FZ_IDX, nvars_), k, j, i)
              + (1.0 - lam1) * Fz_mix[s];
        }
      });  // par_for

  return TaskStatus::complete;
}

}  // namespace radiationm1
