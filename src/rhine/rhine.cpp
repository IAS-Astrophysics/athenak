//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rhine.cpp
//! \brief RHINE r-process heating + composition source term coupled to the transition EOS.
//!
//! The composition is the transition EOS's 7 mass-fraction scalars
//! (Ye, Xn, Xp, Xa, Xh, Ah, E_B). Where the EOS is in full NSE (transition
//! weight w >= 1) the composition is re-synchronized with the table; elsewhere
//! the RHINE network is evaluated and (in apply mode) back-reacts. Heating is
//! implicit: the mass-excess scalar E_B feeds eps through the EOS c2p; only the
//! neutrino loss is an explicit energy/momentum sink. Mirrors the GR-Athena++
//! TransitionNetwork driver.

#include <cstdlib>
#include <iostream>
#include <string>

#include "rhine.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "driver/driver.hpp"
#include "units/units.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/eos_transition.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/reset_floor_transition.hpp"
#include "eos/primitive-solver/logs.hpp"

namespace rhine {

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real SqrtGammaAlphaV(...)
//! \brief sqrt(gamma) * alpha * v^dir at cell (m,k,j,i), the flux argument of the exact
//!        expansion rate. v^dir is the Eulerian velocity, w0(IVX+dir)/W.
template<class ADMVars>
KOKKOS_INLINE_FUNCTION
Real SqrtGammaAlphaV(const ADMVars &adm, const DvceArray5D<Real> &w0,
                     int m, int k, int j, int i, int dir) {
  const Real g11 = adm.g_dd(m,0,0,k,j,i), g12 = adm.g_dd(m,0,1,k,j,i);
  const Real g13 = adm.g_dd(m,0,2,k,j,i), g22 = adm.g_dd(m,1,1,k,j,i);
  const Real g23 = adm.g_dd(m,1,2,k,j,i), g33 = adm.g_dd(m,2,2,k,j,i);
  const Real sdetg = Kokkos::sqrt(adm::SpatialDet(g11, g12, g13, g22, g23, g33));

  const Real wv1 = w0(m, IVX, k, j, i);
  const Real wv2 = w0(m, IVY, k, j, i);
  const Real wv3 = w0(m, IVZ, k, j, i);
  const Real Wvsq = g11*wv1*wv1 + g22*wv2*wv2 + g33*wv3*wv3
                  + 2.0*(g12*wv1*wv2 + g13*wv1*wv3 + g23*wv2*wv3);
  const Real iW = 1.0 / Kokkos::sqrt(1.0 + Wvsq);

  const Real wv_dir = (dir == 0) ? wv1 : ((dir == 1) ? wv2 : wv3);
  return sdetg * adm.alpha(m, k, j, i) * (wv_dir * iW);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn RHINE::RHINE(MeshBlockPack *ppack, ParameterInput *pin)
//! \brief RHINE constructor: validate the configuration and load the networks.
RHINE::RHINE(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack) {
  bool is_dynamical_relativistic =
      (pin->DoesBlockExist("adm") || pin->DoesBlockExist("z4c"))
      && pin->DoesBlockExist("mhd");
  if (!is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "RHINE requires dyn_grmhd!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pin->DoesBlockExist("units")) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Block <units> required with RHINE!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string eos_string = pin->GetString("mhd", "dyn_eos");
  if (eos_string.compare("transition") != 0) {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__ << std::endl
              << "RHINE needs the transition EOS (dyn_eos = transition)!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmhd->nscalars < 7) {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__ << std::endl
              << "RHINE needs >= 7 scalars (Ye,Xn,Xp,Xa,Xh,Ah,E_B)!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pmode   = pin->GetOrAddInteger("rhine", "pmode", 1);
  apply   = pin->GetOrAddBoolean("rhine", "apply", true);
  use_nqt = pin->GetOrAddBoolean("mhd", "use_NQT", false);
  std::string models_path = pin->GetString("rhine", "models_path");
  nets.InitFromFiles(models_path);

  // Diagnostic array.
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nc1 = indcs.nx1 + 2*(indcs.ng);
  int nc2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nc3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(aux, nmb, N_RHINE_AUX, nc3, nc2, nc1);
  Kokkos::realloc(r0, nmb, N_RHINE_SCALARS, nc3, nc2, nc1);
}

//----------------------------------------------------------------------------------------
//! \fn RHINE::~RHINE()
RHINE::~RHINE() {}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RHINE::AddSources(Driver *pdrive, int stage)
//! \brief Per-substep source pass. Freezes the '0' reference at stage 1.
TaskStatus RHINE::AddSources(Driver *pdrive, int stage) {
  if (!apply) { return TaskStatus::complete; }  // diagnostics come from PostStep
  return Dispatch(pdrive, stage, RhinePass::source);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RHINE::PostStep(Driver *pdrive, int stage)
//! \brief Once-per-step pass; a no-op except after the final RK stage.
TaskStatus RHINE::PostStep(Driver *pdrive, int stage) {
  if (stage != pdrive->nexp_stages) { return TaskStatus::complete; }
  return Dispatch(pdrive, stage, RhinePass::poststep);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RHINE::Dispatch(Driver *pdrive, int stage, RhinePass pass)
//! \brief Resolve the EOS/error policy template arguments and run the requested pass.
TaskStatus RHINE::Dispatch(Driver *pdrive, int stage, RhinePass pass) {
  using namespace dyngr;      // NOLINT
  using namespace Primitive;  // NOLINT
  if (pmy_pack->pdyngr->eos_policy != DynGRMHD_EOS::eos_transition) {
    return TaskStatus::complete;
  }
  Real dt_apply = 0.0;
  if (pass == RhinePass::source) {
    if (stage == 1) { SnapshotReference(); }
    dt_apply = (pdrive->beta[stage-1]) * (pmy_pack->pmesh->dt);
  }
  if (pmy_pack->pdyngr->error_policy == DynGRMHD_Error::reset_floor) {
    return use_nqt
        ? RunPassEOS<EOSTransition<NQTLogs>, ResetFloor>(pass, dt_apply)
        : RunPassEOS<EOSTransition<NormalLogs>, ResetFloor>(pass, dt_apply);
  }
  return use_nqt
      ? RunPassEOS<EOSTransition<NQTLogs>, ResetFloorTransition>(pass, dt_apply)
      : RunPassEOS<EOSTransition<NormalLogs>, ResetFloorTransition>(pass, dt_apply);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RHINE::RunPassEOS(RhinePass pass, Real dt_apply_code)
//! \brief Resync first, then evaluate the network (GR-Athena++ ordering).
template<class EOSPolicy, class ErrorPolicy>
TaskStatus RHINE::RunPassEOS(RhinePass pass, Real dt_apply_code) {
  if (pass == RhinePass::poststep) { NSEResyncEOS<EOSPolicy, ErrorPolicy>(); }
  return NetworkStepEOS<EOSPolicy, ErrorPolicy>(dt_apply_code);
}

//----------------------------------------------------------------------------------------
//! \fn void RHINE::SnapshotReference()
//! \brief Freeze w0's composition scalars as the network's '0' reference for this step.
void RHINE::SnapshotReference() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &w0 = pmy_pack->pmhd->w0;
  auto &r0_ = r0;

  par_for("rhine_r0", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int l = 0; l < N_RHINE_SCALARS; ++l) {
      r0_(m, l, k, j, i) = w0(m, I_YE + l, k, j, i);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RHINE::NSEResyncEOS()
//! \brief Re-synchronize the advected composition of NSE cells (w >= 1) with the table.
//!
//! Full block INCLUDING ghosts, as in GR-Athena++: an interior-only resync leaves ghost
//! scalars stale relative to the neighbour's resynced interior, and AthenaK's FOFC mask
//! reads those ghosts on the next step -- flipping the fallback decision on one side of a
//! shared face only, i.e. unequal fluxes and a mass leak. Runs once per step after the
//! final C2P, so w0/temperature are fresh everywhere; keeping it out of the substep path
//! is also what protects the frozen r0 reference.
template<class EOSPolicy, class ErrorPolicy>
TaskStatus RHINE::NSEResyncEOS() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;

  auto &w0   = pmy_pack->pmhd->w0;
  auto &u0   = pmy_pack->pmhd->u0;
  auto &temp = pmy_pack->pdyngr->temperature;
  auto eos = static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy>*>(
                 pmy_pack->pdyngr)->eos.ps.GetEOS();

  par_for("rhine_nse", DevExeSpace(), 0, pmy_pack->nmb_thispack-1,
          0, n3m1, 0, n2m1, 0, n1m1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real n_fm3 = w0(m, IDN, k, j, i) / eos.GetBaryonMass();
    const Real T = temp(m, 0, k, j, i);
    if (eos.GetTransitionFactor(n_fm3, T) < 1.0) { return; }

    Real Y[MAX_SPECIES] = {0.0};
    for (int l = 0; l < N_RHINE_SCALARS; ++l) { Y[SCYE + l] = w0(m, I_YE + l, k, j, i); }

    w0(m, I_XN, k, j, i) = eos.FrYn(n_fm3, T, Y);
    w0(m, I_XP, k, j, i) = eos.FrYp(n_fm3, T, Y);
    w0(m, I_XA, k, j, i) = eos.FrXa(n_fm3, T, Y);
    w0(m, I_XH, k, j, i) = eos.FrXh(n_fm3, T, Y);
    w0(m, I_AH, k, j, i) = eos.AN(n_fm3, T, Y);
    w0(m, I_EB, k, j, i) = eos.GetNSEBindingEnergy(n_fm3, T, Y);

    const Real D = u0(m, IDN, k, j, i);
    u0(m, I_XN, k, j, i) = w0(m, I_XN, k, j, i) * D;
    u0(m, I_XP, k, j, i) = w0(m, I_XP, k, j, i) * D;
    u0(m, I_XA, k, j, i) = w0(m, I_XA, k, j, i) * D;
    u0(m, I_XH, k, j, i) = w0(m, I_XH, k, j, i) * D;
    u0(m, I_AH, k, j, i) = w0(m, I_AH, k, j, i) * D;
    u0(m, I_EB, k, j, i) = w0(m, I_EB, k, j, i) * D;
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RHINE::NetworkStepEOS(Real dt_apply_code)
//! \brief Evaluate RHINE per cell: diagnostics always, back-reaction if dt_apply_code > 0.
template<class EOSPolicy, class ErrorPolicy>
TaskStatus RHINE::NetworkStepEOS(Real dt_apply_code) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  auto &w0   = pmy_pack->pmhd->w0;
  auto &u0   = pmy_pack->pmhd->u0;
  auto &temp = pmy_pack->pdyngr->temperature;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm  = pmy_pack->padm->adm;
  auto &aux_ = aux;
  auto &r0_  = r0;

  // EOS (copy; Kokkos Views inside are device handles).
  auto eos = static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy>*>(
                 pmy_pack->pdyngr)->eos.ps.GetEOS();

  const Real time_cgs = pmy_pack->punit->time_cgs();
  // RHINE's rates are endpoint differences over dt, so the network gets the full
  // step (as in gr-athena) while the sources are applied over the substep.
  const Real dt_code = pmy_pack->pmesh->dt;
  const bool net_on = (dt_code > 0.0);
  // Back-react only in the source pass; there the frozen r0 is the '0' reference.
  const bool apply_ = apply && (dt_apply_code > 0.0);
  const bool use_r0 = apply_;

  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  RhineNets nets_ = nets;
  const int  pmode_ = pmode;

  constexpr Real MEV_TO_ERG = 1.602176634e-6;    // erg per MeV
  constexpr Real M_U_G      = 1.66053906660e-24; // atomic mass unit [g]
  constexpr Real M_U_MEV    = 931.49410242;      // atomic mass unit [MeV]

  par_for("rhine_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real mb_conv = eos.GetBaryonMass();     // combined mass+density factor
    const Real mb_MeV  = eos.GetBaryonMassMeV();   // raw baryon mass [MeV]
    const Real n_fm3   = w0(m, IDN, k, j, i) / mb_conv;
    const Real T       = temp(m, 0, k, j, i);      // MeV

    Real Y[MAX_SPECIES] = {0.0};
    Y[SCYE] = w0(m, I_YE, k, j, i);
    Y[SCXN] = w0(m, I_XN, k, j, i);
    Y[SCXP] = w0(m, I_XP, k, j, i);
    Y[SCXA] = w0(m, I_XA, k, j, i);
    Y[SCXH] = w0(m, I_XH, k, j, i);
    Y[SCAH] = w0(m, I_AH, k, j, i);
    Y[SCEB] = w0(m, I_EB, k, j, i);

    // Always-on diagnostics.
    const Real w = eos.GetTransitionFactor(n_fm3, T);
    aux_(m, A_TRANS, k, j, i) = w;
    aux_(m, A_XERR, k, j, i) = Y[SCXN] + Y[SCXP] + Y[SCXA] + Y[SCXH] - 1.0;

    // Zero the rate diagnostics by default.
    aux_(m, A_HEAT, k, j, i) = 0.0;
    aux_(m, A_FNU,  k, j, i) = 0.0;
    aux_(m, A_QDOT, k, j, i) = 0.0;
    aux_(m, A_LNU,  k, j, i) = 0.0;
    aux_(m, A_DYE,  k, j, i) = 0.0;
    aux_(m, A_DYN,  k, j, i) = 0.0;
    aux_(m, A_DYP,  k, j, i) = 0.0;
    aux_(m, A_DYA,  k, j, i) = 0.0;
    aux_(m, A_DYH,  k, j, i) = 0.0;
    aux_(m, A_DAH,  k, j, i) = 0.0;
    aux_(m, A_DMA,  k, j, i) = 0.0;

    const Real D = u0(m, IDN, k, j, i);

    // --- Full-NSE interior (w >= 1): no network sources. The table resync runs once
    //     per step in NSEResyncEOS; doing it here would overwrite registers between
    //     stage combinations and invalidate the frozen r0 reference.
    if (w >= 1.0) { return; }

    // --- Out-of-NSE (w < 1): evaluate the network.
    if (!net_on || !(T > 0.0)) { return; }
    // Sanitize + convert mass fractions to the network's per-baryon abundances.
    Real Y_s[MAX_SPECIES];
    eos.GetSanitizedMassFractions(Y, Y_s);
    const Real ye = Y_s[SCYE];
    const Real ah = Y_s[SCAH];
    Real yn = Y_s[SCXN];
    Real ya = 0.25 * Y_s[SCXA];
    Real yh = (ah > 0.0) ? Y_s[SCXH] / ah : 0.0;
    const Real s_max = 1.0 - 1e-12;
    const Real s_b   = yn + 4.0*ya + ah*yh;
    if (s_b > s_max) {
      const Real f = s_max / s_b;
      yn *= f; ya *= f; yh *= f;
    }
    const Real g11 = adm.g_dd(m,0,0,k,j,i), g12 = adm.g_dd(m,0,1,k,j,i);
    const Real g13 = adm.g_dd(m,0,2,k,j,i), g22 = adm.g_dd(m,1,1,k,j,i);
    const Real g23 = adm.g_dd(m,1,2,k,j,i), g33 = adm.g_dd(m,2,2,k,j,i);
    const Real wv1 = w0(m, IVX, k, j, i);
    const Real wv2 = w0(m, IVY, k, j, i);
    const Real wv3 = w0(m, IVZ, k, j, i);
    const Real Wvsq = g11*wv1*wv1 + g22*wv2*wv2 + g33*wv3*wv3
                    + 2.0*(g12*wv1*wv2 + g13*wv1*wv3 + g23*wv2*wv3);
    const Real Wlor = Kokkos::sqrt(1.0 + Wvsq);
    const Real alpha = adm.alpha(m, k, j, i);
    const Real dt_net = dt_code * (alpha / Wlor) * time_cgs;  // full-step proper time [s]
    const Real dt_ap  = dt_apply_code * (alpha / Wlor) * time_cgs;  // substep proper [s]

    // Expansion rate D ln(rho)/Dtau = -div(u) [1/s]. Exact up to O(v^2 d_t):
    //   W*trK - (W/(alpha*sqrt(gamma))) d_i(sqrt(gamma) alpha v^i),
    // obtained by eliminating d_t(sqrt(gamma)) via d_t ln sqrt(gamma) = -alpha trK + D_i
    // beta^i, which also cancels the shift. Reduces to -d_i v^i in flat space at rest, and
    // to trK for static matter. RHINE clamps |drho| to >= 0.3/s internally.
    Real div = (SqrtGammaAlphaV(adm, w0, m, k, j, i+1, 0)
              - SqrtGammaAlphaV(adm, w0, m, k, j, i-1, 0)) * 0.5 / size.d_view(m).dx1;
    if (multi_d) {
      div += (SqrtGammaAlphaV(adm, w0, m, k, j+1, i, 1)
            - SqrtGammaAlphaV(adm, w0, m, k, j-1, i, 1)) * 0.5 / size.d_view(m).dx2;
    }
    if (three_d) {
      div += (SqrtGammaAlphaV(adm, w0, m, k+1, j, i, 2)
            - SqrtGammaAlphaV(adm, w0, m, k-1, j, i, 2)) * 0.5 / size.d_view(m).dx3;
    }
    const Real detg = adm::SpatialDet(g11, g12, g13, g22, g23, g33);
    Real uxx, uxy, uxz, uyy, uyz, uzz;
    adm::SpatialInv(1.0/detg, g11, g12, g13, g22, g23, g33,
                    &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
    const Real trK = uxx*adm.vK_dd(m,0,0,k,j,i) + uyy*adm.vK_dd(m,1,1,k,j,i)
                   + uzz*adm.vK_dd(m,2,2,k,j,i)
                   + 2.0*(uxy*adm.vK_dd(m,0,1,k,j,i) + uxz*adm.vK_dd(m,0,2,k,j,i)
                        + uyz*adm.vK_dd(m,1,2,k,j,i));
    const Real drho = (Wlor*trK - (Wlor/(alpha*Kokkos::sqrt(detg))) * div) / time_cgs;

    const Real n_cm3   = n_fm3 * 1e39;
    const Real rho_cgs = M_U_G * n_cm3;                  // RHINE input [g/cm^3]

    Real ye0 = ye, yn0 = yn, ya0 = ya, yh0 = yh, ah0 = ah;
    Real mass0 = mb_MeV*(1.0 + Y_s[SCEB]) - M_U_MEV;  // mass excess [MeV/baryon]
    if (use_r0) {
      Real Y0[MAX_SPECIES] = {0.0};
      Real Y0_s[MAX_SPECIES];
      for (int l = 0; l < N_RHINE_SCALARS; ++l) { Y0[SCYE + l] = r0_(m, l, k, j, i); }
      eos.GetSanitizedMassFractions(Y0, Y0_s);
      // Floored atmosphere at step start carries Ah=1, outside the network's [4,300]:
      // keep the current-state reference for those cells.
      if (Y0_s[SCAH] >= 4.0) {
        ye0 = Y0_s[SCYE];
        ah0 = Y0_s[SCAH];
        yn0 = Y0_s[SCXN];
        ya0 = 0.25 * Y0_s[SCXA];
        yh0 = (ah0 > 0.0) ? Y0_s[SCXH] / ah0 : 0.0;
        const Real s_b0 = yn0 + 4.0*ya0 + ah0*yh0;
        if (s_b0 > s_max) {
          const Real f0 = s_max / s_b0;
          yn0 *= f0; ya0 *= f0; yh0 *= f0;
        }
        mass0 = mb_MeV*(1.0 + Y0_s[SCEB]) - M_U_MEV;
      }
    }

    Real dye, dyn, dyp, dya, dyh, dah, dma, fnu;
    nets_.run(rho_cgs, T, ye, yn, ya, yh, ah, drho, dt_net,
              ye0, yn0, ya0, yh0, ah0, mass0,
              dye, dyn, dyp, dya, dyh, dah, dma, fnu, pmode_);

    if (!(Kokkos::isfinite(dye) && Kokkos::isfinite(dyn) && Kokkos::isfinite(dyp) &&
          Kokkos::isfinite(dya) && Kokkos::isfinite(dyh) && Kokkos::isfinite(dah) &&
          Kokkos::isfinite(dma) && Kokkos::isfinite(fnu))) {
      dye = dyn = dyp = dya = dyh = dah = dma = fnu = 0.0;
    }
    // fnu is only valid while the composition releases energy (dma < 0).
    if (!(dma < 0.0)) { fnu = 0.0; }

    // Diagnostics.
    aux_(m, A_HEAT, k, j, i) = -(1.0 - fnu) * dma * MEV_TO_ERG * n_cm3;
    aux_(m, A_FNU,  k, j, i) = fnu;
    aux_(m, A_DYE,  k, j, i) = dye;
    aux_(m, A_DYN,  k, j, i) = dyn;
    aux_(m, A_DYP,  k, j, i) = dyp;
    aux_(m, A_DYA,  k, j, i) = dya;
    aux_(m, A_DYH,  k, j, i) = dyh;
    aux_(m, A_DAH,  k, j, i) = dah;
    aux_(m, A_DMA,  k, j, i) = dma;
    const Real rate = D * (-dma / mb_MeV) * alpha * time_cgs;  // densitized code rate
    aux_(m, A_QDOT, k, j, i) = (1.0 - fnu) * rate;
    aux_(m, A_LNU,  k, j, i) = fnu * rate;

    if (!apply_) { return; }

    // --- Apply. Heating is implicit (E_B feeds eps via c2p); the neutrino loss
    //     is an explicit tau/momentum sink.
    // (Xh1 - Xh0)/dt_net; must use the '0' reference for the baryon-number identity
    // dyn + dyp + 4 dya + dxh = 0 to hold.
    const Real dxh = dah*yh0 + ah0*dyh + dt_net*dah*dyh;
    u0(m, I_XN, k, j, i) += D * dyn * dt_ap;
    u0(m, I_XP, k, j, i) += D * dyp * dt_ap;
    u0(m, I_XA, k, j, i) += D * 4.0 * dya * dt_ap;
    u0(m, I_XH, k, j, i) += D * dxh * dt_ap;
    u0(m, I_YE, k, j, i) += D * dye * dt_ap;
    u0(m, I_AH, k, j, i) += D * dah * dt_ap;
    u0(m, I_EB, k, j, i) += D * (dma / mb_MeV) * dt_ap;

    // Neutrino energy sink (no 1/W; four-force tau projection).
    u0(m, IEN, k, j, i) += D * fnu * (dma / mb_MeV) * (dt_apply_code * alpha * time_cgs);
    // Neutrino momentum sink (proper-time factor).
    const Real snu = D * fnu * (dma / mb_MeV) * dt_ap;
    const Real u_d_1 = g11*wv1 + g12*wv2 + g13*wv3;
    const Real u_d_2 = g12*wv1 + g22*wv2 + g23*wv3;
    const Real u_d_3 = g13*wv1 + g23*wv2 + g33*wv3;
    u0(m, IM1, k, j, i) += snu * u_d_1;
    u0(m, IM2, k, j, i) += snu * u_d_2;
    u0(m, IM3, k, j, i) += snu * u_d_3;

    // Conservative repair of any negative species landing (transfer to largest).
    const int sp[4] = {I_XN, I_XP, I_XA, I_XH};
    Real Xs[4];
    int lmax = 0;
    bool neg = false;
    for (int l = 0; l < 4; ++l) {
      Xs[l] = u0(m, sp[l], k, j, i);
      if (Xs[l] < 0.0) { neg = true; }
      if (Xs[l] > Xs[lmax]) { lmax = l; }
    }
    if (neg) {
      for (int l = 0; l < 4; ++l) {
        if (Xs[l] < 0.0) { Xs[lmax] += Xs[l]; Xs[l] = 0.0; }
      }
      for (int l = 0; l < 4; ++l) { u0(m, sp[l], k, j, i) = Xs[l]; }
    }
  });

  return TaskStatus::complete;
}

}  // namespace rhine
