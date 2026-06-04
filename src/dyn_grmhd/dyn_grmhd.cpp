//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr.cpp
//! \brief implementation of functions for DynGRMHD and DynGRMHDPS controlling the task
//! list

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "z4c/tmunu.hpp"
#include "dyn_grmhd.hpp"
#include "tasklist/numerical_relativity.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {
namespace {

bool C2PDebugEnabled() {
  static bool enabled = (std::getenv("ATHENA_C2P_DEBUG") != nullptr ||
                         std::getenv("ATHENA_SYM_DEBUG") != nullptr);
  return enabled;
}

bool CoordSrcDebugEnabled() {
  static bool enabled = (std::getenv("ATHENA_COORDSRC_DEBUG") != nullptr ||
                         std::getenv("ATHENA_SYM_DEBUG") != nullptr);
  return enabled;
}

Real EnvReal(const char *name, Real default_value) {
  const char *value = std::getenv(name);
  return (value == nullptr) ? default_value : static_cast<Real>(std::atof(value));
}

int EnvInt(const char *name, int default_value) {
  const char *value = std::getenv(name);
  return (value == nullptr) ? default_value : std::atoi(value);
}

void C2PDebugProbe(MeshBlockPack *ppack, const char *label, Driver *pdrive, int stage) {
  if (!C2PDebugEnabled()) return;
  if (pdrive == nullptr || ppack == nullptr || ppack->pmhd == nullptr ||
      ppack->padm == nullptr) {
    return;
  }

  Mesh *pm = ppack->pmesh;
  const int cycle = pm->ncycle;
  const int target_cycle = EnvInt("ATHENA_C2P_DEBUG_CYCLE", -1);
  if (target_cycle >= 0) {
    if (cycle != target_cycle) return;
  } else if (!(cycle < 3 || cycle == 80 || cycle == 160)) {
    return;
  }
  const int target_stage = EnvInt("ATHENA_C2P_DEBUG_STAGE", -1);
  if (target_stage >= 0 && stage != target_stage) return;

  auto &indcs = pm->mb_indcs;
  auto &size = ppack->pmb->mb_size;
  auto &gid = ppack->pmb->mb_gid;
  auto &lev = ppack->pmb->mb_lev;
  const Real x_target = EnvReal("ATHENA_SYM_X_TARGET", 20.0);
  const Real z_target = EnvReal("ATHENA_SYM_Z_TARGET", 0.0);

  for (int m = 0; m < ppack->nmb_thispack; ++m) {
    const auto &mb = size.h_view(m);
    const bool x_inside = (x_target >= mb.x1min) && (x_target < mb.x1max);
    const bool z_inside = (z_target >= mb.x3min) && (z_target < mb.x3max);
    if (!(x_inside && z_inside)) continue;

    int i = indcs.is + static_cast<int>(std::floor((x_target - mb.x1min)/mb.dx1));
    i = std::max(indcs.is, std::min(indcs.ie, i));
    int k = indcs.ks + static_cast<int>(std::floor((z_target - mb.x3min)/mb.dx3));
    k = std::max(indcs.ks, std::min(indcs.ke, k));

    for (int side = 0; side < 2; ++side) {
      const bool below = (side == 0);
      int j = -1;
      Real best_abs_y = std::numeric_limits<Real>::max();
      for (int jj = indcs.js; jj <= indcs.je; ++jj) {
        const Real y = mb.x2min + (static_cast<Real>(jj - indcs.js) + 0.5)*mb.dx2;
        if ((below && y < 0.0) || (!below && y > 0.0)) {
          if (std::abs(y) < best_abs_y) {
            best_abs_y = std::abs(y);
            j = jj;
          }
        }
      }
      if (j < 0) continue;

      constexpr int ndiag = 36;
      DvceArray1D<Real> diag("dyngr-c2p-debug", ndiag);
      auto u = ppack->pmhd->u0;
      auto w = ppack->pmhd->w0;
      auto bcc = ppack->pmhd->bcc0;
      auto bfc = ppack->pmhd->b0;
      auto adm = ppack->padm->adm;
      Kokkos::parallel_for("dyngr_c2p_debug", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int) {
        const Real gxx = adm.g_dd(m, 0, 0, k, j, i);
        const Real gxy = adm.g_dd(m, 0, 1, k, j, i);
        const Real gxz = adm.g_dd(m, 0, 2, k, j, i);
        const Real gyy = adm.g_dd(m, 1, 1, k, j, i);
        const Real gyz = adm.g_dd(m, 1, 2, k, j, i);
        const Real gzz = adm.g_dd(m, 2, 2, k, j, i);
        const Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
        const Real sdetg = sqrt(detg);
        diag(0) = u(m, IDN, k, j, i);
        diag(1) = u(m, IM1, k, j, i);
        diag(2) = u(m, IM2, k, j, i);
        diag(3) = u(m, IM3, k, j, i);
        diag(4) = u(m, IEN, k, j, i);
        diag(5) = w(m, IDN, k, j, i);
        diag(6) = w(m, IVX, k, j, i);
        diag(7) = w(m, IVY, k, j, i);
        diag(8) = w(m, IVZ, k, j, i);
        diag(9) = w(m, IPR, k, j, i);
        diag(10) = bcc(m, IBX, k, j, i);
        diag(11) = bcc(m, IBY, k, j, i);
        diag(12) = bcc(m, IBZ, k, j, i);
        diag(13) = bfc.x1f(m, k, j, i);
        diag(14) = bfc.x1f(m, k, j, i + 1);
        diag(15) = bfc.x2f(m, k, j, i);
        diag(16) = bfc.x2f(m, k, j + 1, i);
        diag(17) = bfc.x3f(m, k, j, i);
        diag(18) = bfc.x3f(m, k + 1, j, i);
        diag(19) = adm.alpha(m, k, j, i);
        diag(20) = adm.beta_u(m, 0, k, j, i);
        diag(21) = adm.beta_u(m, 1, k, j, i);
        diag(22) = adm.beta_u(m, 2, k, j, i);
        diag(23) = gxx;
        diag(24) = gxy;
        diag(25) = gxz;
        diag(26) = gyy;
        diag(27) = gyz;
        diag(28) = gzz;
        diag(29) = detg;
        diag(30) = sdetg;
        diag(31) = u(m, IDN, k, j, i)/sdetg;
        diag(32) = u(m, IM1, k, j, i)/sdetg;
        diag(33) = u(m, IM2, k, j, i)/sdetg;
        diag(34) = u(m, IM3, k, j, i)/sdetg;
        diag(35) = u(m, IEN, k, j, i)/sdetg;
      });
      Kokkos::fence();
      auto hdiag = Kokkos::create_mirror_view(diag);
      Kokkos::deep_copy(hdiag, diag);

      std::cout << std::setprecision(17)
                << "C2PDBG label=" << label
                << " rank=" << global_variable::my_rank
                << " gid=" << gid.h_view(m)
                << " level=" << lev.h_view(m)
                << " side=" << (below ? "below" : "above")
                << " cycle=" << cycle
                << " time=" << pm->time
                << " stage=" << stage
                << " i=" << i
                << " j=" << j
                << " k=" << k
                << " x=" << (mb.x1min + (static_cast<Real>(i - indcs.is) + 0.5)*mb.dx1)
                << " y=" << (mb.x2min + (static_cast<Real>(j - indcs.js) + 0.5)*mb.dx2)
                << " z=" << (mb.x3min + (static_cast<Real>(k - indcs.ks) + 0.5)*mb.dx3)
                << " rho_u=" << hdiag(0)
                << " momx_u=" << hdiag(1)
                << " momy_u=" << hdiag(2)
                << " momz_u=" << hdiag(3)
                << " tau_u=" << hdiag(4)
                << " rho_w=" << hdiag(5)
                << " vx_w=" << hdiag(6)
                << " vy_w=" << hdiag(7)
                << " vz_w=" << hdiag(8)
                << " press_w=" << hdiag(9)
                << " bccx=" << hdiag(10)
                << " bccy=" << hdiag(11)
                << " bccz=" << hdiag(12)
                << " bfcx_lo=" << hdiag(13)
                << " bfcx_hi=" << hdiag(14)
                << " bfcy_lo=" << hdiag(15)
                << " bfcy_hi=" << hdiag(16)
                << " bfcz_lo=" << hdiag(17)
                << " bfcz_hi=" << hdiag(18)
                << " alpha=" << hdiag(19)
                << " beta_x=" << hdiag(20)
                << " beta_y=" << hdiag(21)
                << " beta_z=" << hdiag(22)
                << " gxx=" << hdiag(23)
                << " gxy=" << hdiag(24)
                << " gxz=" << hdiag(25)
                << " gyy=" << hdiag(26)
                << " gyz=" << hdiag(27)
                << " gzz=" << hdiag(28)
                << " detg=" << hdiag(29)
                << " sdetg=" << hdiag(30)
                << " rho_u_over_sdetg=" << hdiag(31)
                << " momx_u_over_sdetg=" << hdiag(32)
                << " momy_u_over_sdetg=" << hdiag(33)
                << " momz_u_over_sdetg=" << hdiag(34)
                << " tau_u_over_sdetg=" << hdiag(35)
                << std::endl;
    }
  }
}

} // namespace

// A dumb template function containing the switch statement needed to select an EOS.
template<class ErrorPolicy>
DynGRMHD* SelectDynGRMHDEOS(MeshBlockPack *ppack, ParameterInput *pin,
                            DynGRMHD_EOS eos_policy) {
  DynGRMHD* dyn_gr = nullptr;
  bool use_NQT = false;
  switch(eos_policy) {
    case DynGRMHD_EOS::eos_ideal:
      dyn_gr = new DynGRMHDPS<Primitive::IdealGas, ErrorPolicy>(ppack, pin);
      break;
    case DynGRMHD_EOS::eos_piecewise_poly:
      dyn_gr = new DynGRMHDPS<Primitive::PiecewisePolytrope, ErrorPolicy>(ppack, pin);
      break;
    case DynGRMHD_EOS::eos_compose:
      use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT",false);
      if (use_NQT) {
        dyn_gr = new DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                ErrorPolicy>(ppack, pin);
      } else {
        dyn_gr = new DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                ErrorPolicy>(ppack, pin);
      }
      break;
    case DynGRMHD_EOS::eos_hybrid:
      use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT",false);
      if (use_NQT) {
        dyn_gr = new DynGRMHDPS<Primitive::EOSHybrid<Primitive::NQTLogs>,
                                ErrorPolicy>(ppack, pin);
      } else {
        dyn_gr = new DynGRMHDPS<Primitive::EOSHybrid<Primitive::NormalLogs>,
                                ErrorPolicy>(ppack, pin);
      }
      break;
  }
  return dyn_gr;
}

DynGRMHD* BuildDynGRMHD(MeshBlockPack *ppack, ParameterInput *pin) {
  std::string eos_string = pin->GetString("mhd", "dyn_eos");
  std::string error_string = pin->GetString("mhd", "dyn_error");
  DynGRMHD_EOS eos_policy;
  DynGRMHD_Error error_policy;

  if (eos_string.compare("ideal") == 0) {
    eos_policy = DynGRMHD_EOS::eos_ideal;
  } else if (eos_string.compare("piecewise_poly") == 0) {
    eos_policy = DynGRMHD_EOS::eos_piecewise_poly;
  } else if (eos_string.compare("compose") == 0) {
    eos_policy = DynGRMHD_EOS::eos_compose;
  } else if (eos_string.compare("hybrid") == 0) {
    eos_policy = DynGRMHD_EOS::eos_hybrid;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> dyn_eos = '" << eos_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (error_string.compare("reset_floor") == 0) {
    error_policy = DynGRMHD_Error::reset_floor;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> dyn_error = '" << error_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  DynGRMHD* dyn_gr = nullptr;

  switch (error_policy) {
    case DynGRMHD_Error::reset_floor:
      dyn_gr = SelectDynGRMHDEOS<Primitive::ResetFloor>(ppack, pin, eos_policy);
      break;
  }

  dyn_gr->eos_policy = eos_policy;
  dyn_gr->error_policy = error_policy;

  return dyn_gr;
}

DynGRMHD::DynGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp),
    temperature("temperature",1,1,1,1,1) {
  std::string rsolver = pin->GetString("mhd", "rsolver");
  if (rsolver.compare("llf") == 0) {
    rsolver_method = DynGRMHD_RSolver::llf_dyngr;
  } else if (rsolver.compare("hlle") == 0) {
    rsolver_method = DynGRMHD_RSolver::hlle_dyngr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> rsolver = '" << rsolver
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string fofc = pin->GetOrAddString("mhd", "fofc_method", "llf");
  if (fofc.compare("llf") == 0) {
    fofc_method = DynGRMHD_RSolver::llf_dyngr;
  } else if (fofc.compare("hlle") == 0) {
    fofc_method = DynGRMHD_RSolver::hlle_dyngr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> fofc_method = '" << fofc
              << "' not implemented for FOFC" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  scratch_level = pin->GetOrAddInteger("mhd", "dyn_scratch", 0);
  enforce_maximum = pin->GetOrAddBoolean("mhd", "enforce_maximum", true);
  dmp_M = pin->GetOrAddReal("mhd", "dmp_M", 1.2);

  fixed_evolution = pin->GetOrAddBoolean("mhd", "fixed", false);
  zero_tmunu_feedback = pin->GetOrAddBoolean("mhd", "zero_tmunu_feedback", false);
  refresh_tmunu_when_fixed =
      pin->GetOrAddBoolean("mhd", "refresh_tmunu_when_fixed", false);
  dyngr_x3_debug = pin->GetOrAddBoolean("mhd", "dyngr_x3_debug", false);
  dyngr_x3_debug_x = pin->GetOrAddReal("mhd", "dyngr_x3_debug_x", 20.03125);
  dyngr_x3_debug_y_abs = pin->GetOrAddReal("mhd", "dyngr_x3_debug_y_abs", 0.03125);
  dyngr_x3_debug_z_face = pin->GetOrAddReal("mhd", "dyngr_x3_debug_z_face", 0.0);
  dyngr_x3_debug_cycle = pin->GetOrAddInteger("mhd", "dyngr_x3_debug_cycle", -1);
  dyngr_x3_debug_stage = pin->GetOrAddInteger("mhd", "dyngr_x3_debug_stage", -1);

  // allocate memory for temperature
  {
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(temperature, nmb, 1, ncells3, ncells2, ncells1);
  }
}

DynGRMHD::~DynGRMHD() {
}

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::QueueDynGRMHDTasks() {
  using namespace mhd;  // NOLINT(build/namespaces)
  using namespace z4c;  // NOLINT(build/namespaces)
  using namespace numrel; // NOLINT(build/namespaces))
  Z4c *pz4c = pmy_pack->pz4c;
  adm::ADM *padm = pmy_pack->padm;
  MHD *pmhd = pmy_pack->pmhd;
  NumericalRelativity *pnr = pmy_pack->pnr;

  // Start task list
  pnr->QueueTask(&MHD::InitRecv, pmhd, MHD_Recv, "MHD_Recv", Task_Start);

  // Run task list
  pnr->QueueTask(&MHD::CopyCons, pmhd, MHD_CopyU, "MHD_CopyU", Task_Run);

  // Select which CalculateFlux function to add based on rsolver_method.
  // CalcFlux requires metric in flux - must happen before z4ctoadm updates the metric
  if (rsolver_method == DynGRMHD_RSolver::llf_dyngr) {
    pnr->QueueTask(
           &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::llf_dyngr>,
           this, MHD_Flux, "MHD_Flux", Task_Run, {MHD_CopyU});
  } else if (rsolver_method == DynGRMHD_RSolver::hlle_dyngr) {
    pnr->QueueTask(
           &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::hlle_dyngr>,
           this, MHD_Flux, "MHD_Flux", Task_Run, {MHD_CopyU});
  } else { // put more rsolvers here
    abort();
  }

  // Now the rest of the MHD run tasks
  if (pz4c != nullptr) {
    pnr->QueueTask(&DynGRMHD::SetTmunu, this, MHD_SetTmunu, "MHD_SetTmunu",
                   Task_Run, {MHD_CopyU});
  }
  pnr->QueueTask(&MHD::SendFlux, pmhd, MHD_SendFlux, "MHD_SendFlux",
                 Task_Run, {MHD_Flux});
  pnr->QueueTask(&MHD::RecvFlux, pmhd, MHD_RecvFlux, "MHD_RecvFlux",
                 Task_Run, {MHD_SendFlux});
  if (pz4c != nullptr) {
    pnr->QueueTask(&MHD::RKUpdate, pmhd, MHD_ExplRK, "MHD_ExplRK", Task_Run,
                   {MHD_RecvFlux, MHD_SetTmunu});
  } else {
    pnr->QueueTask(&MHD::RKUpdate, pmhd, MHD_ExplRK, "MHD_ExplRK", Task_Run,
                   {MHD_RecvFlux});
  }
  pnr->QueueTask(&MHD::MHDSrcTerms, pmhd, MHD_AddSrc, "MHD_AddSrc", Task_Run,
                 {MHD_ExplRK});
  pnr->QueueTask(&MHD::RestrictU, pmhd, MHD_RestU, "MHD_RestU", Task_Run, {MHD_AddSrc});
  pnr->QueueTask(&MHD::SendU, pmhd, MHD_SendU, "MHD_SendU", Task_Run, {MHD_RestU});
  pnr->QueueTask(&MHD::RecvU, pmhd, MHD_RecvU, "MHD_RecvU", Task_Run, {MHD_SendU});
  pnr->QueueTask(&MHD::CornerE, pmhd, MHD_EField, "MHD_EField", Task_Run, {MHD_RecvU});
  pnr->QueueTask(&MHD::SendE, pmhd, MHD_SendE, "MHD_SendE", Task_Run, {MHD_EField});
  pnr->QueueTask(&MHD::RecvE, pmhd, MHD_RecvE, "MHD_RecvE", Task_Run, {MHD_SendE});
  pnr->QueueTask(&MHD::CT, pmhd, MHD_CT, "MHD_CT", Task_Run, {MHD_RecvE});
  pnr->QueueTask(&MHD::RestrictB, pmhd, MHD_RestB, "MHD_RestB", Task_Run, {MHD_CT});
  pnr->QueueTask(&MHD::SendB, pmhd, MHD_SendB, "MHD_SendB", Task_Run, {MHD_RestB});
  pnr->QueueTask(&MHD::RecvB, pmhd, MHD_RecvB, "MHD_RecvB", Task_Run, {MHD_SendB});
  pnr->QueueTask(&MHD::ApplyPhysicalBCs, pmhd, MHD_BCS, "MHD_BCS", Task_Run, {MHD_RecvB});
  //pnr->QueueTask(&DynGRMHD::ApplyPhysicalBCs, this, MHD_BCS, "MHD_BCS", Task_Run,
  //                 {MHD_RecvB});
  pnr->QueueTask(&MHD::Prolongate, pmhd, MHD_Prolong, "MHD_Prolong", Task_Run, {MHD_BCS});
  if (pz4c == nullptr && padm->is_dynamic == true) {
    pnr->QueueTask(&DynGRMHD::SetADMVariables, this, MHD_SetADM, "MHD_SetADM", Task_Run,
                    {MHD_ExplRK});
    pnr->QueueTask(&DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim, this, MHD_C2P,
                   "MHD_C2P", Task_Run, {MHD_Prolong, MHD_SetADM}, {Z4c_Excise});
    pnr->QueueTask(&DynGRMHD::UpdateExcisionMasks, this, MHD_Excise, "MHD_Excise",
                   Task_Run, {MHD_SetADM});
  } else {
    pnr->QueueTask(&DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim, this, MHD_C2P,
                   "MHD_C2P", Task_Run, {MHD_Prolong}, {Z4c_Excise});
  }
  pnr->QueueTask(&MHD::NewTimeStep, pmhd, MHD_Newdt, "MHD_Newdt", Task_Run, {MHD_C2P});

  // End task list
  pnr->QueueTask(&MHD::ClearSend, pmhd, MHD_ClearS, "MHD_ClearS", Task_End);
  pnr->QueueTask(&MHD::ClearRecv, pmhd, MHD_ClearR, "MHD_ClearR", Task_End);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::PrimToConInit(int is, int ie, int js, int je,
                                                    int ks, int ke) {
  eos.PrimToCons(pmy_pack->pmhd->w0, pmy_pack->pmhd->bcc0, pmy_pack->pmhd->u0,
                 is, ie, js, je, ks, ke);
  if (pmy_pack->ptmunu != nullptr) {
    bool fixed = fixed_evolution;
    fixed_evolution = false;
    SetTmunu(nullptr, 0);
    fixed_evolution = fixed;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void DynGRMHD::ConvertInternalEnergyToPressure
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConvertInternalEnergyToPressure(int is, int ie,
    int js, int je, int ks, int ke) {
  int nmb = pmy_pack->nmb_thispack;
  auto &prim = pmy_pack->pmhd->w0;
  auto &eos_ = eos.ps.GetEOS();
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  const Real mb = eos_.GetBaryonMass();

  par_for("coord_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real n = prim(m, IDN, k, j, i) / mb;
    Real egas = mb*n + prim(m, IEN, k, j, i);
    Real Y[MAX_SPECIES] = {0.};
    for (int s = 0; s < nscal; s++) {
      Y[s] = prim(m, nmhd + s, k, j, i);
    }
    Real T;
    // Note that this is done explicitly rather than with a flooring policy because we
    // don't have the temperature yet, and it's probable that the energy is bunk if the
    // density is. There may be a cleaner way to do this elsewhere.
    if (n < eos_.GetMinimumDensity()) {
      n = eos_.GetMinimumDensity();
      T = eos_.GetMinimumTemperature();
    } else {
      T = eos_.GetTemperatureFromE(n, egas, Y);
    }
    prim(m, IPR, k, j, i) = eos_.GetPressure(n, T, Y);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim(Driver *pdrive, int stage) {
  if (fixed_evolution) {
    return TaskStatus::complete;
  }

  // Extract the indices
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  C2PDebugProbe(pmy_pack, "DynGRMHD_C2P_Before", pdrive, stage);
  eos.ConsToPrim(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0, pmy_pack->pmhd->bcc0,
                 pmy_pack->pmhd->w0, temperature, 0, n1m1, 0, n2m1, 0, n3m1, false);
  C2PDebugProbe(pmy_pack, "DynGRMHD_C2P_After", pdrive, stage);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::ConToPrimBC(int is, int ie, int js, int je, int ks, int ke)
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrimBC(int is, int ie, int js, int je,
                                                int ks, int ke) {
  if (fixed_evolution) {
    return;
  }
  eos.ConsToPrim(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0, pmy_pack->pmhd->bcc0,
                 pmy_pack->pmhd->w0, temperature, is, ie, js, je, ks, ke, false);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTerms(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs, int nghost, int stage) {
  switch (nghost) {
    case 2: AddCoordTermsEOS<2>(prim, bcc, dt, rhs, stage);
            break;
    case 3: AddCoordTermsEOS<3>(prim, bcc, dt, rhs, stage);
            break;
    case 4: AddCoordTermsEOS<4>(prim, bcc, dt, rhs, stage);
            break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ApplyPhysicalBCs(Driver *pdrive, int stage)
//  \brief
TaskStatus DynGRMHD::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // We need the first physical point on all the boundaries in order to calculate
  // the boundaries. So, we need to perform a ConToPrim at these points.
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto pm = pmy_pack->pmesh;
  auto pmhd = pmy_pack->pmhd;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  // X1-boundary
  ConToPrimBC(is-ng, is+ng, 0, (n2-1), 0, (n3-1));
  ConToPrimBC(ie-ng, ie+ng, 0, (n2-1), 0, (n3-1));
  // X2-boundary
  if (pm->multi_d) {
    ConToPrimBC(0, (n1-1), js-ng, js+ng, 0, (n3-1));
    ConToPrimBC(0, (n1-1), je-ng, je+ng, 0, (n3-1));
  }
  // X3-boundary
  if (pm->three_d) {
    ConToPrimBC(0, (n1-1), 0, (n2-1), ks-ng, ks+ng);
    ConToPrimBC(0, (n1-1), 0, (n2-1), ke-ng, ke+ng);
  }

  // Physical boundaries
  pmhd->pbval_u->HydroBCs((pmy_pack), (pmhd->pbval_u->u_in), pmhd->w0);
  pmhd->pbval_b->BFieldBCs((pmy_pack), (pmhd->pbval_b->b_in), pmhd->b0);

  // User BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  // We now need to do a PrimToCon on all these boundary points.
  // X1-boundary
  PrimToConInit(is-ng, is, 0, (n2-1), 0, (n3-1));
  PrimToConInit(ie, ie+ng, 0, (n2-1), 0, (n3-1));
  // X2-boundary
  if (pm->multi_d) {
    PrimToConInit(0, (n1-1), js-ng, js, 0, (n3-1));
    PrimToConInit(0, (n1-1), je, je+ng, 0, (n3-1));
  }
  // X3-boundary
  if (pm->three_d) {
    PrimToConInit(0, (n1-1), 0, (n2-1), ks-ng, ks);
    PrimToConInit(0, (n1-1), 0, (n2-1), ke, ke+ng);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::SetTmunu(Driver *pdrive, int stage)
//! \brief Add the perfect fluid contribution to the stress-energy tensor. This is assumed
//!  to be the first contribution, so it sets the values rather than adding.
TaskStatus DynGRMHD::SetTmunu(Driver *pdrive, int stage) {
  if (zero_tmunu_feedback) {
    if (pmy_pack->ptmunu != nullptr) {
      Kokkos::deep_copy(DevExeSpace(), pmy_pack->ptmunu->u_tmunu, 0.0);
    }
    return TaskStatus::complete;
  }
  if (fixed_evolution && !refresh_tmunu_when_fixed) {
    return TaskStatus::complete;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  //auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &tmunu = pmy_pack->ptmunu->tmunu;
  //auto &nhyd = pmy_pack->pmhd->nmhd;
  //int &nscal = pmy_pack->pmhd->nscalars;
  auto &prim = pmy_pack->pmhd->w0;
  // TODO(JMF): double-check that this needs to be u1, not u0!
  auto &cons = pmy_pack->pmhd->u0;
  auto &bcc = pmy_pack->pmhd->bcc0;

  par_for("dyngr_tmunu_loop",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Calculate the determinant/volume form
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i),adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i),adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i),adm.g_dd(m,2,2,k,j,i));
    Real ivol = 1.0/sqrt(detg);

    // Calculate the lower velocity components
    Real v_d[3] = {0.0};
    Real iW = 0.;
    Real B_d[3] = {0.0};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        v_d[a] += prim(m, IVX + b, k, j, i)*adm.g_dd(m, a, b, k, j, i);
        iW += prim(m, IVX + a, k, j, i)*prim(m, IVX + b, k, j, i) *
                adm.g_dd(m, a, b, k, j, i);
        B_d[a] += bcc(m, b, k, j, i)*adm.g_dd(m, a, b, k, j, i)*ivol;
      }
    }
    iW = 1.0/sqrt(1. + iW);
    Real Bv = 0.;
    Real Bsq = 0.;
    for (int a = 0; a < 3; ++a) {
      Bv += bcc(m, a, k, j, i) * v_d[a]*ivol;
      Bsq += bcc(m, a, k, j, i) * B_d[a]*ivol;
    }
    Real bsq = (Bsq + Bv*Bv)*(iW*iW);

    tmunu.E(m, k, j, i) = (cons(m, IEN, k, j, i) + cons(m, IDN, k, j, i))*ivol;
    for (int a = 0; a < 3; ++a) {
      tmunu.S_d(m, a, k, j, i) = cons(m, IM1 + a, k, j, i)*ivol;
      for (int b = a; b < 3; ++b) {
        tmunu.S_dd(m, a, b, k, j, i) =
              cons(m, IM1 + a, k, j, i)*ivol*v_d[b]*iW
              - (B_d[a] + Bv*v_d[a])*SQR(iW)*B_d[b]
              + (prim(m, IPR, k, j, i) + 0.5*bsq)*adm.g_dd(m, a, b, k, j, i);
      }
    }
  });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::SetADMVariables
//! \brief

TaskStatus DynGRMHD::SetADMVariables(Driver *pdrive, int stage) {
  pmy_pack->padm->SetADMVariables(pmy_pack);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::UpdateExcisionMasks
//! \brief

TaskStatus DynGRMHD::UpdateExcisionMasks(Driver *pdrive, int stage) {
  if (pmy_pack->pcoord->coord_data.bh_excise && stage == pdrive->nexp_stages) {
    pmy_pack->pcoord->UpdateExcisionMasks();
  }
  return TaskStatus::complete;
}

template<class EOSPolicy, class ErrorPolicy> template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs, int stage) {
  if (fixed_evolution) {
    return;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = eos.ps.GetEOS();
  //auto &tmunu = pmy_pack->ptmunu->tmunu;

  int &nhyd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  constexpr int ncoord_diag = 66;
  bool do_coord_src_debug = false;
  int dbg_m[2] = {-1, -1};
  int dbg_i[2] = {-1, -1};
  int dbg_j[2] = {-1, -1};
  int dbg_k[2] = {-1, -1};
  Real dbg_x[2] = {0.0, 0.0};
  Real dbg_y[2] = {0.0, 0.0};
  Real dbg_z[2] = {0.0, 0.0};
  DvceArray2D<Real> coord_diag;
  if (CoordSrcDebugEnabled()) {
    Mesh *pm = pmy_pack->pmesh;
    const int cycle = pm->ncycle;
    const int target_cycle = EnvInt("ATHENA_COORDSRC_DEBUG_CYCLE", -1);
    if ((target_cycle >= 0 && cycle == target_cycle) ||
        (target_cycle < 0 && (cycle < 3 || cycle == 80 || cycle == 160))) {
      const int target_stage = EnvInt("ATHENA_COORDSRC_DEBUG_STAGE", -1);
      if (target_stage < 0 || stage == target_stage) {
        auto &gid_size = pmy_pack->pmb->mb_size;
        const Real x_target = EnvReal("ATHENA_SYM_X_TARGET", 20.0);
        const Real z_target = EnvReal("ATHENA_SYM_Z_TARGET", 0.0);
        for (int side = 0; side < 2; ++side) {
          const bool below = (side == 0);
          Real best_abs_y = std::numeric_limits<Real>::max();
          for (int m = 0; m < pmy_pack->nmb_thispack; ++m) {
            const auto &mb = gid_size.h_view(m);
            const bool x_inside = (x_target >= mb.x1min) && (x_target < mb.x1max);
            const bool z_inside = (z_target >= mb.x3min) && (z_target < mb.x3max);
            if (!(x_inside && z_inside)) continue;

            int i = is + static_cast<int>(std::floor((x_target - mb.x1min)/mb.dx1));
            i = std::max(is, std::min(ie, i));
            int k = ks + static_cast<int>(std::floor((z_target - mb.x3min)/mb.dx3));
            k = std::max(ks, std::min(ke, k));

            for (int jj = js; jj <= je; ++jj) {
              const Real y = mb.x2min + (static_cast<Real>(jj - js) + 0.5)*mb.dx2;
              if (((below && y < 0.0) || (!below && y > 0.0)) &&
                  std::abs(y) < best_abs_y) {
                best_abs_y = std::abs(y);
                dbg_m[side] = m;
                dbg_i[side] = i;
                dbg_j[side] = jj;
                dbg_k[side] = k;
                dbg_x[side] = mb.x1min + (static_cast<Real>(i - is) + 0.5)*mb.dx1;
                dbg_y[side] = y;
                dbg_z[side] = mb.x3min + (static_cast<Real>(k - ks) + 0.5)*mb.dx3;
              }
            }
          }
        }
        do_coord_src_debug = (dbg_m[0] >= 0 || dbg_m[1] >= 0);
      }
    }
  }
  if (do_coord_src_debug) {
    coord_diag = DvceArray2D<Real>("coord-src-debug", 2, ncoord_diag);
    Kokkos::deep_copy(coord_diag, std::numeric_limits<Real>::quiet_NaN());
  }

  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  const int imap[3][3] = {
    {S11, S12, S13},
    {S12, S22, S23},
    {S13, S23, S33}
  };

  // Check the number of dimensions to determine which derivatives we need.
  int ndim;
  if (pmy_pack->pmesh->one_d) {
    ndim = 1;
  } else if (pmy_pack->pmesh->two_d) {
    ndim = 2;
  } else {
    ndim = 3;
  }

  const int dbg_m0 = dbg_m[0];
  const int dbg_i0 = dbg_i[0];
  const int dbg_j0 = dbg_j[0];
  const int dbg_k0 = dbg_k[0];
  const int dbg_m1 = dbg_m[1];
  const int dbg_i1 = dbg_i[1];
  const int dbg_j1 = dbg_j[1];
  const int dbg_k1 = dbg_k[1];

  par_for("coord_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    int debug_side = -1;
    if (do_coord_src_debug) {
      if (m == dbg_m0 && i == dbg_i0 && j == dbg_j0 && k == dbg_k0) {
        debug_side = 0;
      } else if (m == dbg_m1 && i == dbg_i1 && j == dbg_j1 && k == dbg_k1) {
        debug_side = 1;
      }
    }
    const Real pre_rho = (debug_side >= 0) ? rhs(m, IDN, k, j, i) : 0.0;
    const Real pre_momx = (debug_side >= 0) ? rhs(m, IM1, k, j, i) : 0.0;
    const Real pre_momy = (debug_side >= 0) ? rhs(m, IM2, k, j, i) : 0.0;
    const Real pre_momz = (debug_side >= 0) ? rhs(m, IM3, k, j, i) : 0.0;
    const Real pre_tau = (debug_side >= 0) ? rhs(m, IEN, k, j, i) : 0.0;

    // Extract the metric and coordinate quantities.
    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
    const Real& alpha = adm.alpha(m, k, j, i);
    Real beta_u[3] = {adm.beta_u(m,0,k,j,i),
                      adm.beta_u(m,1,k,j,i), adm.beta_u(m,2,k,j,i)};
    Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                g3d[S22], g3d[S23], g3d[S33]);
    Real vol = sqrt(detg);
    Real g3u[NSPMETRIC] = {0.};
    adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                    &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);

    // Calculate the metric derivatives
    Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2, 1./size.d_view(m).dx3};
    Real dalpha_d[3] = {0.};
    for (int a = 0; a < ndim; a++) {
      dalpha_d[a] = Dx<NGHOST>(a, idx, adm.alpha, m, k, j, i);
    }
    Real dbeta_du[3][3] = {0.};
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < ndim; b++) {
        dbeta_du[b][a] = Dx<NGHOST>(b, idx, adm.beta_u, m, a, k, j, i);
      }
    }
    Real dg_ddd[3][3][3] = {0.};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < ndim; ++c) {
          dg_ddd[c][a][b] = Dx<NGHOST>(c, idx, adm.g_dd, m, a, b, k, j, i);
        }
      }
    }

    // Fluid quantities
    Real prim_pt[NPRIM] = {0.0};
    prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
    prim_pt[PVX] = prim(m, IVX, k, j, i);
    prim_pt[PVY] = prim(m, IVY, k, j, i);
    prim_pt[PVZ] = prim(m, IVZ, k, j, i);
    for (int s = 0; s < nscal; s++) {
      prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
    }
    prim_pt[PPR] = prim(m, IPR, k, j, i);
    prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);

    // Get the conserved variables. Note that we don't use PrimitiveSolver here --
    // that's because we would need to recalculate quantities used in E and S_d in order
    // to get S_dd.
    Real H =
      prim(m, IDN, k, j, i)*eos_.GetEnthalpy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
    Real usq = Primitive::SquareVector(&prim_pt[PVX], g3d);
    Real const Wsq = 1.0 + usq;
    Real const W = sqrt(Wsq);
    Real B_u[NMAG] = {bcc(m, IBX, k, j, i)/vol,
                      bcc(m, IBY, k, j, i)/vol,
                      bcc(m, IBZ, k, j, i)/vol};
    Real Bv = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Bv += adm.g_dd(m,a,b,k,j,i)*prim_pt[PVX + a]*B_u[b];
      }
    }
    Real Bsq = Primitive::SquareVector(B_u, g3d);
    Bv = Bv/W;
    Real bsq = Bv*Bv + Bsq/Wsq;

    Real E = (H*Wsq + Bsq) - prim_pt[PPR] - 0.5*bsq;
    //Real E = tmunu.E(m,k,j,i);

    Real S_d[3] = {0.0};
    for (int a = 0; a < 3; a++) {
      //S_d[a] = tmunu.S_d(m,a,k,j,i);
      for (int b = 0; b < 3; b++) {
        S_d[a] += ((H*Wsq + Bsq)*prim_pt[PVX + b]/W - Bv*B_u[b])*g3d[imap[a][b]];
      }
    }

    Real S_uu[3][3];
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b <= a; b++) {
        S_uu[a][b] = (H + Bsq/Wsq)*prim_pt[PVX + a]*prim_pt[PVX + b]
                      - B_u[a]*B_u[b]/Wsq
                      - Bv*(B_u[a]*prim_pt[PVX + b] + B_u[b]*prim_pt[PVX + a])/W
                      + (prim_pt[PPR] + 0.5*bsq)*g3u[imap[a][b]];
        /*S_uu[a][b] = 0.0;
        for (int c = 0; c < 3; c++) {
          for (int d = 0; d < 3; d++) {
            S_uu[a][b] += tmunu.S_dd(m,c,d,k,j,i)*g3u[imap[a][c]]*g3u[imap[b][d]];
          }
        }*/
        S_uu[b][a] = S_uu[a][b];
      }
    }

    Real src_tau_k = 0.0;
    Real src_tau_alpha = 0.0;
    Real src_mom_geom[3] = {0.0};
    Real src_mom_beta[3] = {0.0};
    Real src_mom_alpha[3] = {0.0};

    // Assemble energy RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        rhs(m, IEN, k, j, i) += dt*vol*(
            alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b] -
            g3u[imap[a][b]]*S_d[a]*dalpha_d[b]);
        if (debug_side >= 0) {
          src_tau_k += dt*vol*alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b];
          src_tau_alpha += -dt*vol*g3u[imap[a][b]]*S_d[a]*dalpha_d[b];
        }
      }
    }

    // Assemble momentum RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        for (int c = 0; c < 3; c++) {
          const Real geom_src = 0.5*dt*alpha*vol*S_uu[b][c]*dg_ddd[a][b][c];
          rhs(m,IM1+a, k, j, i) += geom_src;
          if (debug_side >= 0) src_mom_geom[a] += geom_src;
        }
        const Real beta_src = dt*vol*S_d[b]*dbeta_du[a][b];
        rhs(m, IM1+a, k, j, i) += beta_src;
        if (debug_side >= 0) src_mom_beta[a] += beta_src;
      }
      const Real alpha_src = -dt*vol*E*dalpha_d[a];
      rhs(m, IM1+a, k, j, i) += alpha_src;
      if (debug_side >= 0) src_mom_alpha[a] += alpha_src;
    }

    if (debug_side >= 0) {
      coord_diag(debug_side, 0) = pre_rho;
      coord_diag(debug_side, 1) = pre_momx;
      coord_diag(debug_side, 2) = pre_momy;
      coord_diag(debug_side, 3) = pre_momz;
      coord_diag(debug_side, 4) = pre_tau;
      coord_diag(debug_side, 5) = 0.0;
      coord_diag(debug_side, 6) = src_mom_geom[0] + src_mom_beta[0] + src_mom_alpha[0];
      coord_diag(debug_side, 7) = src_mom_geom[1] + src_mom_beta[1] + src_mom_alpha[1];
      coord_diag(debug_side, 8) = src_mom_geom[2] + src_mom_beta[2] + src_mom_alpha[2];
      coord_diag(debug_side, 9) = src_tau_k + src_tau_alpha;
      coord_diag(debug_side, 10) = rhs(m, IDN, k, j, i);
      coord_diag(debug_side, 11) = rhs(m, IM1, k, j, i);
      coord_diag(debug_side, 12) = rhs(m, IM2, k, j, i);
      coord_diag(debug_side, 13) = rhs(m, IM3, k, j, i);
      coord_diag(debug_side, 14) = rhs(m, IEN, k, j, i);
      coord_diag(debug_side, 15) = alpha;
      coord_diag(debug_side, 16) = beta_u[0];
      coord_diag(debug_side, 17) = beta_u[1];
      coord_diag(debug_side, 18) = beta_u[2];
      coord_diag(debug_side, 19) = detg;
      coord_diag(debug_side, 20) = vol;
      coord_diag(debug_side, 21) = dalpha_d[0];
      coord_diag(debug_side, 22) = dalpha_d[1];
      coord_diag(debug_side, 23) = dalpha_d[2];
      for (int dir = 0; dir < 3; ++dir) {
        for (int comp = 0; comp < 3; ++comp) {
          coord_diag(debug_side, 24 + 3*dir + comp) = dbeta_du[dir][comp];
        }
      }
      for (int dir = 0; dir < 3; ++dir) {
        coord_diag(debug_side, 33 + 6*dir + 0) = dg_ddd[dir][0][0];
        coord_diag(debug_side, 33 + 6*dir + 1) = dg_ddd[dir][0][1];
        coord_diag(debug_side, 33 + 6*dir + 2) = dg_ddd[dir][0][2];
        coord_diag(debug_side, 33 + 6*dir + 3) = dg_ddd[dir][1][1];
        coord_diag(debug_side, 33 + 6*dir + 4) = dg_ddd[dir][1][2];
        coord_diag(debug_side, 33 + 6*dir + 5) = dg_ddd[dir][2][2];
      }
      coord_diag(debug_side, 51) = E;
      coord_diag(debug_side, 52) = S_d[0];
      coord_diag(debug_side, 53) = S_d[1];
      coord_diag(debug_side, 54) = S_d[2];
      coord_diag(debug_side, 55) = src_mom_geom[0];
      coord_diag(debug_side, 56) = src_mom_beta[0];
      coord_diag(debug_side, 57) = src_mom_alpha[0];
      coord_diag(debug_side, 58) = src_mom_geom[1];
      coord_diag(debug_side, 59) = src_mom_beta[1];
      coord_diag(debug_side, 60) = src_mom_alpha[1];
      coord_diag(debug_side, 61) = src_mom_geom[2];
      coord_diag(debug_side, 62) = src_mom_beta[2];
      coord_diag(debug_side, 63) = src_mom_alpha[2];
      coord_diag(debug_side, 64) = src_tau_k;
      coord_diag(debug_side, 65) = src_tau_alpha;
    }
  });

  if (do_coord_src_debug) {
    Kokkos::fence();
    auto hdiag = Kokkos::create_mirror_view(coord_diag);
    Kokkos::deep_copy(hdiag, coord_diag);
    auto &gid = pmy_pack->pmb->mb_gid;
    auto &lev = pmy_pack->pmb->mb_lev;
    for (int side = 0; side < 2; ++side) {
      if (dbg_m[side] < 0) continue;
      std::cout << std::setprecision(17)
                << "COORDSRCDBG label=DynGRMHD_AddCoordTerms"
                << " rank=" << global_variable::my_rank
                << " gid=" << gid.h_view(dbg_m[side])
                << " level=" << lev.h_view(dbg_m[side])
                << " side=" << (side == 0 ? "below" : "above")
                << " cycle=" << pmy_pack->pmesh->ncycle
                << " time=" << pmy_pack->pmesh->time
                << " stage=" << stage
                << " i=" << dbg_i[side]
                << " j=" << dbg_j[side]
                << " k=" << dbg_k[side]
                << " x=" << dbg_x[side]
                << " y=" << dbg_y[side]
                << " z=" << dbg_z[side]
                << " pre_rho=" << hdiag(side, 0)
                << " pre_momx=" << hdiag(side, 1)
                << " pre_momy=" << hdiag(side, 2)
                << " pre_momz=" << hdiag(side, 3)
                << " pre_tau=" << hdiag(side, 4)
                << " src_rho=" << hdiag(side, 5)
                << " src_momx=" << hdiag(side, 6)
                << " src_momy=" << hdiag(side, 7)
                << " src_momz=" << hdiag(side, 8)
                << " src_tau=" << hdiag(side, 9)
                << " post_rho=" << hdiag(side, 10)
                << " post_momx=" << hdiag(side, 11)
                << " post_momy=" << hdiag(side, 12)
                << " post_momz=" << hdiag(side, 13)
                << " post_tau=" << hdiag(side, 14)
                << " alpha=" << hdiag(side, 15)
                << " beta_x=" << hdiag(side, 16)
                << " beta_y=" << hdiag(side, 17)
                << " beta_z=" << hdiag(side, 18)
                << " detg=" << hdiag(side, 19)
                << " sdetg=" << hdiag(side, 20)
                << " dalpha_x=" << hdiag(side, 21)
                << " dalpha_y=" << hdiag(side, 22)
                << " dalpha_z=" << hdiag(side, 23)
                << " dbeta_dx_bx=" << hdiag(side, 24)
                << " dbeta_dx_by=" << hdiag(side, 25)
                << " dbeta_dx_bz=" << hdiag(side, 26)
                << " dbeta_dy_bx=" << hdiag(side, 27)
                << " dbeta_dy_by=" << hdiag(side, 28)
                << " dbeta_dy_bz=" << hdiag(side, 29)
                << " dbeta_dz_bx=" << hdiag(side, 30)
                << " dbeta_dz_by=" << hdiag(side, 31)
                << " dbeta_dz_bz=" << hdiag(side, 32)
                << " dg_dx_gxx=" << hdiag(side, 33)
                << " dg_dx_gxy=" << hdiag(side, 34)
                << " dg_dx_gxz=" << hdiag(side, 35)
                << " dg_dx_gyy=" << hdiag(side, 36)
                << " dg_dx_gyz=" << hdiag(side, 37)
                << " dg_dx_gzz=" << hdiag(side, 38)
                << " dg_dy_gxx=" << hdiag(side, 39)
                << " dg_dy_gxy=" << hdiag(side, 40)
                << " dg_dy_gxz=" << hdiag(side, 41)
                << " dg_dy_gyy=" << hdiag(side, 42)
                << " dg_dy_gyz=" << hdiag(side, 43)
                << " dg_dy_gzz=" << hdiag(side, 44)
                << " dg_dz_gxx=" << hdiag(side, 45)
                << " dg_dz_gxy=" << hdiag(side, 46)
                << " dg_dz_gxz=" << hdiag(side, 47)
                << " dg_dz_gyy=" << hdiag(side, 48)
                << " dg_dz_gyz=" << hdiag(side, 49)
                << " dg_dz_gzz=" << hdiag(side, 50)
                << " E=" << hdiag(side, 51)
                << " Sx=" << hdiag(side, 52)
                << " Sy=" << hdiag(side, 53)
                << " Sz=" << hdiag(side, 54)
                << " src_momx_geom=" << hdiag(side, 55)
                << " src_momx_beta=" << hdiag(side, 56)
                << " src_momx_alpha=" << hdiag(side, 57)
                << " src_momy_geom=" << hdiag(side, 58)
                << " src_momy_beta=" << hdiag(side, 59)
                << " src_momy_alpha=" << hdiag(side, 60)
                << " src_momz_geom=" << hdiag(side, 61)
                << " src_momz_beta=" << hdiag(side, 62)
                << " src_momz_alpha=" << hdiag(side, 63)
                << " src_tau_k=" << hdiag(side, 64)
                << " src_tau_alpha=" << hdiag(side, 65)
                << std::endl;
    }
  }
}

// Instantiated templates
template class DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::PiecewisePolytrope, Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                          Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                          Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::EOSHybrid<Primitive::NormalLogs>,
                          Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::EOSHybrid<Primitive::NQTLogs>,
                          Primitive::ResetFloor>;

// Macro for defining CoordTerms templates
#define INSTANTIATE_COORD_TERMS(EOSPolicy, ErrorPolicy) \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<2>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs, int stage); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<3>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs, int stage); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<4>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs, int stage);

INSTANTIATE_COORD_TERMS(Primitive::IdealGas, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::PiecewisePolytrope, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::EOSCompOSE<Primitive::NormalLogs>,
                        Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::EOSHybrid<Primitive::NormalLogs>,
                        Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor);

#undef INSTANTIATE_COORD_TERMS

} // namespace dyngr
