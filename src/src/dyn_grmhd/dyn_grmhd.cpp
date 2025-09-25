//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr.cpp
//! \brief implementation of functions for DynGRMHD and DynGRMHDPS controlling the task
//! list

#include <math.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

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
  eos.ConsToPrim(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0, pmy_pack->pmhd->bcc0,
                 pmy_pack->pmhd->w0, temperature, 0, n1m1, 0, n2m1, 0, n3m1, false);
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
    const Real dt, DvceArray5D<Real> &rhs, int nghost) {
  switch (nghost) {
    case 2: AddCoordTermsEOS<2>(prim, bcc, dt, rhs);
            break;
    case 3: AddCoordTermsEOS<3>(prim, bcc, dt, rhs);
            break;
    case 4: AddCoordTermsEOS<4>(prim, bcc, dt, rhs);
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
  if (fixed_evolution) {
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
    const Real dt, DvceArray5D<Real> &rhs) {
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

  par_for("coord_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

    // Assemble energy RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        rhs(m, IEN, k, j, i) += dt*vol*(alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b] -
            g3u[imap[a][b]] * S_d[a]*dalpha_d[b]);
      }
    }

    // Assemble momentum RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        for (int c = 0; c < 3; c++) {
          rhs(m,IM1+a, k, j, i) += 0.5*dt*alpha*vol*S_uu[b][c]*dg_ddd[a][b][c];
        }
        rhs(m, IM1+a, k, j, i) += dt*vol*S_d[b]*dbeta_du[a][b];
      }
      rhs(m, IM1+a, k, j, i) -= dt*vol*E*dalpha_d[a];
    }
  });
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
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<3>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<4>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs);

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
