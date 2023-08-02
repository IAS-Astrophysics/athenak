//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr.cpp
//  \brief implementation of functions for DynGR and DynGRPS controlling the task list

#include <iostream>
#include <math.h>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
//#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "adm/adm.hpp"
#include "tmunu/tmunu.hpp"
#include "dyngr.hpp"
#include "numerical-relativity/numerical_relativity.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

// A dumb template function containing the switch statement needed to select an EOS.
template<class ErrorPolicy>
DynGR* SelectDynGREOS(MeshBlockPack *ppack, ParameterInput *pin, DynGR_EOS eos_policy) {
  DynGR* dyn_gr = nullptr;
  switch(eos_policy) {
    case DynGR_EOS::eos_ideal:
      dyn_gr = new DynGRPS<Primitive::IdealGas, ErrorPolicy>(ppack, pin);
      break;
    case DynGR_EOS::eos_piecewise_poly:
      dyn_gr = new DynGRPS<Primitive::PiecewisePolytrope, ErrorPolicy>(ppack, pin);
      break;
  }
  return dyn_gr;
}

DynGR* BuildDynGR(MeshBlockPack *ppack, ParameterInput *pin) {
  std::string eos_string = pin->GetString("mhd", "dyn_eos");
  std::string error_string = pin->GetString("mhd", "dyn_error");
  DynGR_EOS eos_policy;
  DynGR_Error error_policy;

  if (eos_string.compare("ideal") == 0) {
    eos_policy = DynGR_EOS::eos_ideal;
  } else if (eos_string.compare("piecewise_poly") == 0) {
    eos_policy = DynGR_EOS::eos_piecewise_poly;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> dyn_eos = '" << eos_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (error_string.compare("reset_floor") == 0) {
    error_policy = DynGR_Error::reset_floor;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> dyn_error = '" << error_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  DynGR* dyn_gr = nullptr;

  switch (error_policy) {
    case DynGR_Error::reset_floor:
      dyn_gr = SelectDynGREOS<Primitive::ResetFloor>(ppack, pin, eos_policy);
      break;
  }

  dyn_gr->eos_policy = eos_policy;
  dyn_gr->error_policy = error_policy;

  return dyn_gr;
}

DynGR::DynGR(MeshBlockPack *pp, ParameterInput *pin) : pmy_pack(pp) {
  std::string rsolver = pin->GetString("mhd", "rsolver");
  if (rsolver.compare("llf") == 0) {
    rsolver_method = DynGR_RSolver::llf_dyngr;
  } else if (rsolver.compare("hlle") == 0) {
    rsolver_method = DynGR_RSolver::hlle_dyngr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> rsolver = '" << rsolver
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  scratch_level = pin->GetOrAddInteger("mhd", "dyn_scratch", 0);
}

DynGR::~DynGR() {

}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
  pmy_pack->pz4c->ADMMatterSource(pmy_pack);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void DynGRPS::AssembleDynGRTasks
//  \brief Adds dyngr tasks to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysics() function directly after Hydro constrctr
//
//  Stage start tasks are those that must be cmpleted over all MeshBlocks before EACH
//  stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//
//  Stage run tasks are those performed in EACH stage
//
//  Stage end tasks are those that can only be cmpleted after all the stage run tasks are
//  finished over all MeshBlocks for EACH stage, such as clearing all MPI non-blocking
//  sends, etc.
template<class EOSPolicy, class ErrorPolicy>
void DynGRPS<EOSPolicy, ErrorPolicy>::AssembleDynGRTasks(TaskList &start, 
    TaskList &run, TaskList &end) {
  TaskID none(0);
  auto &indcs = pmy_pack->pmesh->mb_indcs;

  using namespace mhd;  // NOLINT(build/namespaces)
  using namespace z4c;    // NOLINT(build/namespaces)
  Z4c *pz4c = pmy_pack->pz4c;
  MHD *pmhd = pmy_pack->pmhd;

  // naming convention: hydro task id names are unchanged, all z4c tasks now have a z at the beginning

  // start task list
  // Hydro
  id.irecv = start.AddTask(&MHD::InitRecv, pmhd, none);
  // Z4c
  if (pz4c != nullptr) {
    id.zrecv = start.AddTask(&Z4c::InitRecv, pz4c, none);
  }

  // run task list
  id.copyu = run.AddTask(&MHD::CopyCons, pmhd, none);
  // select which calculate flux function to add based on rsolver_method.
  // Calc flux requires metric in flux - must happen before z4ctoadm updates the metric
  if (rsolver_method == DynGR_RSolver::llf_dyngr) {
    id.flux = run.AddTask(&DynGRPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGR_RSolver::llf_dyngr>,this,id.copyu);
  } // put more rsolvers here
  else if (rsolver_method == DynGR_RSolver::hlle_dyngr) {
    id.flux = run.AddTask(&DynGRPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGR_RSolver::hlle_dyngr>,this,id.copyu);
  }
   // put more rsolvers here
  else{
    abort();
  }

  // now the rest of the Hydro run tasks
  id.settmunu = run.AddTask(&DynGR::SetTmunu, this, id.copyu);
  id.sendf = run.AddTask(&MHD::SendFlux, pmhd, id.flux);
  id.recvf = run.AddTask(&MHD::RecvFlux, pmhd, id.sendf);
  id.rkdep = id.recvf | id.settmunu;
  id.expl  = run.AddTask(&MHD::ExpRKUpdate, pmhd, id.rkdep); // requires metric in geometric source terms - must happen before z4ctoadm
  id.restu = run.AddTask(&MHD::RestrictU, pmhd, id.expl);
  id.sendu = run.AddTask(&MHD::SendU, pmhd, id.restu);
  id.recvu = run.AddTask(&MHD::RecvU, pmhd, id.sendu);
  id.efld = run.AddTask(&MHD::CornerE, pmhd, id.recvu);
  id.sende = run.AddTask(&MHD::SendE, pmhd, id.efld);
  id.recve = run.AddTask(&MHD::RecvE, pmhd, id.sende);
  id.ct = run.AddTask(&MHD::CT, pmhd, id.recve);
  id.restb = run.AddTask(&MHD::RestrictB, pmhd, id.ct);
  id.sendb = run.AddTask(&MHD::SendB, pmhd, id.restb);
  id.recvb = run.AddTask(&MHD::RecvB, pmhd, id.sendb);
  id.bcs   = run.AddTask(&MHD::ApplyPhysicalBCs, pmhd, id.recvb);

  // Z4c
  if (pz4c != nullptr) {
    id.zmattersrc = run.AddTask(&DynGR::ADMMatterSource_, this, none); // must happen before c2p, must happen before CalcRHS
    id.zcopyu = run.AddTask(&Z4c::CopyU, pz4c, none);
    id.zcrhsdep = id.zcopyu | id.zmattersrc;
    switch (indcs.ng) {
      case 2: id.zcrhs = run.AddTask(&Z4c::CalcRHS<2>, pz4c, id.zcrhsdep);
              break;
      case 3: id.zcrhs = run.AddTask(&Z4c::CalcRHS<3>, pz4c, id.zcrhsdep);
              break;
      case 4: id.zcrhs = run.AddTask(&Z4c::CalcRHS<4>, pz4c, id.zcrhsdep);
              break;
    }
    id.zsombc = run.AddTask(&Z4c::Z4cBoundaryRHS, pz4c, id.zcrhs);
    id.zexpl  = run.AddTask(&Z4c::ExpRKUpdate, pz4c, id.zsombc); // after this point z4c variables are timestepped
    id.zrestu = run.AddTask(&Z4c::RestrictU, pz4c, id.zexpl);
    id.zsendu = run.AddTask(&Z4c::SendU, pz4c, id.zrestu);
    id.zrecvu = run.AddTask(&Z4c::RecvU, pz4c, id.zsendu);
    id.zbcs   = run.AddTask(&Z4c::ApplyPhysicalBCs, pz4c, id.zrecvu);
    id.zalgc  = run.AddTask(&Z4c::EnforceAlgConstr, pz4c, id.zbcs);
    id.zadep  = id.zalgc | id.flux | id.expl;
    id.z4tad  = run.AddTask(&Z4c::Z4cToADM_, pz4c, id.zadep); // after this point ADM variables are updated - now cannot calcualte geometric source terms/fluxes etc.
    id.zadmc  = run.AddTask(&Z4c::ADMConstraints_, pz4c, id.z4tad);
    id.znewdt = run.AddTask(&Z4c::NewTimeStep, pz4c, id.zadmc); // only need 1 time step
  }
  // TODO MHD

  if (pz4c != nullptr) {
    id.c2pdep = id.bcs | id.z4tad;
  } else {
    id.c2pdep = id.bcs;
  }
  id.c2p = run.AddTask(&DynGRPS<EOSPolicy, ErrorPolicy>::ConToPrim, this, id.c2pdep);

  // c2p takes time-stepped cons -> time-stepped prims, should use time-stepped metric
  // should happen after z4ctoadm. Prims are now updated so can no longer calculate source
  // for Z4c equations, must happen after calcz4crhs (but this happens before z4ctoadm anyway)
  id.newdt = run.AddTask(&MHD::NewTimeStep, pmhd, id.c2p); // only need 1 timestep


  // end task list
  id.clear = end.AddTask(&MHD::ClearSend, pmhd, none);

  if (pz4c != nullptr) {
    id.zclear = end.AddTask(&Z4c::ClearSend, pz4c, none);
  }
}

template<class EOSPolicy, class ErrorPolicy>
void DynGRPS<EOSPolicy, ErrorPolicy>::QueueDynGRTasks() {
  using namespace mhd;  // NOLINT(build/namespaces)
  using namespace z4c;  // NOLINT(build/namespaces)
  using namespace numrel; // NOLINT(build/namespaces)
  Z4c *pz4c = pmy_pack->pz4c;
  MHD *pmhd = pmy_pack->pmhd;
  NumericalRelativity *pnr = pmy_pack->pnr;

  std::vector<TaskName> none;
  std::vector<TaskName> dep;
  std::vector<TaskName> opt;

  // Start task list
  pnr->QueueTask(&MHD::InitRecv, pmhd, MHD_Recv, "MHD_Recv", Task_Start, none, none);

  // Run task list
  pnr->QueueTask(&MHD::CopyCons, pmhd, MHD_CopyU, "MHD_CopyU", Task_Run, none, none);

  // Select which CalculateFlux function to add based on rsolver_method.
  // CalcFlux requires metric in flux - must happen before z4ctoadm updates the metric
  dep.push_back(MHD_CopyU);
  if (rsolver_method == DynGR_RSolver::llf_dyngr) {
    pnr->QueueTask(&DynGRPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGR_RSolver::llf_dyngr>, this,
                   MHD_Flux, "MHD_Flux", Task_Run, dep, none);
  }
  else if (rsolver_method == DynGR_RSolver::hlle_dyngr) {
    pnr->QueueTask(&DynGRPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGR_RSolver::hlle_dyngr>, this,
                   MHD_Flux, "MHD_Flux", Task_Run, dep, none);
  }
  // put more rsolvers here
  else {
    abort();
  }

  // Now the rest of the MHD run tasks
  if (pz4c != nullptr) {
    pnr->QueueTask(&DynGR::SetTmunu, this, MHD_SetTmunu, "MHD_SetTmunu", Task_Run, dep, none);
  }
  dep.clear();
  dep.push_back(MHD_Flux);
  pnr->QueueTask(&MHD::SendFlux, pmhd, MHD_SendFlux, "MHD_SendFlux", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_SendFlux);
  pnr->QueueTask(&MHD::RecvFlux, pmhd, MHD_RecvFlux, "MHD_RecvFlux", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_RecvFlux);
  if (pz4c != nullptr) {
    dep.push_back(MHD_SetTmunu);
  }
  pnr->QueueTask(&MHD::ExpRKUpdate, pmhd, MHD_ExplRK, "MHD_ExplRK", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_ExplRK);
  pnr->QueueTask(&MHD::RestrictU, pmhd, MHD_RestU, "MHD_RestU", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_RestU);
  pnr->QueueTask(&MHD::SendU, pmhd, MHD_SendU, "MHD_SendU", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_SendU);
  pnr->QueueTask(&MHD::RecvU, pmhd, MHD_RecvU, "MHD_RecvU", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_RecvU);
  pnr->QueueTask(&MHD::CornerE, pmhd, MHD_EField, "MHD_EField", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_EField);
  pnr->QueueTask(&MHD::SendE, pmhd, MHD_SendE, "MHD_SendE", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_SendE);
  pnr->QueueTask(&MHD::RecvE, pmhd, MHD_RecvE, "MHD_RecvE", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_RecvE);
  pnr->QueueTask(&MHD::CT, pmhd, MHD_CT, "MHD_CT", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_CT);
  pnr->QueueTask(&MHD::RestrictB, pmhd, MHD_RestB, "MHD_RestB", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_RestB);
  pnr->QueueTask(&MHD::SendB, pmhd, MHD_SendB, "MHD_SendB", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_SendB);
  pnr->QueueTask(&MHD::RecvB, pmhd, MHD_RecvB, "MHD_RecvB", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_RecvB);
  pnr->QueueTask(&MHD::ApplyPhysicalBCs, pmhd, MHD_BCS, "MHD_BCS", Task_Run, dep, none);

  dep.clear();
  dep.push_back(MHD_BCS);
  opt.push_back(Z4c_Z4c2ADM);
  pnr->QueueTask(&DynGRPS<EOSPolicy, ErrorPolicy>::ConToPrim, this, MHD_C2P, "MHD_C2P", Task_Run, dep, opt);

  dep.clear();
  dep.push_back(MHD_C2P);
  pnr->QueueTask(&MHD::NewTimeStep, pmhd, MHD_Newdt, "MHD_Newdt", Task_Run, dep, none);

  pnr->QueueTask(&MHD::ClearSend, pmhd, MHD_Clear, "MHD_Clear", Task_End, none, none);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRPS<EOSPolicy, ErrorPolicy>::PrimToConInit(int is, int ie, int js, int je, int ks, int ke) {
  eos.PrimToCons(pmy_pack->pmhd->w0, pmy_pack->pmhd->bcc0, pmy_pack->pmhd->u0, is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRPS<EOSPolicy, ErrorPolicy>::ConToPrim(Driver *pdrive, int stage) {
  // Extract the indices
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  eos.ConsToPrim(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0, pmy_pack->pmhd->bcc0, pmy_pack->pmhd->w0,
                 0, n1m1, 0, n2m1, 0, n3m1, false);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTerms(const DvceArray5D<Real> &prim,
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
//! \fn  TaskStatus DynGR::SetTmunu(Driver *pdrive, int stage)
//! \brief Add the perfect fluid contribution to the stress-energy tensor. This is assumed
//!  to be the first contribution, so it sets the values rather than adding.
TaskStatus DynGR::SetTmunu(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  //auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int ncells1 = indcs.nx1+indcs.ng; // Align scratch buffers with variables
  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &tmunu = pmy_pack->ptmunu->tmunu;
  //auto &nhyd = pmy_pack->pmhd->nmhd;
  //int &nscal = pmy_pack->pmhd->nscalars;
  auto &prim = pmy_pack->pmhd->w0;
  // TODO: double-check that this needs to be u1, not u0!
  auto &cons = pmy_pack->pmhd->u0;
  auto &bcc = pmy_pack->pmhd->bcc0;

  int scr_level = scratch_level;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*4     // scalars
                  + ScrArray2D<Real>::shmem_size(3, ncells1)*2; // vectors
  //size_t scr_size = 0;
  par_for_outer("dyngr_tmunu_loop",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    // Scratch space
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> ivol;  // sqrt of 3-metric determinant
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> iW;     // Lorentz factor
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> Bv;    // B^i v_i
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> Bsq;   // B^i B_i
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> v_d;   // Velocity form
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> B_d;   // Magnetic field form

    ivol.NewAthenaScratchTensor(member, scr_level, ncells1);
    iW.NewAthenaScratchTensor(member, scr_level, ncells1);
    Bv.NewAthenaScratchTensor(member, scr_level, ncells1);
    Bsq.NewAthenaScratchTensor(member, scr_level, ncells1);
    v_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    B_d.NewAthenaScratchTensor(member, scr_level, ncells1);

    // Calculate the determinant/volume form
    par_for_inner(member, is, ie, [&](int const i) {
      Real detg = SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                             adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      ivol(i) = 1.0/sqrt(detg);
    });
    // Calculate the lower velocity components
    // TODO: a view should be initialized to zero by default, but it would be well
    // to check that this is actually the case.
    /*par_for_inner(member, is, ie, [&](int const i) {
      Real invW = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          v_d(a, i) += prim(m, IVX + b, k, j, i)*adm.g_dd(m, a, b, k, j, i);
          invW += prim(m, IVX + a, k, j, i)*prim(m, IVX + b, k, j, i)*adm.g_dd(m, a, b, k, j, i);
        }
      }
      invW = 1.0/sqrt(1. + invW);
      for (int a = 0; a < 3; ++a) {
        v_d(a, i) = v_d(a, i)*invW;
      }
    });*/
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](int const i) {
          v_d(a, i) += prim(m, IVX + b, k, j, i)*adm.g_dd(m, a, b, k, j, i);
          iW(i) += prim(m, IVX + a, k, j, i)*prim(m, IVX + b, k, j, i)*adm.g_dd(m, a, b, k, j, i);
          B_d(a, i) += bcc(m, b, k, j, i)*adm.g_dd(m, a, b, k, j, i)*ivol(i);
        });
      }
    }
    // TODO: need a member barrier here.
    member.team_barrier();
    par_for_inner(member, is, ie, [&](int const i) {
      iW(i) = 1.0/sqrt(1. + iW(i));
    });
    for (int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](int const i) {
        Bv(i) += bcc(m, a, k, j, i) * v_d(a, i) * ivol(i);
        Bsq(i) += bcc(m, a, k, j, i) * B_d(a, i) * ivol(i);
      });
    }

    // TODO: member barrier here?

    // Save the fluid quantities
    par_for_inner(member, is, ie, [&](int const i) {
      tmunu.E(m, k, j, i) = (cons(m, IDN, k, j, i) + cons(m, IEN, k, j, i))*ivol(i);
    });
    for (int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](int const i) {
        tmunu.S_d(m, a, k, j, i) = cons(m, IM1 + a, k, j, i)*ivol(i);
      });
    }
    /*par_for_inner(member, is, ie, [&](int const i) {
      tmunu.S_d(m, 0, k, j, i) = cons(m, IM1, k, j, i)*ivol(i);
      tmunu.S_d(m, 1, k, j, i) = cons(m, IM2, k, j, i)*ivol(i);
      tmunu.S_d(m, 2, k, j, i) = cons(m, IM3, k, j, i)*ivol(i);
    });*/
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](int const i) {
          tmunu.S_dd(m, a, b, k, j, i) = cons(m, IM1 + a, k, j, i)*ivol(i)*v_d(b, i)*iW(i)
                                       - (B_d(a, i)*SQR(iW(i)) + Bv(i)*v_d(a, i))*B_d(b, i)
                                       + (prim(m, IPR, k, j, i) 
                                          + 0.5*(Bv(i)*Bv(i)/(iW(i)*iW(i)) + Bsq(i)))
                                       *adm.g_dd(m, a, b, k, j, i);
        });
      }
    }
  });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGR::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
// TODO: Add MHD
template<class EOSPolicy, class ErrorPolicy> template<int NGHOST>
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS(const DvceArray5D<Real> &prim, 
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int ncells1 = indcs.nx1+indcs.ng; // Align scratch buffers with variables
  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &tmunu = pmy_pack->ptmunu->tmunu;
  auto &eos_ = eos.ps.GetEOS();

  int &nhyd = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  const Real mb = eos.ps.GetEOS().GetBaryonMass();

  // Check the number of dimensions to determine which derivatives we need.
  int ndim;
  if (pmy_pack->pmesh->one_d) {
    ndim = 1;
  }
  else if (pmy_pack->pmesh->two_d) {
    ndim = 2;
  }
  else {
    ndim = 3;
  }

  int scr_level = scratch_level;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*2       // scalars
                  + ScrArray2D<Real>::shmem_size(3, ncells1)*2    // vectors
                  + ScrArray2D<Real>::shmem_size(6, ncells1)*2    // symmetric 2 tensors
                  + ScrArray2D<Real>::shmem_size(9, ncells1)*1    // general 2 tensors
                  + ScrArray2D<Real>::shmem_size(18, ncells1)*1;  // symmetric 3 tensors
  par_for_outer("dyngr_coord_terms_loop",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    // Scratch space
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> vol;      // sqrt of determinant of spatial metric
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> E;        // fluid energy density

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d; // lapse 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> S_d;      // matter momentum

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;     // inverse metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> S_uu;     // spatial component of stress tensor

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du; // derivatives of the shift
    
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;    // metric 1st drvts

    vol.NewAthenaScratchTensor(member, scr_level, ncells1);
    E.NewAthenaScratchTensor(member, scr_level, ncells1);
    dalpha_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    S_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    dbeta_du.NewAthenaScratchTensor(member, scr_level, ncells1);
    g_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
    S_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
    dg_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);

    par_for_inner(member, is, ie, [&](int const i) {
      Real detg = SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                             adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      vol(i) = sqrt(detg);
    });
    member.team_barrier();
    
    par_for_inner(member, is, ie, [&](int const i) {
      SpatialInv(1.0/SQR(vol(i)),
                 adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                 adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                 &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
                 &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    });
    member.team_barrier();

    // Metric derivatives
    Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
    for (int a =0; a < ndim; ++a) {
      par_for_inner(member, is, ie, [&](int const i){
        dalpha_d(a, i) = Dx<NGHOST>(a, idx, adm.alpha, m, k, j, i);
      });
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < ndim; ++b) {
        par_for_inner(member, is, ie, [&](int const i){
          dbeta_du(b,a,i) = Dx<NGHOST>(b, idx, adm.beta_u, m, a, k, j, i);
        });
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < ndim; ++c) {
          par_for_inner(member, is, ie, [&](int const i) {
            dg_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.g_dd, m, a, b, k, j, i);
          });
        }
      }
    }

    // Fluid quantities
    par_for_inner(member, is, ie, [&](int const i) {
      Real prim_pt[NPRIM] = {0.0};
      prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
      prim_pt[PVX] = prim(m, IVX, k, j, i);
      prim_pt[PVY] = prim(m, IVY, k, j, i);
      prim_pt[PVZ] = prim(m, IVZ, k, j, i);
      for (int s = 0; s < nscal; s++) {
        prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
      }
      // FIXME: Go back and change to temperature after validating it all works.
      prim_pt[PPR] = prim(m, IPR, k, j, i);
      prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);

      // Get the conserved variables. Note that we don't use PrimitiveSolver here -- that's
      // because we would need to recalculate quantities used in E and S_d in order to get S_dd.
      Real H = prim(m, IDN, k, j, i)*eos_.GetEnthalpy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
      Real usq = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          usq += adm.g_dd(m,a,b,k,j,i)*prim_pt[PVX + a]*prim_pt[PVX + b];
        }
      }
      Real const Wsq = 1.0 + usq;
      Real const W = sqrt(Wsq);
      Real B_u[NMAG] = {bcc(m, IBX, k, j, i)/vol(i), 
                        bcc(m, IBY, k, j, i)/vol(i),
                        bcc(m, IBZ, k, j, i)/vol(i)};
      Real Bv = 0.0;
      Real Bsq = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          Bv += adm.g_dd(m,a,b,k,j,i)*prim_pt[PVX + a]*B_u[b];
          Bsq += adm.g_dd(m,a,b,k,j,i)*B_u[a]*B_u[b];
        }
      }
      Bv = Bv/W;
      Real bsq = Bv*Bv + Bsq/Wsq;

      E(i) = (H*Wsq + Bsq) - prim_pt[PPR] - 0.5*bsq;

      for (int a = 0; a < 3; ++a) {
        S_d(a, i) = 0.0;
        for (int b = 0; b < 3; ++b) {
          S_d(a, i) += ((H*Wsq + Bsq)*prim_pt[PVX + b]/W - Bv*B_u[b])*
                        adm.g_dd(m, a, b, k, j, i);
        }
      }

      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          //S_uu(a,b,i) = H*prim_pt[PVX + a]*prim_pt[PVX + b] + prim_pt[PPR]*g_uu(a,b,i);
          S_uu(a,b,i) = (H + Bsq/Wsq)*prim_pt[PVX + a]*prim_pt[PVX + b] 
                        - B_u[a]*B_u[b]/Wsq 
                        - Bv*(B_u[a]*prim_pt[PVX + b] + B_u[b]*prim_pt[PVX + a])/W
                        + (prim_pt[PPR] + 0.5*bsq)*g_uu(a,b,i);
        }
      }
    });
    member.team_barrier();
    // Raise the indices on S_dd to get S_uu
    /*for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        // TODO: Check that S_uu(a, b, i) is zero here!
        par_for_inner(member, is, ie, [&](int const i) {
          S_uu(a, b, i) = 0.0;
        });
        for (int c = 0; c < 3; ++c) {
          for (int d = 0; d < 3; ++d) {
            par_for_inner(member, is, ie, [&](int const i) {
              S_uu(a, b, i) += tmunu.S_dd(m, c, d, k, j, i) * g_uu(a, c, i) * g_uu(b, d, i);
            });
          }
        }
      }
    }
    // TODO: Check that this is necessary!
    member.team_barrier();*/
    /*for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](int const i) {
          S_uu(a, b, i) = 0.0;
          for (int c = 0; c < 3; ++c) {
            for (int d = 0; d < 3; ++d) {
              S_uu(a, b, i) += tmunu.S_dd(m, c, d, k, j, i) * g_uu(a, c, i) * g_uu(b, d, i);
            }
          }
        });
      }
    }
    member.team_barrier();*/

    // Assemble energy RHS
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](int const i) {
          /*rhs(m, IEN, k, j, i) += dt * vol(i) * (
            adm.alpha(m, k, j, i) * adm.K_dd(m, a, b, k, j, i) * S_uu(a, b, i) -
            g_uu(a, b, i) * tmunu.S_d(m, a, k, j, i) * dalpha_d(b, i));*/
          rhs(m, IEN, k, j, i) += dt * vol(i) * (
            adm.alpha(m, k, j, i) * adm.K_dd(m, a, b, k, j, i) * S_uu(a, b, i) -
            g_uu(a, b, i) * S_d(a, i) * dalpha_d(b, i));
        });
      }
    }

    // Assemble momentum RHS
    for (int a = 0; a < ndim; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          par_for_inner(member, is, ie, [&](int const i) {
            rhs(m,IM1+a, k, j, i) += 0.5 * dt * adm.alpha(m,k,j,i) * vol(i) *
              S_uu(b, c, i) * dg_ddd(a,b,c,i);
          });
        }
        par_for_inner(member, is, ie, [&](int const i) {
          //rhs(m, IM1 + a, k, j, i) += dt * vol(i) * tmunu.S_d(m, b, k, j, i) * dbeta_du(a, b, i);
          rhs(m, IM1 + a, k, j, i) += dt * vol(i) * S_d(b, i) * dbeta_du(a, b, i);
        });
      }
      par_for_inner(member, is, ie, [&](int const i) {
        //rhs(m, IM1 + a, k, j, i) -= dt * vol(i) * tmunu.E(m, k, j, i) * dalpha_d(a, i);
        rhs(m, IM1 + a, k, j, i) -= dt * vol(i) * E(i) * dalpha_d(a, i);
      });
    }
  });
}

// Instantiated templates
template class DynGRPS<Primitive::IdealGas, Primitive::ResetFloor>;
template class DynGRPS<Primitive::PiecewisePolytrope, Primitive::ResetFloor>;

// Macro for defining CoordTerms templates
#define INSTANTIATE_COORD_TERMS(EOSPolicy, ErrorPolicy) \
template \
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<2>(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<3>(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<4>(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs);

INSTANTIATE_COORD_TERMS(Primitive::IdealGas, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::PiecewisePolytrope, Primitive::ResetFloor);

#undef INSTANTIATE_COORD_TERMS

} // namespace dyngr
