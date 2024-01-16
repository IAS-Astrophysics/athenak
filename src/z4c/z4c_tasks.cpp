//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_tasks.cpp
//! \brief functions that control z4c tasks in the appropriate task list

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "z4c/z4c.hpp"

namespace z4c {
//----------------------------------------------------------------------------------------
//! \fn  void Z4c::AssembleZ4cTasks
//! \brief Adds z4c tasks to appropriate task lists used by time integrators.
//  Called by MeshBlockPack::AddPhysics() function directly after Z4c constrctor

void Z4c::AssembleZ4cTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  printf("AssembleZ4cTasks\n");
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  // "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&Z4c::InitRecv, this, none);

  // "stagen" task list
  // id.ptrack = tl["stagen"]->AddTask(&Z4c::PunctureTracker, this, none);
  id.copyu = tl["stagen"]->AddTask(&Z4c::CopyU, this, none); // id.ptrack);

  switch (indcs.ng) {
      case 2: id.crhs  = tl["stagen"]->AddTask(&Z4c::CalcRHS<2>, this, id.copyu);
              break;
      case 3: id.crhs  = tl["stagen"]->AddTask(&Z4c::CalcRHS<3>, this, id.copyu);
              break;
      case 4: id.crhs  = tl["stagen"]->AddTask(&Z4c::CalcRHS<4>, this, id.copyu);
              break;
  }
  id.sombc = tl["stagen"]->AddTask(&Z4c::Z4cBoundaryRHS, this, id.crhs);
  id.expl  = tl["stagen"]->AddTask(&Z4c::ExpRKUpdate, this, id.sombc);
  id.restu = tl["stagen"]->AddTask(&Z4c::RestrictU, this, id.expl);
  id.sendu = tl["stagen"]->AddTask(&Z4c::SendU, this, id.restu);
  id.recvu = tl["stagen"]->AddTask(&Z4c::RecvU, this, id.sendu);
  id.bcs   = tl["stagen"]->AddTask(&Z4c::ApplyPhysicalBCs, this, id.recvu);
  id.prol  = tl["stagen"]->AddTask(&Z4c::Prolongate, this, id.bcs);
  id.algc  = tl["stagen"]->AddTask(&Z4c::EnforceAlgConstr, this, id.prol);
  id.z4tad = tl["stagen"]->AddTask(&Z4c::Z4cToADM_, this, id.algc);
  id.admc  = tl["stagen"]->AddTask(&Z4c::ADMConstraints_, this, id.z4tad);
  id.newdt = tl["stagen"]->AddTask(&Z4c::NewTimeStep, this, id.admc);
  // "after_stagen" task list
  id.csend = tl["after_stagen"]->AddTask(&Z4c::ClearSend, this, none);
  id.crecv = tl["after_stagen"]->AddTask(&Z4c::ClearRecv, this, id.csend);

  // if (pmy_pack->pmesh->ncycle%64 == 0) {
    // place holder for horizon finder
  // }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::InitRecv
//! \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Wave variables.

TaskStatus Z4c::InitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->InitRecv(nz4c);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nz4c);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::ClearRecv
//! \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Z4c::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR check receives of restricted fluxes of U complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxRecv();
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ClearSend
//! \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Z4c::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR check sends of restricted fluxes of U complete
  // do not check flux send for ICs (stage < 0)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxSend();
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::CopyU
//! \brief  copy u0 --> u1 in first stage

TaskStatus Z4c::CopyU(Driver *pdrive, int stage) {
  auto integrator = pdrive->integrator;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nz4c;
  auto &u0 = pmy_pack->pz4c->u0;
  auto &u1 = pmy_pack->pz4c->u1;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator.
  // Important to use vector inner loop for good performance on cpus
  if (integrator == "rk4") {
    Real &delta = pdrive->delta[stage-1];
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      par_for("CopyCons", DevExeSpace(),0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
        u1(m,n,k,j,i) += delta*u0(m,n,k,j,i);
      });
    }
  } else {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::SendU
//! \brief sends cell-centered conserved variables

TaskStatus Z4c::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RecvU
//! \brief receives cell-centered conserved variables

TaskStatus Z4c::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::EnforceAlgConstr
//! \brief

TaskStatus Z4c::EnforceAlgConstr(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    AlgConstr(pmy_pack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ADMToZ4c_
//! \brief

TaskStatus Z4c::Z4cToADM_(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    Z4cToADM(pmy_pack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ADM_Constraints_
//! \brief

TaskStatus Z4c::ADMConstraints_(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (stage == pdrive->nexp_stages) {
    switch (indcs.ng) {
      case 2: ADMConstraints<2>(pmy_pack);
              break;
      case 3: ADMConstraints<3>(pmy_pack);
              break;
      case 4: ADMConstraints<4>(pmy_pack);
              break;
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RestrictU
//! \brief

TaskStatus Z4c::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Z4c::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse boundaries with SMR/AMR

TaskStatus Z4c::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
//    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    pbval_u->ProlongateCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ApplyPhysicalBCs
//! \brief

TaskStatus Z4c::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // only apply BCs if domain is not strictly periodic
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    // physical BCs
    pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);

    // user BCs
    if (pmy_pack->pmesh->pgen->user_bcs) {
      (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
    }
  }
  return TaskStatus::complete;
}

} // namespace z4c
