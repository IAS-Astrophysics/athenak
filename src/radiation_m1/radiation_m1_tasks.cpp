//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_tasks.cpp
//  \brief functions that control M1 tasks stored in tasklists in MeshBlockPack

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "tasklist/task_list.hpp"

namespace radiationm1 {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::AssembleRadiationM1Tasks
//! \brief Adds radiatoin tasks to appropriate task lists used by time
//! integrators. Called by MeshBlockPack::AddPhysics() function directly after
//! Hydro constructor. Many of the functions in the task list are implemented in
//! this file because they are simple, or they are wrappers that call one or
//! more other functions.
//!
//! "before_stagen" tasks are those that must be cmpleted over all MeshBlocks
//! BEFORE each stage can be run (such as posting MPI receives, setting
//! BoundaryCommStatus flags, etc)
//!
//! "stagen" tasks are those performed DURING each stage
//!
//! "after_stagen" tasks are those that can only be completed AFTER all the
//! "stagen" tasks are completed over ALL MeshBlocks for each stage, such as
//! clearing all MPI calls, etc.
//!
//! In addition there are "before_timeintegrator" and "after_timeintegrator"
//! task lists in the tl map, which are generally used for operator split tasks.

void RadiationM1::AssembleRadiationM1Tasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // assemble "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&RadiationM1::InitRecv, this, none);

  // assemble "stagen" task list
  id.copyu = tl["stagen"]->AddTask(&RadiationM1::CopyCons, this, none);
  id.flux = tl["stagen"]->AddTask(&RadiationM1::Fluxes, this, id.copyu);
  id.sendf = tl["stagen"]->AddTask(&RadiationM1::SendFlux, this, id.flux);
  id.recvf = tl["stagen"]->AddTask(&RadiationM1::RecvFlux, this, id.sendf);
  id.rkupdt = tl["stagen"]->AddTask(&RadiationM1::RKUpdate, this, id.recvf);
  id.srctrms =
      tl["stagen"]->AddTask(&RadiationM1::RadiationM1SrcTerms, this, id.rkupdt);
  id.restu = tl["stagen"]->AddTask(&RadiationM1::RestrictU, this, id.srctrms);
  id.sendu = tl["stagen"]->AddTask(&RadiationM1::SendU, this, id.restu);
  id.recvu = tl["stagen"]->AddTask(&RadiationM1::RecvU, this, id.sendu);
  id.bcs =
      tl["stagen"]->AddTask(&RadiationM1::ApplyPhysicalBCs, this, id.recvu);
  id.prol = tl["stagen"]->AddTask(&RadiationM1::Prolongate, this, id.bcs);
  id.newdt = tl["stagen"]->AddTask(&RadiationM1::NewTimeStep, this, id.prol);

  // assemble "after_stagen" task list
  id.csend = tl["after_stagen"]->AddTask(&RadiationM1::ClearSend, this, none);
  // although RecvFlux/U functions check that all recvs complete, add ClearRecv
  // to task list anyways to catch potential bugs in MPI communication logic
  id.crecv =
      tl["after_stagen"]->AddTask(&RadiationM1::ClearRecv, this, id.csend);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI),
//! and initialize all boundary receive status flags to waiting (with or without
//! MPI).

TaskStatus RadiationM1::InitRecv(Driver *pdrive, int stage) {
  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nvars);
  if (tstat != TaskStatus::complete)
    return tstat;

  // with SMR/AMR post receives for fluxes of U
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nvars);
  }
  if (tstat != TaskStatus::complete)
    return tstat;

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void RadiationM1::CopyCons
//! \brief Simple task list function that copies u0 --> u1 in first stage.
//! Extended to
//!  handle RK register logic at given stage

TaskStatus RadiationM1::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  } else {
    if (pdrive->integrator == "rk4") {
      // parallel loop to update u1 with u0 at later stages, only for rk4
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int is = indcs.is, ie = indcs.ie;
      int js = indcs.js, je = indcs.je;
      int ks = indcs.ks, ke = indcs.ke;
      int nmb1 = pmy_pack->nmb_thispack - 1;
      int nvar = nvars;
      auto &u0 = pmy_pack->pradm1->u0;
      auto &u1 = pmy_pack->pradm1->u1;
      Real &delta = pdrive->delta[stage - 1];
      par_for(
          "rk4_copy_cons", DevExeSpace(), 0, nmb1, 0, nvar - 1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
            u1(m, n, k, j, i) += delta * u0(m, n, k, j, i);
          });
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationM1::Fluxes
//! \brief Wrapper task list function that calls everything necessary to compute
//! fluxes of conserved variables

TaskStatus RadiationM1::Fluxes(Driver *pdrive, int stage) {
  // compute fluxes
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes
//! of conserved variables at fine/coarse boundaries

TaskStatus RadiationM1::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryVaLUES function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->PackAndSendFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes
//! of conserved variables at fine/coarse boundaries

TaskStatus RadiationM1::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::RadiationM1SrcTerms
//! \brief Wrapper task list function to apply source terms to conservative vars
//! Note source terms must be computed using only primitives (w0), as the
//! conserved variables (u0) have already been partially updated when this fn
//! called.

TaskStatus RadiationM1::RadiationM1SrcTerms(Driver *pdrive, int stage) {
  Real beta_dt = (pdrive->beta[stage - 1]) * (pmy_pack->pmesh->dt);

  // Add source terms

  // Add user source terms
  if (pmy_pack->pmesh->pgen->user_srcs) {
    (pmy_pack->pmesh->pgen->user_srcs_func)(pmy_pack->pmesh, beta_dt);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::RestrictU
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus RadiationM1::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved
//! variables

TaskStatus RadiationM1::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved
//! variables

TaskStatus RadiationM1::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::ApplyPhysicalBCs
//! \brief Wrapper task list function to call funtions that set physical and
//! user BCs,

TaskStatus RadiationM1::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic)
    return TaskStatus::complete;

  // physical BCs
  // add radiation M1 BCs
  // pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive)
//! variables at fine/coarse boundaries with SMR/AMR

TaskStatus RadiationM1::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) { // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    pbval_u->ProlongateCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.
//! Used in TaskList and in Driver::InitBoundaryValuesAndPrimitives() If
//! stage=(last stage):      clears sends of U, Flx_U, U_OA, U_Shr If (last
//! stage)>stage>=(0): clears sends of U, Flx_U,       U_Shr If stage=(-1):
//! clears sends of U If stage=(-4):              clears sends of U_Shr

TaskStatus RadiationM1::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat;
  // check sends of U complete
  if ((stage >= 0) || (stage == -1)) {
    tstat = pbval_u->ClearSend();
    if (tstat != TaskStatus::complete)
      return tstat;
  }

  // with SMR/AMR check sends of restricted fluxes of U complete
  // do not check flux send for ICs (stage < 0)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxSend();
    if (tstat != TaskStatus::complete)
      return tstat;
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have
//! completed. Used in TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears recvs of U, Flx_U, U_OA, U_Shr
//! If (last stage)>stage>=(0): clears recvs of U, Flx_U,       U_Shr
//! If stage=(-1):              clears recvs of U
//! If stage=(-4):              clears recvs of                 U_Shr

TaskStatus RadiationM1::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat;
  // check receives of U complete
  if ((stage >= 0) || (stage == -1)) {
    tstat = pbval_u->ClearRecv();
    if (tstat != TaskStatus::complete)
      return tstat;
  }

  // with SMR/AMR check receives of restricted fluxes of U complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxRecv();
    if (tstat != TaskStatus::complete)
      return tstat;
  }

  return tstat;
}

} // namespace radiationm1
