//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_tasks.cpp
//  \brief implementation of functions that control z4c tasks in the task list:
//  stagestart_tl, stagerun_tl, stageend_tl, operatorsplit_tl (currently not used)

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
//  \brief Adds z4c tasks to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysics() function directly after Z4c constrctor
//
//  Stage start tasks are those that must be completed over all MeshBlocks before EACH
//  stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//
//  Stage run tasks are those performed in EACH stage
//
//  Stage end tasks are those that can only be cmpleted after all the stage run tasks are
//  finished over all MeshBlocks for EACH stage, such as clearing all MPI non-blocking
//  sends, etc.

void Z4c::AssembleZ4cTasks(TaskList &start, TaskList &run, TaskList &end) {
  TaskID none(0);
  printf("AssembleZ4cTasks\n");
  // start task list
  id.irecv = start.AddTask(&Z4c::InitRecv, this, none);

  // run task list
  id.copyu = run.AddTask(&Z4c::CopyU, this, none);
  id.crhs  = run.AddTask(&Z4c::CalcRHS, this, id.copyu);
  id.sombc = run.AddTask(&Z4c::Z4cBoundaryRHS, this, id.crhs);
  id.expl  = run.AddTask(&Z4c::ExpRKUpdate, this, id.sombc);
  id.restu = run.AddTask(&Z4c::RestrictU, this, id.expl);
  id.sendu = run.AddTask(&Z4c::SendU, this, id.restu);
  id.recvu = run.AddTask(&Z4c::RecvU, this, id.sendu);
  id.bcs   = run.AddTask(&Z4c::ApplyPhysicalBCs, this, id.recvu);
  id.algc  = run.AddTask(&Z4c::EnforceAlgConstr, this, id.bcs);
  id.z4tad = run.AddTask(&Z4c::Z4cToADM_, this, id.algc);
  id.admc  = run.AddTask(&Z4c::ADMConstraints_, this, id.z4tad); 
  id.newdt = run.AddTask(&Z4c::NewTimeStep, this, id.admc);
  // end task list
  id.clear = end.AddTask(&Z4c::ClearSend, this, none);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Wave variables.

TaskStatus Z4c::InitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->InitRecv(N_Z4c);
  if (tstat != TaskStatus::complete) return tstat;

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::ClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Z4c::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Z4c::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::CopyU
//  \brief  copy u0 --> u1 in first stage

TaskStatus Z4c::CopyU(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::SendU
//  \brief sends cell-centered conserved variables

TaskStatus Z4c::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RecvU
//  \brief receives cell-centered conserved variables

TaskStatus Z4c::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::EnforceAlgConstr
//  \brief

TaskStatus Z4c::EnforceAlgConstr(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    AlgConstr(pmy_pack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ADMToZ4c_
//  \brief

TaskStatus Z4c::Z4cToADM_(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    Z4cToADM(pmy_pack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ADM_Constraints_
//  \brief

TaskStatus Z4c::ADMConstraints_(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    ADMConstraints(pmy_pack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RestrictU
//  \brief

TaskStatus Z4c::RestrictU(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  pmy_pack->pmesh->RestrictCC(u0, coarse_u0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ApplyPhysicalBCs
//  \brief

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
