//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_task_list.cpp
//! \brief functions for MultigridTaskList class

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>    // setprecision

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "multigrid.hpp"

//namespace multigrid{ // NOLINT (build/namespace)

//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::DoTaskListOneStage(MultigridDriver *pmd)
//! \brief completes all tasks in this list, will not return until all are tasks done
void MultigridDriver::DoTaskListOneStage() {
  return;
}

TaskStatus MultigridDriver::StartReceiveFluxCons(Driver *pdrive, int stage) {
  //pmg->pmgbval->StartReceivingMultigrid(pmg->btype, true);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::SendBoundary(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->PackAndSendMG(pmg->GetCurrentData());
  return tstat;
}

TaskStatus MultigridDriver::RecvBoundary(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->RecvAndUnpackMG(pmg->GetCurrentData());
  return tstat;
}

TaskStatus MultigridDriver::StartReceiveForProlongation(Driver *pdrive, int stage) {
  //pmg->pmgbval->StartReceivingMultigrid(pmg->btype, true);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::ClearBoundary(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = TaskStatus::complete;//pmg->pmgbval->ClearBoundaryMultigrid(pmg->btype);
  return tstat;
}

TaskStatus MultigridDriver::ClearBoundaryFluxCons(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = TaskStatus::complete;//pmg->pmgbval->ClearBoundaryMultigrid(pmg->btypef);
  return tstat;
}

TaskStatus MultigridDriver::SendBoundaryFluxCons(Driver *pdrive, int stage) {
  TaskStatus tstat;
  //tstat = pmg->pmgbval->SendMultigridBoundaryBuffers(pmg->btypef, false);
  //if (!tstat)
  //  return tstat;
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::SendBoundaryForProlongation(Driver *pdrive, int stage) {
  //if (!(pmg->pmgbval->SendMultigridBoundaryBuffers(pmg->btype, pmg->pmy_driver_->ffas_)))
  //  return MGTaskStatus::fail;
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::ReceiveBoundaryFluxCons(Driver *pdrive, int stage) {
  //if (!(pmg->pmgbval->ReceiveMultigridBoundaryBuffers(pmg->btypef, false)))
  //  return MGTaskStatus::fail;
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::ReceiveBoundaryForProlongation(Driver *pdrive, int stage) {
  //if (!(pmg->pmgbval->ReceiveMultigridBoundaryBuffers(pmg->btype,
  //                                                    pmg->pmy_driver_->ffas_)))
  //  return TaskStatus::fail;
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::SmoothRed(Driver *pdrive, int stage) {
  pmg->SmoothPack(coffset_);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::SmoothBlack(Driver *pdrive, int stage) {
  pmg->SmoothPack(1-coffset_);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::Restrict(Driver *pdrive, int stage) {
  pmg->RestrictPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::Prolongate(Driver *pdrive, int stage) {
  pmg->ProlongateAndCorrectPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::CalculateFASRHS(Driver *pdrive, int stage) {
  if (current_level_ < ntotallevel_-1){
    pmg->StoreOldData();
  }
  pmg->CalculateFASRHSPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::ProlongateBoundary(Driver *pdrive, int stage) {
  //pmg->pmgbval->ProlongateMultigridBoundariesFluxCons();
  return TaskStatus::complete;
}


TaskStatus MultigridDriver::ProlongateBoundaryForProlongation(Driver *pdrive, int stage) {
  //pmg->pmgbval->ProlongateMultigridBoundaries(pmg->pmy_driver_->ffas_, false);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::PhysicalBoundary(Driver *pdrive, int stage) {
    // do not apply BCs if domain is strictly periodic
  if (pmy_pack_->pmesh->strictly_periodic) return TaskStatus::complete;

  //// physical BCs
  //pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);
//
  //// user BCs
  //if (pmy_pack->pmesh->pgen->user_bcs) {
  //  (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  //}
  ////pmg->pmgbval->ApplyPhysicalBoundaries(0, false);
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToFiner(int nsmooth, int ngh, int flag)
//! \brief Set the task list for prolongation and post smoothing

void MultigridDriver::SetMGTaskListToFiner(int nsmooth, int ngh) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_finer");
  tl.emplace(std::make_pair("mg_to_finer", std::make_shared<TaskList>()));
  TaskID none(0);

  // Coarse-level boundary comm before prolongation (needed once)
  id.send0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, none);
  id.recv0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv0);

  // Prolongate and correct
  id.prolongate = tl["mg_to_finer"]->AddTask(&MultigridDriver::Prolongate, this, id.physb0);

  // Fine-level boundary comm after prolongation (reuse for post-smoothing)
  id.send1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.prolongate);
  id.recv1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send1);
  id.physb1   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv1);
  

  // Post-smoothing (red-black) — reuse ghost data from above
  id.smoothR   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothRed, this, id.physb1);
  id.sendR    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR);
  id.recvR    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendR);
  id.physbR   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR);
  id.smoothB   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this, id.physbR );
  // No extra communication at the end
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToCoarser(int nsmooth, int ngh)
//! \brief Set the task list for pre smoothing and restriction

void MultigridDriver::SetMGTaskListToCoarser(int nsmooth, int cycle) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_coarser");
  tl.emplace(std::make_pair("mg_to_coarser",std::make_shared<TaskList>()));
  TaskID none(0);
  id.send0      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, none);
  id.recv0      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv0);
  id.calc_rhs   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::CalculateFASRHS, this, id.physb0);
  // Pre-smoothing (red-black)
  id.smoothR   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this, id.calc_rhs);
  id.sendR      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR);
  id.recvR      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendR);
  id.physbR     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR);
  id.smoothB   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this, id.physbR);
  id.sendB      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothB);
  id.recvB      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendB);
  id.physbB     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvB);
  // Restriction — still has fresh ghost data; no extra comm needed if ngh is large enough
  // Restriction:  Calculates defect and restricts to coarser level -> rhs on coarser level
  // For FAS, rhs = R(defect) + Lap(u) is calculated in CalculateFASRHS above
  // For FAS, u is also restricted in RestrictPack()
  id.restrict  = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this, id.physbB);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListBoundaryCommunication()
//! \brief Set the task list for boundary communication

//void MultigridTaskList::SetMGTaskListBoundaryCommunication() {
//  ClearTaskList();
//  AddMultigridTask(MG_STARTRECVL, NONE);
//  AddMultigridTask(MG_SENDBNDL,   MG_STARTRECVL);
//  AddMultigridTask(MG_RECVBNDL,   MG_STARTRECVL);
//  if (pmy_mgdriver_->nreflevel_ > 0) {
//    AddMultigridTask(MG_PRLNGFCL, MG_SENDBNDL|MG_RECVBNDL);
//    AddMultigridTask(MG_PHYSBNDL, MG_PRLNGFCL);
//  } else {
//    AddMultigridTask(MG_PHYSBNDL, MG_RECVBNDL|MG_SENDBNDL);
//  }
//  AddMultigridTask(MG_CLEARBNDL,  MG_PHYSBNDL);
//}
//} // namespace multigrid