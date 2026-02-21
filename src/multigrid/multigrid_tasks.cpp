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

TaskStatus MultigridDriver::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->ClearRecv();
  return tstat;
}

TaskStatus MultigridDriver::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->ClearSend();
  return tstat;
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

TaskStatus MultigridDriver::StartReceive(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->InitRecvMG(pmg->nvar_);
  return tstat;
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

TaskStatus MultigridDriver::FMGProlongateTask(Driver *pdrive, int stage) {
  pmg->FMGProlongatePack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::CalculateFASRHS(Driver *pdrive, int stage) {
  if (current_level_ < fmglevel_){
    pmg->StoreOldData();
    pmg->CalculateFASRHSPack();
  }
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

TaskStatus MultigridDriver::FillFCBoundary(Driver *pdrive, int stage) {
  if (nreflevel_ == 0) return TaskStatus::complete;
  pmg->pbval->FillFineCoarseMGGhosts(pmg->GetCurrentData());
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::PhysicalBoundary(Driver *pdrive, int stage) {
    // do not apply BCs if domain is strictly periodic
  if (pmy_pack_->pmesh->strictly_periodic) return TaskStatus::complete;
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
  id.ircv0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, none);
  id.send0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, none);
  id.recv0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv0);
  id.fc_ghosts0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physb0);

  // Prolongate and correct
  id.prolongate = tl["mg_to_finer"]->AddTask(&MultigridDriver::Prolongate, this, id.fc_ghosts0);

  if (nsmooth > 0) {
    // Fine-level boundary comm after prolongation (reuse for post-smoothing)
    id.ircv1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghosts0);
    id.send1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.prolongate);
    id.recv1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send1);
    id.physb1   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv1);
    id.fc_ghosts_prol = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physb1);

    // Post-smoothing (red-black)
    id.ircvR    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghosts_prol);
    id.smoothR   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothRed, this, id.fc_ghosts_prol);
    id.sendR    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR);
    id.recvR    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendR);
    id.physbR   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR);
    id.fc_ghostsR = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbR);

    id.smoothB   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this, id.fc_ghostsR);
    if (nsmooth > 1) {
      id.ircvB    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghostsR);
      id.sendB    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothB);
      id.recvB    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.ircvB);
      id.physbB   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvB);
      id.fc_ghostsB = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbB);

      id.smoothR2   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothRed, this, id.fc_ghostsB);

      id.ircvR2    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghostsB);
      id.sendR2    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR2);
      id.recvR2    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this, id.ircvR2);
      id.physbR2   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR2);
      id.fc_ghostsR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbR2);

      id.smoothB2   = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this, id.fc_ghostsR2);
    }
  }
  id.clear_send0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearSend, this, none);
  id.clear_recv0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearRecv, this, id.clear_send0);
  // No extra communication at the end
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetMGTaskListFMGProlongate(int ngh)
//! \brief Set the task list for FMG prolongation only (no smoothing)

void MultigridDriver::SetMGTaskListFMGProlongate(int ngh) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_fmg_prolongate");
  tl.emplace(std::make_pair("mg_fmg_prolongate", std::make_shared<TaskList>()));
  TaskID none(0);

  // Boundary comm before prolongation
  id.ircv0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::StartReceive, this, none);
  id.send0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::SendBoundary, this, none);
  id.recv0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0   = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::PhysicalBoundary, this, id.recv0);
  id.fc_ghosts0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::FillFCBoundary, this, id.physb0);

  // FMG prolongation (direct overwrite)
  id.fmg_prolongate = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::FMGProlongateTask, this, id.fc_ghosts0);

  id.clear_send0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ClearSend, this, none);
  id.clear_recv0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ClearRecv, this, id.clear_send0);
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToCoarser(int nsmooth, int ngh)
//! \brief Set the task list for pre smoothing and restriction

void MultigridDriver::SetMGTaskListToCoarser(int nsmooth, int cycle) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_coarser");
  tl.emplace(std::make_pair("mg_to_coarser",std::make_shared<TaskList>()));
  TaskID none(0);

  id.ircv0    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this, none);
  id.send0      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, none);
  id.recv0      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recv0);
  id.fc_ghosts0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physb0);

  id.calc_rhs   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::CalculateFASRHS, this, id.fc_ghosts0);

  if (nsmooth > 0) {
    id.ircvR      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghosts0);
    id.smoothR    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this, id.calc_rhs);
    id.sendR      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR);
    id.recvR      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendR);
    id.physbR     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR);
    id.fc_ghostsR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbR);

    id.ircvB      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this, id.fc_ghostsR);
    id.smoothB    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this, id.fc_ghostsR);
    id.sendB      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothB);
    id.recvB      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendB);
    id.physbB     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvB);
    id.fc_ghostsB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbB);
    if (nsmooth > 1) {
      id.smoothR2   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this, id.fc_ghostsB);

      id.ircvR2    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this, id.smoothR2);
      id.sendR2      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothR2);
      id.recvR2     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.sendR2);
      id.physbR2     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvR2);
      id.fc_ghostsR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbR2);

      id.smoothB2   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this, id.fc_ghostsR2);

      id.sendB2      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this, id.smoothB2);
      id.recvB2      = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this, id.ircvB2);
      id.physbB2     = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this, id.recvB2);
      id.fc_ghostsB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this, id.physbB2);

      id.restrict_  = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this, id.fc_ghostsB2);
    } else {
      id.restrict_  = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this, id.fc_ghostsB);
    }
  } else {
    id.restrict_  = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this, id.calc_rhs);
  }
  id.clear_send0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::ClearSend, this, none);
  id.clear_recv0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::ClearRecv, this, id.clear_send0);

}