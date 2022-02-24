//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_tasks.cpp
//  \brief implementation of functions that control MHD tasks in the four task lists:
//  stagestart_tl, stagerun_tl, stageend_tl, operatorsplit_tl (currently not used)

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::AssembleMHDTasks
//  \brief Adds mhd tasks to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysics() function directly after MHD constrctr

void MHD::AssembleMHDTasks(TaskList &start, TaskList &run, TaskList &end) {
  TaskID none(0);

  // start task list
  id.irecv = start.AddTask(&MHD::InitRecv, this, none);

  // run task list
  id.copyu = run.AddTask(&MHD::CopyCons, this, none);
  // select which calculate_flux function to add based on rsolver_method
  if (rsolver_method == MHD_RSolver::advect) {
    id.flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::advect>, this, id.copyu);
  } else if (rsolver_method == MHD_RSolver::llf) {
    id.flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::llf>, this, id.copyu);
  } else if (rsolver_method == MHD_RSolver::hlle) {
    id.flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::hlle>, this, id.copyu);
  } else if (rsolver_method == MHD_RSolver::hlld) {
    id.flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::hlld>, this, id.copyu);
  } else if (rsolver_method == MHD_RSolver::llf_sr) {
    id.flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::llf_sr>, this, id.copyu);
  } else if (rsolver_method == MHD_RSolver::hlle_sr) {
    id.flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::hlle_sr>, this, id.copyu);
  } else if (rsolver_method == MHD_RSolver::hlle_gr) {
    id.flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::hlle_gr>, this, id.copyu);
  }
  id.sendf = run.AddTask(&MHD::SendFlux, this, id.flux);
  id.recvf = run.AddTask(&MHD::RecvFlux, this, id.sendf);
  id.expl  = run.AddTask(&MHD::ExpRKUpdate, this, id.recvf);
  id.restu = run.AddTask(&MHD::RestrictU, this, id.expl);
  id.sendu = run.AddTask(&MHD::SendU, this, id.restu);
  id.recvu = run.AddTask(&MHD::RecvU, this, id.sendu);
  id.efld  = run.AddTask(&MHD::CornerE, this, id.recvu);
  id.sende = run.AddTask(&MHD::SendE, this, id.efld);
  id.recve = run.AddTask(&MHD::RecvE, this, id.sende);
  id.ct    = run.AddTask(&MHD::CT, this, id.recve);
  id.restb = run.AddTask(&MHD::RestrictB, this, id.ct);
  id.sendb = run.AddTask(&MHD::SendB, this, id.restb);
  id.recvb = run.AddTask(&MHD::RecvB, this, id.sendb);
  id.bcs   = run.AddTask(&MHD::ApplyPhysicalBCs, this, id.recvb);
  id.c2p   = run.AddTask(&MHD::ConToPrim, this, id.bcs);
  id.newdt = run.AddTask(&MHD::NewTimeStep, this, id.c2p);

  // end task list
  id.clear = end.AddTask(&MHD::ClearSend, this, none);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI).  Note this must be done for
//  communication of BOTH conserved (cell-centered) and face-centered fields

TaskStatus MHD::InitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->InitRecv(nmhd+nscalars);
  if (tstat != TaskStatus::complete) return tstat;

  tstat = pbval_b->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nmhd+nscalars);
    if (tstat != TaskStatus::complete) return tstat;

    tstat = pbval_b->InitFluxRecv(3);
    if (tstat != TaskStatus::complete) return tstat;
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue
//  With MHD, clears both receives of U and B

TaskStatus MHD::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  tstat = pbval_b->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;

    tstat = pbval_b->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void MHD::ClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue
//  With MHD, clears both sends of U and B

TaskStatus MHD::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  tstat = pbval_b->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;

    tstat = pbval_b->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;
  }
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void MHD::CopyCons
//  \brief  copy u0 --> u1, and b0 --> b1 in first stage

TaskStatus MHD::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::SendU
//  \brief sends cell-centered conserved variables

TaskStatus MHD::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RecvU
//  \brief receives cell-centered conserved variables

TaskStatus MHD::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::SendE
//  \brief sends face-centered magnetic fields

TaskStatus MHD::SendE(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  TaskStatus tstat = pbval_b->PackAndSendFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RecvE
//  \brief receives face-centered magnetic fields

TaskStatus MHD::RecvE(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  TaskStatus tstat = pbval_b->RecvAndUnpackFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::SendB
//  \brief sends face-centered magnetic fields

TaskStatus MHD::SendB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->PackAndSendFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RecvB
//  \brief receives face-centered magnetic fields

TaskStatus MHD::RecvB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->RecvAndUnpackFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::SendFlux
//  \brief

TaskStatus MHD::SendFlux(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  TaskStatus tstat = pbval_u->PackAndSendFluxCC(uflx);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RecvFlux
//  \brief

TaskStatus MHD::RecvFlux(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  TaskStatus tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RestrictU
//  \brief

TaskStatus MHD::RestrictU(Driver *pdrive, int stage) {
  // Skip if this calculation does not use SMR/AMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  pmy_pack->pmesh->RestrictCC(u0, coarse_u0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RestrictB
//  \brief

TaskStatus MHD::RestrictB(Driver *pdrive, int stage) {
  // Skip if this calculation does not use SMR/AMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  pmy_pack->pmesh->RestrictFC(b0, coarse_b0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ApplyPhysicalBCs
//  \brief

TaskStatus MHD::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // only apply BCs if domain is not strictly periodic
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    // physical BCs
    pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);
    pbval_b->BFieldBCs((pmy_pack), (pbval_b->b_in), b0);

    // user BCs
    if (pmy_pack->pmesh->pgen->user_bcs) {
      (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ConToPrim
//! \brief Convert conservative to primitive variables over entire mesh, including gz.

TaskStatus MHD::ConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  peos->ConsToPrim(u0, b0, w0, bcc0, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

} // namespace mhd
