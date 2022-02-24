//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_tasks.cpp
//  \brief implementation of functions that control Hydro tasks in the four task lists:
//  stagestart_tl, stagerun_tl, stageend_tl, operatorsplit_tl (currently not used)

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleHydroTasks
//  \brief Adds hydro tasks to stage start/run/end task lists
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

void Hydro::AssembleHydroTasks(TaskList &start, TaskList &run, TaskList &end) {
  TaskID none(0);

  // start task list
  id.irecv = start.AddTask(&Hydro::InitRecv, this, none);

  // run task list
  id.copyu = run.AddTask(&Hydro::CopyCons, this, none);
  // select which calculate_flux function to add based on rsolver_method
  if (rsolver_method == Hydro_RSolver::advect) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::advect>,this,id.copyu);
  } else if (rsolver_method == Hydro_RSolver::llf) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::llf>,this,id.copyu);
  } else if (rsolver_method == Hydro_RSolver::hlle) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::hlle>,this,id.copyu);
  } else if (rsolver_method == Hydro_RSolver::hllc) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::hllc>,this,id.copyu);
  } else if (rsolver_method == Hydro_RSolver::roe) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::roe>,this,id.copyu);
  } else if (rsolver_method == Hydro_RSolver::llf_sr) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::llf_sr>,this,id.copyu);
  } else if (rsolver_method == Hydro_RSolver::hlle_sr) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::hlle_sr>,this,id.copyu);
  } else if (rsolver_method == Hydro_RSolver::hllc_sr) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::hllc_sr>,this,id.copyu);
  } else if (rsolver_method == Hydro_RSolver::hlle_gr) {
    id.flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::hlle_gr>,this,id.copyu);
  }
  // now the rest of the Hydro run tasks
  id.sendf = run.AddTask(&Hydro::SendFlux, this, id.flux);
  id.recvf = run.AddTask(&Hydro::RecvFlux, this, id.sendf);
  id.expl  = run.AddTask(&Hydro::ExpRKUpdate, this, id.recvf);
  id.restu = run.AddTask(&Hydro::RestrictU, this, id.expl);
  id.sendu = run.AddTask(&Hydro::SendU, this, id.restu);
  id.recvu = run.AddTask(&Hydro::RecvU, this, id.sendu);
  id.bcs   = run.AddTask(&Hydro::ApplyPhysicalBCs, this, id.recvu);
  id.c2p   = run.AddTask(&Hydro::ConToPrim, this, id.bcs);
  id.newdt = run.AddTask(&Hydro::NewTimeStep, this, id.c2p);

  // end task list
  id.clear = end.AddTask(&Hydro::ClearSend, this, none);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Hydro variables.

TaskStatus Hydro::InitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->InitRecv(nhydro+nscalars);
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nhydro+nscalars);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Hydro::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxRecv();
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Hydro::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxSend();
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CopyCons
//  \brief  copy u0 --> u1 in first stage

TaskStatus Hydro::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::SendU
//  \brief sends cell-centered conserved variables

TaskStatus Hydro::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::RecvU
//  \brief receives cell-centered conserved variables

TaskStatus Hydro::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::SendFlux
//  \brief

TaskStatus Hydro::SendFlux(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  TaskStatus tstat = pbval_u->PackAndSendFluxCC(uflx);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::RecvFlux
//  \brief

TaskStatus Hydro::RecvFlux(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  TaskStatus tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::RestrictU
//  \brief

TaskStatus Hydro::RestrictU(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  pmy_pack->pmesh->RestrictCC(u0, coarse_u0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ApplyPhysicalBCs
//  \brief

TaskStatus Hydro::ApplyPhysicalBCs(Driver *pdrive, int stage) {
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

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ConToPrim
//! \brief Convert conservative to primitive variables over entire mesh, including gz.

TaskStatus Hydro::ConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  peos->ConsToPrim(u0, w0, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

} // namespace hydro
