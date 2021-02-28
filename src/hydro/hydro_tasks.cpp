//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_tasks.cpp
//  \brief implementation of functions that control Hydro tasks in the four task lists:
//  stagestart_tl, stagerun_tl, stageend_tl
//  operatorsplit_tl

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "utils/create_mpitag.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleStageStartTasks
//  \brief adds Hydro tasks to stage start TaskList
//  These are tasks that must be cmpleted (such as posting MPI receives, setting 
//  BoundaryCommStatus flags, etc) over all MeshBlocks before EACH stage can be run.

void Hydro::AssembleStageStartTasks(TaskList &tl, TaskID start)
{
  auto hydro_init = tl.AddTask(&Hydro::InitRecv, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleStageRunTasks
//  \brief adds Hydro tasks to stage run TaskList

void Hydro::AssembleStageRunTasks(TaskList &tl, TaskID start)
{
  auto hydro_copycons = tl.AddTask(&Hydro::CopyCons, this, start);
  auto hydro_fluxes = tl.AddTask(&Hydro::CalcFluxes, this, hydro_copycons);
  auto visc_fluxes = tl.AddTask(&Hydro::ViscousFluxes, this, hydro_fluxes);
  auto hydro_update = tl.AddTask(&Hydro::Update, this, visc_fluxes);
  auto hydro_src = tl.AddTask(&Hydro::ApplyUnsplitSourceTerms, this, hydro_update);
  auto hydro_send = tl.AddTask(&Hydro::SendU, this, hydro_src); // hydro_update);
  auto hydro_recv = tl.AddTask(&Hydro::RecvU, this, hydro_send);
  auto hydro_phybcs = tl.AddTask(&Hydro::ApplyPhysicalBCs, this, hydro_recv);
  auto hydro_con2prim = tl.AddTask(&Hydro::ConToPrim, this, hydro_phybcs);
  auto hydro_newdt = tl.AddTask(&Hydro::NewTimeStep, this, hydro_con2prim);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleStageEndTasks
//  \brief adds Hydro tasks to stage end TaskList
//  These are tasks that can only be cmpleted after all the stage run tasks are finished
//  over all MeshBlocks for EACH stage, such as clearing all MPI non-blocking sends, etc.

void Hydro::AssembleStageEndTasks(TaskList &tl, TaskID start)
{
  auto hydro_clear = tl.AddTask(&Hydro::ClearSend, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssmebleOperatorSplitTasks
//  \brief adds Hydro tasks to operator split TaskList
  
void Hydro::AssembleOperatorSplitTasks(TaskList &tl, TaskID start)
{ 
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Hydro variables.

TaskStatus Hydro::InitRecv(Driver *pdrive, int stage)
{
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications for cell-centered conserved variables
  auto &rbufu = pbval_u->recv_buf;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::FluidCons_ID);
          auto recv_data = Kokkos::subview(rbufu[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr[n].rank.h_view(m), tag, MPI_COMM_WORLD, &(rbufu[n].comm_req[m]));
        }
#endif
        // initialize boundary receive status flag
        rbufu[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Hydro::ClearRecv(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for U to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Hydro::ClearSend(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for U to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CopyCons
//  \brief  copy u0 --> u1 in first stage

TaskStatus Hydro::CopyCons(Driver *pdrive, int stage)
{
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::SendU
//  \brief sends cell-centered conserved variables

TaskStatus Hydro::SendU(Driver *pdrive, int stage) 
{
  TaskStatus tstat = pbval_u->SendBuffersCC(u0, VariablesID::FluidCons_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::RecvU
//  \brief receives cell-centered conserved variables

TaskStatus Hydro::RecvU(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_u->RecvBuffersCC(u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ConToPrim
//  \brief

TaskStatus Hydro::ConToPrim(Driver *pdrive, int stage)
{
  peos->ConsToPrim(u0, w0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ViscousFluxes
//  \brief

TaskStatus Hydro::ViscousFluxes(Driver *pdrive, int stage)
{
  if (pvisc != nullptr) pvisc->AddViscousFlux(u0, uflx);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ApplyUnsplitSourceTerms
//  \brief adds source terms to hydro variables in EACH stage of stage run task list.
//  These are source terms that will be included as an unsplit algorithm.

TaskStatus Hydro::ApplyUnsplitSourceTerms(Driver *pdrive, int stage)
{
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ApplyOperatorSplitSourceTerms
//  \brief adds source terms to hydro variables in operator split task list.
//  These are source terms that will be included as an operator split algorithm.

TaskStatus Hydro::ApplyOperatorSplitSourceTerms(Driver *pdrive, int stage)
{
  return TaskStatus::complete;
}

} // namespace hydro
