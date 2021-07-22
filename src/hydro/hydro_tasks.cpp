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
#include "utils/create_mpitag.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleHydroTasks
//  \brief Adds hydro tasks to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysicsModules() function directly after Hydro constrctr
//
//  Stage start tasks are those that must be cmpleted over all MeshBlocks before EACH
//  stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//
//  Stage run tasks are those performed in EACH stage
//
//  Stage end tasks are those that can only be cmpleted after all the stage run tasks are
//  finished over all MeshBlocks for EACH stage, such as clearing all MPI non-blocking
//  sends, etc.

void Hydro::AssembleHydroTasks(TaskList &start, TaskList &run, TaskList &end)
{
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
  id.expl  = run.AddTask(&Hydro::ExpRKUpdate, this, id.flux);
  id.sendu = run.AddTask(&Hydro::SendU, this, id.expl);
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

} // namespace hydro
