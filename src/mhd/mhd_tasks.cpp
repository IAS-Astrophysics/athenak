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
#include "utils/create_mpitag.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::AssembleMHDTasks
//  \brief Adds mhd tasks to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysicsModules() function directly after MHD constrctr
  
void MHD::AssembleMHDTasks(TaskList &start, TaskList &run, TaskList &end)
{ 
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
  id.expl = run.AddTask(&MHD::ExpRKUpdate, this, id.flux);
  id.sendu = run.AddTask(&MHD::SendU, this, id.expl);
  id.recvu = run.AddTask(&MHD::RecvU, this, id.sendu);
  id.efld = run.AddTask(&MHD::CornerE, this, id.recvu);
  id.ct = run.AddTask(&MHD::CT, this, id.efld);
  id.sendb = run.AddTask(&MHD::SendB, this, id.ct);
  id.recvb = run.AddTask(&MHD::RecvB, this, id.sendb);
  id.bcs = run.AddTask(&MHD::ApplyPhysicalBCs, this, id.recvb);
  id.c2p = run.AddTask(&MHD::ConToPrim, this, id.bcs);
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

TaskStatus MHD::InitRecv(Driver *pdrive, int stage)
{
  int &nmb = pmy_pack->pmb->nmb;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications for both cell-centered conserved variables and 
  // face-centered magnetic fields
  auto &rbufu = pbval_u->recv_buf;
  auto &rbufb = pbval_b->recv_buf;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          {
          // Receive requests for U
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::FluidCons_ID);
          auto recv_data = Kokkos::subview(rbufu[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr.h_view(m,n).rank, tag, MPI_COMM_WORLD, &(rbufu[n].comm_req[m]));
          }

          {
          // Receive requests for B
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::BField_ID);
          auto recv_data = Kokkos::subview(rbufb[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr.h_view(m,n).rank, tag, MPI_COMM_WORLD, &(rbufb[n].comm_req[m]));
          }
        }
#endif
        // initialize boundary receive status flag
        rbufu[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
        rbufb[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue
//  With MHD, clears both receives of U and B

TaskStatus MHD::ClearRecv(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for U and B to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
          MPI_Wait(&(pbval_b->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void MHD::ClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue
//  With MHD, clears both sends of U and B

TaskStatus MHD::ClearSend(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for U and B to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
          MPI_Wait(&(pbval_b->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void MHD::CopyCons
//  \brief  copy u0 --> u1, and b0 --> b1 in first stage

TaskStatus MHD::CopyCons(Driver *pdrive, int stage)
{
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

TaskStatus MHD::SendU(Driver *pdrive, int stage) 
{
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0, VariablesID::FluidCons_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RecvU
//  \brief receives cell-centered conserved variables

TaskStatus MHD::RecvU(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::SendB
//  \brief sends face-centered magnetic fields

TaskStatus MHD::SendB(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_b->PackAndSendFC(b0, VariablesID::BField_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::RecvB
//  \brief receives face-centered magnetic fields

TaskStatus MHD::RecvB(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_b->RecvAndUnpackFC(b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ConToPrim
//  \brief

TaskStatus MHD::ConToPrim(Driver *pdrive, int stage)
{
  peos->ConsToPrim(u0, b0, w0, bcc0);
  return TaskStatus::complete;
}

} // namespace mhd
