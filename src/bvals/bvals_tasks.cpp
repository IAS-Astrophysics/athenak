//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_tasks.cpp
//! \brief functions included in task lists to post/clear non-blocking MPI calls for
//! Mesh variables. These are generic functions that work for both CC and FC variables.
//!
//! Note: InitFluxRecv() functions for flux correction step are specific to CC/FC vars,
//! and are implemented in flux_correct_XX.cpp files respectively. The ClearFluxRecv()
//! and ClearFluxSend() functions are generic and implemented below.
//!
//! Note2: task list functions for particle communication are all implemented in
//! bvals_part.cpp file.

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications of vars.

TaskStatus MeshBoundaryValues::InitRecv(const int nvars) {
#if MPI_PARALLEL_ENABLED
  if (rank_packed_bvals_nvars_ != nvars ||
      rank_packed_mesh_seq_ != pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq()) {
    BuildRankPackedVarMetadata(nvars);
  } else {
    std::fill(recv_var_reqs_.begin(), recv_var_reqs_.end(), MPI_REQUEST_NULL);
    std::fill(send_var_reqs_.begin(), send_var_reqs_.end(), MPI_REQUEST_NULL);
  }

  // Payload-only Irecv: the (lid,dn,data_size) layout of each peer's payload
  // is already known from the one-shot header exchange done during
  // BuildRankPackedVarMetadata, so no per-step header receive is posted.
  bool no_errors = true;
  for (std::size_t i = 0; i < recv_var_msgs_.size(); ++i) {
    auto &msg = recv_var_msgs_[i];
    int ierr = MPI_Irecv(rank_recvbuf_vars_.data() + msg.offset, msg.data_size,
                     MPI_ATHENA_REAL, msg.rank, 1, comm_vars, &recv_var_reqs_[i]);
    if (ierr != MPI_SUCCESS) no_errors = false;
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting rank-packed receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::ClearRecv
//! \brief Waits for all MPI receives associated with communcation of boundary variables
//! to complete before allowing execution to continue

TaskStatus MeshBoundaryValues::ClearRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors = true;
  for (std::size_t i = 0; i < recv_var_reqs_.size(); ++i) {
    int ierr = MPI_Wait(&recv_var_reqs_[i], MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) no_errors = false;
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in clearing rank-packed receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::ClearSend
//! \brief Waits for all MPI sends associated with communcation of boundary variables
//! to complete before allowing execution to continue

TaskStatus MeshBoundaryValues::ClearSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors = true;
  for (std::size_t i = 0; i < send_var_reqs_.size(); ++i) {
    int ierr = MPI_Wait(&send_var_reqs_[i], MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) no_errors = false;
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in clearing rank-packed sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::ClearFluxRecv
//! \brief Waits for all MPI receives associated with communcation of boundary fluxes
//! to complete before allowing execution to continue

TaskStatus MeshBoundaryValues::ClearFluxRecv() {
  bool no_errors=true;
#if MPI_PARALLEL_ENABLED
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for fluxes to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) &&
           (recvbuf[n].flux_req[m] != MPI_REQUEST_NULL) ) {
        int ierr = MPI_Wait(&(recvbuf[n].flux_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
#endif
  if (no_errors) return TaskStatus::complete;

  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::ClearFluxSend
//! \brief Waits for all MPI sends associated with communcation of boundary fluxes to
//!  complete before allowing execution to continue

TaskStatus MeshBoundaryValues::ClearFluxSend() {
  bool no_errors=true;
#if MPI_PARALLEL_ENABLED
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for fluxes to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) &&
           (sendbuf[n].flux_req[m] != MPI_REQUEST_NULL) ) {
        int ierr = MPI_Wait(&(sendbuf[n].flux_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
#endif
  if (no_errors) return TaskStatus::complete;

  return TaskStatus::fail;
}
