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
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications of variables
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        // rank of destination buffer
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          // calculate amount of data to be passed, get pointer to variables
          int data_size = nvars;
          if ( nghbr.h_view(m,n).lev < pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= recvbuf[n].icoar_ndat;
          } else if ( nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m) ) {
            if (is_z4c_) {
              data_size *= recvbuf[n].isame_z4c_ndat;
            } else {
              data_size *= recvbuf[n].isame_ndat;
            }
          } else {
            data_size *= recvbuf[n].ifine_ndat;
          }
          auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, Kokkos::ALL);

          // Post non-blocking receive for this buffer on this MeshBlock
          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(recvbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
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
  bool no_errors=true;
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for vars to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) ) {
        int ierr = MPI_Wait(&(recvbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
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
  bool no_errors=true;
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for vars to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) ) {
        int ierr = MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
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
