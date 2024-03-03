//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.cpp
//! \brief functions to pack/send and recv/unpack/prolongate boundary values for
//! cell-centered variables, implemented as part of the BValCC class.

#include <cstdlib>
#include <iostream>
#include <utility>
#include <algorithm> // max

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// BoundaryValues constructor:

BoundaryValues::BoundaryValues(MeshBlockPack *pp, ParameterInput *pin, bool z4c) :
    pmy_pack(pp),
    is_z4c_(z4c),
    u_in("uin",1,1),
    b_in("bin",1,1),
    i_in("iin",1,1) {
  int nnghbr = pmy_pack->pmb->nnghbr;

  // sendbuf and recvbuf are fixed-length [56-element] arrays
  // Initialize some of the data in appropriate elements based on dimensionality of
  // problem (indicated by value of nnghbr)
  for (int n=0; n<nnghbr; ++n) {
#if MPI_PARALLEL_ENABLED
    // allocate vector of MPI requests (if needed)
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    sendbuf[n].vars_req = new MPI_Request[nmb];
    sendbuf[n].flux_req = new MPI_Request[nmb];
    recvbuf[n].vars_req = new MPI_Request[nmb];
    recvbuf[n].flux_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      sendbuf[n].flux_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].flux_req[m] = MPI_REQUEST_NULL;
    }
#endif
    // initialize data sizes in each send/recv buffer to zero
    sendbuf[n].isame_ndat = 0;
    sendbuf[n].isame_z4c_ndat = 0;
    sendbuf[n].icoar_ndat = 0;
    sendbuf[n].ifine_ndat = 0;
    sendbuf[n].iflxs_ndat = 0;
    sendbuf[n].iflxc_ndat = 0;
    recvbuf[n].isame_ndat = 0;
    recvbuf[n].isame_z4c_ndat = 0;
    recvbuf[n].icoar_ndat = 0;
    recvbuf[n].ifine_ndat = 0;
    recvbuf[n].iflxs_ndat = 0;
    recvbuf[n].iflxc_ndat = 0;
  }

#if MPI_PARALLEL_ENABLED
  // create unique communicators for variables and fluxes in this BoundaryValues object
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_vars);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_flux);
#endif
}

//----------------------------------------------------------------------------------------
// destructor

BoundaryValues::~BoundaryValues() {
#if MPI_PARALLEL_ENABLED
  int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<nnghbr; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] sendbuf[n].flux_req;
    delete [] recvbuf[n].vars_req;
    delete [] recvbuf[n].flux_req;
  }
  for (int n=0; n<2; ++n) {
    delete [] sendbuf_orb[n].vars_req;
    delete [] recvbuf_orb[n].vars_req;
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::InitializeBuffers
//! \brief initialize each element of send/recv BoundaryBuffers fixed-length arrays
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector
//! NOTE2: work here cannot be done in BoundaryValues constructor since it calls pure
//! virtual functions that only get instantiated when the derived classes are constructed

void BoundaryValues::InitializeBuffers(const int nvar) {
  // allocate memory for inflow BCs (but only if domain not strictly periodic)
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    Kokkos::realloc(u_in, nvar, 6);
    Kokkos::realloc(b_in, 3, 6);   // always 3 components of face-fields
    Kokkos::realloc(i_in, nvar, 6);
  }

  // set number of subblocks in x2- and x3-dirs
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  // initialize buffers used for uniform grid and SMR/AMR calculations

  // x1 faces; NeighborIndex = [0,...,7]
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy = 0; fy<nfy; fy++) {
        int indx = pmy_pack->pmb->NeighborIndx(n,0,0,fy,fz);
        InitSendIndices(sendbuf[indx],n, 0, 0, fy, fz);
        InitRecvIndices(recvbuf[indx],n, 0, 0, fy, fz);
        sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        indx++;
      }
    }
  }

  // add more buffers in 2D
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces; NeighborIndex = [8,...,15]
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,m,0,fx,fz);
          InitSendIndices(sendbuf[indx],0, m, 0, fx, fz);
          InitRecvIndices(recvbuf[indx],0, m, 0, fx, fz);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x1x2 edges; NeighborIndex = [16,...,23]
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int indx = pmy_pack->pmb->NeighborIndx(n,m,0,fz,0);
          InitSendIndices(sendbuf[indx],n, m, 0, fz, 0);
          InitRecvIndices(recvbuf[indx],n, m, 0, fz, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }
  }

  // add more buffers in 3D
  if (pmy_pack->pmesh->three_d) {
    // x3 faces; NeighborIndex = [24,...,31]
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,0,l,fx,fy);
          InitSendIndices(sendbuf[indx],0, 0, l, fx, fy);
          InitRecvIndices(recvbuf[indx],0, 0, l, fx, fy);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x3x1 edges; NeighborIndex = [32,...,39]
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int indx = pmy_pack->pmb->NeighborIndx(n,0,l,fy,0);
          InitSendIndices(sendbuf[indx],n, 0, l, fy, 0);
          InitRecvIndices(recvbuf[indx],n, 0, l, fy, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x2x3 edges; NeighborIndex = [40,...,47]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,m,l,fx,0);
          InitSendIndices(sendbuf[indx],0, m, l, fx, 0);
          InitRecvIndices(recvbuf[indx],0, m, l, fx, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // corners; NeighborIndex = [48,...,55]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int indx = pmy_pack->pmb->NeighborIndx(n,m,l,0,0);
          InitSendIndices(sendbuf[indx],n, m, l, 0, 0);
          InitRecvIndices(recvbuf[indx],n, m, l, 0, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValues::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications of vars.

TaskStatus BoundaryValues::InitRecv(const int nvars) {
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
//! \fn  void BoundaryValues::ClearRecv
//! \brief Waits for all MPI receives associated with communcation of boundary variables
//! to complete before allowing execution to continue

TaskStatus BoundaryValues::ClearRecv() {
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
//! \fn  void BoundaryValues::ClearSend
//! \brief Waits for all MPI sends associated with communcation of boundary variables
//! to complete before allowing execution to continue

TaskStatus BoundaryValues::ClearSend() {
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
//! \fn  void BoundaryValues::ClearFluxRecv
//! \brief Waits for all MPI receives associated with communcation of boundary fluxes
//! to complete before allowing execution to continue

TaskStatus BoundaryValues::ClearFluxRecv() {
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
//! \fn  void BoundaryValues::ClearFluxSend
//! \brief Waits for all MPI sends associated with communcation of boundary fluxes to
//!  complete before allowing execution to continue

TaskStatus BoundaryValues::ClearFluxSend() {
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
