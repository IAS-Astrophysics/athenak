//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.cpp
//! \brief constructors and initializers for both particle and Mesh variable boundary
//! classes.

#include <cstdlib>
#include <iostream>
#include <utility>
#include <algorithm> // max

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// MeshBoundaryValues constructor:

MeshBoundaryValues::MeshBoundaryValues(MeshBlockPack *pp, ParameterInput *pin, bool z4c) :
  pmy_pack(pp),
  is_z4c_(z4c),
  u_in("uin",1,1),
  b_in("bin",1,1),
  i_in("iin",1,1) {
  // allocate vector of status flags and MPI requests (if needed)
  int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<nnghbr; ++n) {
#if MPI_PARALLEL_ENABLED
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    send_buf[n].vars_req = new MPI_Request[nmb];
    send_buf[n].flux_req = new MPI_Request[nmb];
    recv_buf[n].vars_req = new MPI_Request[nmb];
    recv_buf[n].flux_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      send_buf[n].vars_req[m] = MPI_REQUEST_NULL;
      send_buf[n].flux_req[m] = MPI_REQUEST_NULL;
      recv_buf[n].vars_req[m] = MPI_REQUEST_NULL;
      recv_buf[n].flux_req[m] = MPI_REQUEST_NULL;
    }
#endif
    // initialize data sizes in each send/recv buffer to zero
    send_buf[n].isame_ndat = 0;
    send_buf[n].isame_z4c_ndat = 0;
    send_buf[n].icoar_ndat = 0;
    send_buf[n].ifine_ndat = 0;
    send_buf[n].iflxs_ndat = 0;
    send_buf[n].iflxc_ndat = 0;
    recv_buf[n].isame_ndat = 0;
    recv_buf[n].isame_z4c_ndat = 0;
    recv_buf[n].icoar_ndat = 0;
    recv_buf[n].ifine_ndat = 0;
    recv_buf[n].iflxs_ndat = 0;
    recv_buf[n].iflxc_ndat = 0;
  }

#if MPI_PARALLEL_ENABLED
  // create unique communicators for variables and fluxes in this BoundaryValues object
  MPI_Comm_dup(MPI_COMM_WORLD, &vars_comm);
  MPI_Comm_dup(MPI_COMM_WORLD, &flux_comm);
#endif
}

//----------------------------------------------------------------------------------------
// MeshBoundaryValues destructor

MeshBoundaryValues::~MeshBoundaryValues() {
#if MPI_PARALLEL_ENABLED
  int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<nnghbr; ++n) {
    delete [] send_buf[n].vars_req;
    delete [] send_buf[n].flux_req;
    delete [] recv_buf[n].vars_req;
    delete [] recv_buf[n].flux_req;
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::InitializeBuffers
//! \brief initialize each element of send/recv MeshBoundaryBuffers fixed-length arrays
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector
//! NOTE2: work here cannot be done in MeshBoundaryValues constructor since it calls pure
//! virtual functions that only get instantiated when the derived classes are constructed

void MeshBoundaryValues::InitializeBuffers(const int nvar) {
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
        InitSendIndices(send_buf[indx],n, 0, 0, fy, fz);
        InitRecvIndices(recv_buf[indx],n, 0, 0, fy, fz);
        send_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        recv_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
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
          InitSendIndices(send_buf[indx],0, m, 0, fx, fz);
          InitRecvIndices(recv_buf[indx],0, m, 0, fx, fz);
          send_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recv_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x1x2 edges; NeighborIndex = [16,...,23]
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int indx = pmy_pack->pmb->NeighborIndx(n,m,0,fz,0);
          InitSendIndices(send_buf[indx],n, m, 0, fz, 0);
          InitRecvIndices(recv_buf[indx],n, m, 0, fz, 0);
          send_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recv_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
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
          InitSendIndices(send_buf[indx],0, 0, l, fx, fy);
          InitRecvIndices(recv_buf[indx],0, 0, l, fx, fy);
          send_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recv_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x3x1 edges; NeighborIndex = [32,...,39]
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int indx = pmy_pack->pmb->NeighborIndx(n,0,l,fy,0);
          InitSendIndices(send_buf[indx],n, 0, l, fy, 0);
          InitRecvIndices(recv_buf[indx],n, 0, l, fy, 0);
          send_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recv_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x2x3 edges; NeighborIndex = [40,...,47]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,m,l,fx,0);
          InitSendIndices(send_buf[indx],0, m, l, fx, 0);
          InitRecvIndices(recv_buf[indx],0, m, l, fx, 0);
          send_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recv_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // corners; NeighborIndex = [48,...,55]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int indx = pmy_pack->pmb->NeighborIndx(n,m,l,0,0);
          InitSendIndices(send_buf[indx],n, m, l, 0, 0);
          InitRecvIndices(recv_buf[indx],n, m, l, 0, 0);
          send_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recv_buf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
// ParticlesBoundaryValues constructor:

particles::ParticlesBoundaryValues::ParticlesBoundaryValues(
  particles::Particles *pp, ParameterInput *pin) :
    sendlist("sendlist",1),
#if MPI_PARALLEL_ENABLED
    prtcl_rsendbuf("rsend",1),
    prtcl_rrecvbuf("rrecv",1),
    prtcl_isendbuf("isend",1),
    prtcl_irecvbuf("irecv",1),
#endif
    pmy_part(pp) {
#if MPI_PARALLEL_ENABLED
  // Guess that no more than 10% of particles will be communicated to set size of buffer
  int npart = pmy_part->nprtcl_thispack;

  //resize vectors over number of ranks
  nsends_eachrank.resize(global_variable::nranks);

  // create unique communicator for particles
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_part);
#endif
}

//----------------------------------------------------------------------------------------
// destructor

particles::ParticlesBoundaryValues::~ParticlesBoundaryValues() {
}
