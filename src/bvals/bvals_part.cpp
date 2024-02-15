//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_part.cpp
//! \brief

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::UpdateGID()
//! \brief

namespace particles {

KOKKOS_INLINE_FUNCTION
void UpdateGID(int &newgid, NeighborBlock nghbr, int myrank) {
  newgid = nghbr.gid;
/*
void UpdateGID(int &mp, NeighborBlock &nghbr, int myrank, DvceArray1D<int> &counter,
               DualArray1D<ParticleSendData> &prtcl_sendlist, int p) {
  mp = nghbr.gid;
  if (nghbr.rank != myrank) {
    int index = atomic_fetch_add(&counter(),1);
    prtcl_sendlist.d_view(index).prtcl_indx = p;
    prtcl_sendlist.d_view(index).prtcl_gid  = nghbr.gid;
    prtcl_sendlist.d_view(index).dest_rank  = nghbr.rank;
  }
*/
  // TODO (@jmstone)
  // Use dual array of tuples (rank_tosend, prtcl_indx, prtcl_gid) for sends with MPI
  // Use atomics to get index of dual array in which to store data?
  //  OR use Kokkos::vector ???
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::SetNewGID()
//! \brief

TaskStatus ParticlesBoundaryValues::SetNewGID() {
  // create local references for variables in kernel
  auto &ppos = pmy_part->prtcl_pos;
  auto &pgid = pmy_part->prtcl_gid;
  int npart = pmy_part->nprtcl_thispack;
  auto mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto meshsize = pmy_part->pmy_pack->pmesh->mesh_size;
  auto myrank = global_variable::my_rank;
  auto nghbr = pmy_part->pmy_pack->pmb->nghbr;
  DvceArray1D<int> counter("SendCounter",1);

  par_for("part_update",DevExeSpace(),0,npart, KOKKOS_LAMBDA(const int p) {
    int oldgid = pgid.d_view(p);  // prevent updated GID being used to index array
    Real x1 = ppos(p,IPX);
    Real x2 = ppos(p,IPY);
    Real x3 = ppos(p,IPZ);

    if (x1 < mbsize.d_view(oldgid).x1min) {
      if (x2 < mbsize.d_view(oldgid).x2min) {
        if (x3 < mbsize.d_view(oldgid).x3min) {
          // corner
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,48), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // corner
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,52), myrank);
        } else {
          // x1x2 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,16), myrank);
        }
      } else if (x2 > mbsize.d_view(oldgid).x2max) {
        if (x3 < mbsize.d_view(oldgid).x3min) {
          // corner
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,50), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // corner
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,54), myrank);
        } else {
          // x1x2 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,20), myrank);
        }
      } else {
        if (x3 < mbsize.d_view(oldgid).x3min) {
          // x3x1 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,32), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // x3x1 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,36), myrank);
        } else {
          // x1 face
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,0), myrank);
        }
      }

    } else if (x1 > mbsize.d_view(oldgid).x1max) {
      if (x2 < mbsize.d_view(oldgid).x2min) {
        if (x3 < mbsize.d_view(oldgid).x3min) {
          // corner
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,49), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // corner
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,53), myrank);
        } else {
          // x1x2 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,18), myrank);
        }
      } else if (x2 > mbsize.d_view(oldgid).x2max) {
        if (x3 < mbsize.d_view(oldgid).x3min) {
          // corner
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,51), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // corner
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,55), myrank);
        } else {
          // x1x2 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,22), myrank);
        }
      } else {
        if (x3 < mbsize.d_view(oldgid).x3min) {
          // x3x1 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,34), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // x3x1 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,38), myrank);
        } else {
          // x1 face
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,4), myrank);
        }
      }

    } else {
      if (x2 < mbsize.d_view(oldgid).x2min) {
        if (x3 < mbsize.d_view(oldgid).x3min) {
          // x2x3 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,40), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // x2x3 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,44), myrank);
        } else {
          // x2 face
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,8), myrank);
        }
      } else if (x2 > mbsize.d_view(oldgid).x2max) {
        if (x3 < mbsize.d_view(oldgid).x3min) {
          // x2x3 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,42), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // x2x3 edge
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,46), myrank);
        } else {
          // x2 face
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,12), myrank);
        }
      } else {
        if (x2 < mbsize.d_view(oldgid).x3min) {
          // x3 face
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,24), myrank);
        } else if (x3 > mbsize.d_view(oldgid).x3max) {
          // x3 face
          UpdateGID(pgid.d_view(p), nghbr.d_view(oldgid,28), myrank);
        }
      }
    }

    // reset x,y,z positions if particle crosses Mesh boundary using periodic BCs
    if (x1 < meshsize.x1min) {
      ppos(p,IPX) += (meshsize.x1max - meshsize.x1min);
    } else if (x1 > meshsize.x1max) {
      ppos(p,IPX) -= (meshsize.x1max - meshsize.x1min);
    }
    if (x2 < meshsize.x2min) {
      ppos(p,IPY) += (meshsize.x2max - meshsize.x2min);
    } else if (x2 > meshsize.x2max) {
      ppos(p,IPY) -= (meshsize.x2max - meshsize.x2min);
    }
    if (x3 < meshsize.x3min) {
      ppos(p,IPZ) += (meshsize.x3max - meshsize.x3min);
    } else if (x3 > meshsize.x3max) {
      ppos(p,IPZ) -= (meshsize.x3max - meshsize.x3min);
    }
/*
*/
  });

/***/
for (int p=0; p<npart; ++p) {
    Real x1 = ppos(p,IPX);
    Real x2 = ppos(p,IPY);
    Real x3 = ppos(p,IPZ);
    if ((x1 < meshsize.x1min) || (x1 > meshsize.x1max)) {
std::cout <<"p="<<p<<"  x="<<ppos(p,IPX) << std::endl;
    }
    if ((x2 < meshsize.x2min) || (x2 > meshsize.x2max)) {
std::cout <<"p="<<p<<"  y="<<ppos(p,IPY) << std::endl;
    }
    if ((x3 < meshsize.x3min) || (x3 > meshsize.x3max)) {
std::cout <<"p="<<p<<"  z="<<ppos(p,IPZ) << std::endl;
    }
}

#if MPI_PARALLEL_ENABLED
  // TODO (@jmstone)
  // Sync dual array of tuples to host
  //
  // Sort dual array on host by rank_sendto.
  //
  // Store nrank_sendto in vector of length nrank, stored in my_rank location
  //
  // Sync dual array to device
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::InitRecv()
//! \brief

/*

TaskStatus ParticlesBoundaryValues::InitRecv() {
#if MPI_PARALLEL_ENABLED

  // TODO (@jmstone)
  // GatherAllv nrank_sendto across all ranks

  MPI_Allgatherv(MPI_IN_PLACE, pmy_mesh->nmb_eachrank[global_variable::my_rank],
                 MPI_INT, refine_flag.h_view.data(), pmy_mesh->nmb_eachrank,
                 pmy_mesh->gids_eachrank, MPI_INT, MPI_COMM_WORLD);

  //
  // Create vector of tuples of (rank_tosend, nparticle_tosend) of length nrank_sendto
  // from Dual array
  //
  // Gatherallv npart_tosend across all ranks
  //
  // Figure out how many receives will occur on this rank, store nrank_recvfrom
  //
  // Allocate receive buffer
  //
  // Post non-blocking receives

#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::PackAndSendParticles()
//! \brief

TaskStatus ParticlesBoundaryValues::PackAndSendParticles() {
#if MPI_PARALLEL_ENABLED

  // TODO (@jmstone)
  // Allocate send buffer
  //
  // Use DualArray of tuples to load particles into send buffer
  //
  // Post sends

#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::RecvAndUnpackParticles()
//! \brief

TaskStatus ParticlesBoundaryValues::RecvAndUnpackParticles() {
#if MPI_PARALLEL_ENABLED

  // TODO (@jmstone)
  // if (npart_recv > npart_send)
  //    increase size of particle arrays
  // else if (npart_recv < npart_send)
  //    trim size of particle arrays
  // else do nothing
  //
  // blocking wait until all receives finish
  //
  // Use DualArray of tuples to load particles into particle array
  //
  // Update nparticles_thisrank.  Update cost array (use npart_thismb[nmb]?)
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearSend()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearSend() {
#if MPI_PARALLEL_ENABLED

  // TODO (@jmstone)
  // clear MPI send calls
  //
  // deallocate any vectors created (nranks_sendto, etc.)

#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearRecv() {
#if MPI_PARALLEL_ENABLED

  // TODO (@jmstone)
  // clear MPI receive calls
  //
  // deallocate any vectors created (nranks_recvfrom, etc.)

#endif
}

*/

} // end namaspace particles
