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
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::UpdateGID()
//! \brief Updates GID of particles that cross boundary of their parent MeshBlock.  If
//! the new GID is on a different rank, then store (1) index of particle in prtcl array,
//! (2) destination GID, and (3) destination rank in prtcl_sendlist DvceArray.

namespace particles {

KOKKOS_INLINE_FUNCTION
void UpdateGID(int &newgid, NeighborBlock nghbr, int myrank, int *pcounter,
               DvceArray1D<ParticleSendData> prtcl_sendlist, int p) {
  newgid = nghbr.gid;
#if MPI_PARALLEL_ENABLED
  if (nghbr.rank != myrank) {
    int index = Kokkos::atomic_fetch_add(pcounter,1);
    prtcl_sendlist(index).prtcl_indx = p;
    prtcl_sendlist(index).dest_gid   = nghbr.gid;
    prtcl_sendlist(index).dest_rank  = nghbr.rank;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::SetNewGID()
//! \brief

TaskStatus ParticlesBoundaryValues::SetNewPrtclGID() {
  // create local references for variables in kernel
  auto gids = pmy_part->pmy_pack->gids;
  auto &ppos = pmy_part->prtcl_pos;
  auto &pgid = pmy_part->prtcl_gid;
  int npart = pmy_part->nprtcl_thispack;
  auto mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto meshsize = pmy_part->pmy_pack->pmesh->mesh_size;
  auto myrank = global_variable::my_rank;
  auto nghbr = pmy_part->pmy_pack->pmb->nghbr;
  auto &psendl = prtcl_sendlist;
  int counter=0;
  int *pcounter = &counter;

  par_for("part_update",DevExeSpace(),0,npart, KOKKOS_LAMBDA(const int p) {
    int m = pgid(p) - gids;
    Real x1 = ppos(p,IPX);
    Real x2 = ppos(p,IPY);
    Real x3 = ppos(p,IPZ);

    if (x1 < mbsize.d_view(m).x1min) {
      if (x2 < mbsize.d_view(m).x2min) {
        if (x3 < mbsize.d_view(m).x3min) {
          // corner
          UpdateGID(pgid(p), nghbr.d_view(m,48), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // corner
          UpdateGID(pgid(p), nghbr.d_view(m,52), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          UpdateGID(pgid(p), nghbr.d_view(m,16), myrank, pcounter, psendl, p);
        }
      } else if (x2 > mbsize.d_view(m).x2max) {
        if (x3 < mbsize.d_view(m).x3min) {
          // corner
          UpdateGID(pgid(p), nghbr.d_view(m,50), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // corner
          UpdateGID(pgid(p), nghbr.d_view(m,54), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          UpdateGID(pgid(p), nghbr.d_view(m,20), myrank, pcounter, psendl, p);
        }
      } else {
        if (x3 < mbsize.d_view(m).x3min) {
          // x3x1 edge
          UpdateGID(pgid(p), nghbr.d_view(m,32), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x3x1 edge
          UpdateGID(pgid(p), nghbr.d_view(m,36), myrank, pcounter, psendl, p);
        } else {
          // x1 face
          UpdateGID(pgid(p), nghbr.d_view(m,0), myrank, pcounter, psendl, p);
        }
      }

    } else if (x1 > mbsize.d_view(m).x1max) {
      if (x2 < mbsize.d_view(m).x2min) {
        if (x3 < mbsize.d_view(m).x3min) {
          // corner
          UpdateGID(pgid(p), nghbr.d_view(m,49), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // corner
          UpdateGID(pgid(p), nghbr.d_view(m,53), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          UpdateGID(pgid(p), nghbr.d_view(m,18), myrank, pcounter, psendl, p);
        }
      } else if (x2 > mbsize.d_view(m).x2max) {
        if (x3 < mbsize.d_view(m).x3min) {
          // corner
          UpdateGID(pgid(p), nghbr.d_view(m,51), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // corner
          UpdateGID(pgid(p), nghbr.d_view(m,55), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          UpdateGID(pgid(p), nghbr.d_view(m,22), myrank, pcounter, psendl, p);
        }
      } else {
        if (x3 < mbsize.d_view(m).x3min) {
          // x3x1 edge
          UpdateGID(pgid(p), nghbr.d_view(m,34), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x3x1 edge
          UpdateGID(pgid(p), nghbr.d_view(m,38), myrank, pcounter, psendl, p);
        } else {
          // x1 face
          UpdateGID(pgid(p), nghbr.d_view(m,4), myrank, pcounter, psendl, p);
        }
      }

    } else {
      if (x2 < mbsize.d_view(m).x2min) {
        if (x3 < mbsize.d_view(m).x3min) {
          // x2x3 edge
          UpdateGID(pgid(p), nghbr.d_view(m,40), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x2x3 edge
          UpdateGID(pgid(p), nghbr.d_view(m,44), myrank, pcounter, psendl, p);
        } else {
          // x2 face
          UpdateGID(pgid(p), nghbr.d_view(m,8), myrank, pcounter, psendl, p);
        }
      } else if (x2 > mbsize.d_view(m).x2max) {
        if (x3 < mbsize.d_view(m).x3min) {
          // x2x3 edge
          UpdateGID(pgid(p), nghbr.d_view(m,42), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x2x3 edge
          UpdateGID(pgid(p), nghbr.d_view(m,46), myrank, pcounter, psendl, p);
        } else {
          // x2 face
          UpdateGID(pgid(p), nghbr.d_view(m,12), myrank, pcounter, psendl, p);
        }
      } else {
        if (x2 < mbsize.d_view(m).x2min) {
          // x3 face
          UpdateGID(pgid(p), nghbr.d_view(m,24), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x3 face
          UpdateGID(pgid(p), nghbr.d_view(m,28), myrank, pcounter, psendl, p);
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
  });
  nprtcl_send = counter;

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::CountSendsAndRecvs()
//! \brief

TaskStatus ParticlesBoundaryValues::CountSendsAndRecvs() {
#if MPI_PARALLEL_ENABLED
  // Get copy of send list on host
  auto sendlist = Kokkos::subview(prtcl_sendlist, std::make_pair(0,nprtcl_send));
  auto sendlist_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sendlist);

  // Sort send list on host by new_rank.
  namespace KE = Kokkos::Experimental;
  std::sort(KE::begin(sendlist_h), KE::end(sendlist_h), SortByRank);

/***/
for (int n=0; n<nprtcl_send; ++n) {
std::cout << "rank="<<global_variable::my_rank<<"  (n,indx,rank,gid)=" << n<<"  "<<sendlist_h(n).prtcl_indx<<"  "<<sendlist_h(n).dest_rank<<"  "<<sendlist_h(n).dest_gid << std::endl;
}
/****/

  // load STL::vector with <sendrank, recvrank, nprtcl_tosend> tuples for particles sends
  // from this rank. Length will be nsends; initially this length is unknown
  sends_thisrank.clear();
  if (nprtcl_send > 0) {
    int &myrank = global_variable::my_rank;
    int rank = sendlist_h(0).dest_rank;
    int nprtcl = 1;

    for (int n=1; n<nprtcl_send; ++n) {
      if (sendlist_h(n).dest_rank == rank) {
        ++nprtcl;
      } else {
        sends_thisrank.emplace_back(std::make_tuple(myrank,rank,nprtcl));
        rank = sendlist_h(n).dest_rank;
        nprtcl = 1;
      }
    }
    sends_thisrank.emplace_back(std::make_tuple(myrank,rank,nprtcl));
  }
  nsends = sends_thisrank.size();

/***/
{
int ierr = MPI_Barrier(MPI_COMM_WORLD);
for (int n=0; n<nsends; ++n) {
std::cout << "n="<<n<< "  (sendrank,destrank,npart)=" << std::get<0>(sends_thisrank[n])<<"  "<<std::get<1>(sends_thisrank[n]) << "  "<<std::get<2>(sends_thisrank[n]) << std::endl;
}
}
/****/

  // Share number of ranks to send to amongst all ranks
  nsends_eachrank[global_variable::my_rank] = nsends;
  MPI_Allgather(&nsends, 1, MPI_INT, nsends_eachrank.data(), 1, MPI_INT, mpi_comm_part);

/***/
{
int ierr = MPI_Barrier(MPI_COMM_WORLD);
if (global_variable::my_rank == 0) {
for (int n=0; n<global_variable::nranks; ++n) {
std::cout << "n="<<n<<"  sends_eachrank="<<nsends_eachrank[n]<< std::endl;
}
}
}
/****/

  // Now share <sendrank, recvrank, nprtcl_tosend> tuples amongst all ranks
  // First create vector of starting indices of tuples in full vector
  std::vector<int> nsends_displ;
  nsends_displ.resize(global_variable::nranks);
  nsends_displ[0] = 0;
  for (int n=1; n<(global_variable::nranks); ++n) {
    nsends_displ[n] = nsends_displ[n-1] + nsends_eachrank[n-1];
  }
  int nsends_allranks = nsends_displ[global_variable::nranks - 1] +
                        nsends_eachrank[global_variable::nranks - 1];
/***/
{
int ierr = MPI_Barrier(MPI_COMM_WORLD);
if (global_variable::my_rank == 0) {
for (int n=0; n<global_variable::nranks; ++n) {
std::cout << "n="<<n<<"  nsends_displ="<<nsends_displ[n]<< std::endl;
}
std::cout << "nsends_allranks = " << nsends_allranks << std::endl;
}
}
/****/

  // Load tuples on this rank into full vector
  sends_allranks.resize(nsends_allranks);
  for (int n=0; n<nsends_eachrank[global_variable::my_rank]; ++n) {
    sends_allranks[n + nsends_displ[global_variable::my_rank]] = sends_thisrank[n];
  }

  // Share tuples using MPI derived data type for tuple of 3*int
  MPI_Datatype mpi_ituple;
  MPI_Type_contiguous(3, MPI_INT, &mpi_ituple);
  MPI_Allgatherv(MPI_IN_PLACE, nsends_eachrank[global_variable::my_rank],
                   mpi_ituple, sends_allranks.data(), nsends_eachrank.data(),
                   nsends_displ.data(), mpi_ituple, mpi_comm_part);
/***/
{
int ierr = MPI_Barrier(MPI_COMM_WORLD);
if (global_variable::my_rank == 0) {
for (int n=0; n<nsends_allranks; ++n) {
std::cout << "n="<<n<< "  (sendrank,destrank,npart)=" << std::get<0>(sends_allranks[n])<<"  "<<std::get<1>(sends_allranks[n]) << "  "<<std::get<2>(sends_allranks[n]) << std::endl;
}
}
}
/****/

#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::InitPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::InitPrtclRecv() {
#if MPI_PARALLEL_ENABLED

  // load STL::vector with <sendrank,recvrank,nprtcl_recv> tuples for particle receives
  // on this rank. Length will be nrecvs, initially this length is unknown
  recvs_thisrank.clear();

  int nsends_allranks = sends_allranks.size();
  for (int n=0; n<nsends_allranks; ++n) {
    if (std::get<1>(sends_allranks[n]) == global_variable::my_rank) {
      recvs_thisrank.emplace_back(sends_allranks[n]);
    }
  }
  nrecvs = recvs_thisrank.size();

  // Figure out how many particles will be received from all ranks
  nprtcl_recv=0;
  for (int n=0; n<nrecvs; ++n) {
    nprtcl_recv += std::get<2>(recvs_thisrank[n]);
  }

  // Allocate receive buffer
  Kokkos::realloc(prtcl_recvbuf, nprtcl_recv);

  // Post non-blocking receives
  bool no_errors=true;
  int data_start=0;
  recv_req.clear();
  for (int n=0; n<nrecvs; ++n) { recv_req.emplace_back(MPI_REQUEST_NULL); }

  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = std::get<2>(recvs_thisrank[n])*sizeof(ParticleData);
    int data_end = data_start + std::get<2>(recvs_thisrank[n]);
    auto recv_ptr = Kokkos::subview(prtcl_recvbuf, std::make_pair(data_start, data_end));
    int drank = std::get<0>(recvs_thisrank[n]);

    // Post non-blocking receive
    int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_BYTE, drank, MPI_ANY_TAG,
                         mpi_comm_part, &(recv_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
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

/*
//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::PackAndSendParticles()
//! \brief

TaskStatus ParticlesBoundaryValues::PackAndSendParticles() {
#if MPI_PARALLEL_ENABLED

  // Figure out how many particles will be sent total
  // Allocate send buffer
  //
  // Use DualArray of tuples to load particles into send buffer ordered by dest_rank
  //
  // Post sends using address of send buffer with appropriate offset and nelements

#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::RecvAndUnpackParticles()
//! \brief

TaskStatus ParticlesBoundaryValues::RecvAndUnpackParticles() {
#if MPI_PARALLEL_ENABLED

  // TODO (@jmstone)
  // if (npart_recv > nprtcl_send)
  //    increase size of particle arrays
  // else if (npart_recv < nprtcl_send)
  //    trim size of particle arrays
  // else do nothing
  //
  // blocking wait until all receives finish
  //
  // Sort DualArray by particle index
  // Use DualArray of tuples to load particles into particle array at blank index spots
  //
  // Update nparticles_thisrank.  Update cost array (use npart_thismb[nmb]?)
#endif
  return TaskStatus::complete;
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
  return TaskStatus::complete;
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
  return TaskStatus::complete;
}

*/

} // end namaspace particles
