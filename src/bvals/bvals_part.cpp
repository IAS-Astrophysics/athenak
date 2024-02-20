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
//! \brief

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
  npart_send = counter;

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::SendCounts()
//! \brief

TaskStatus ParticlesBoundaryValues::SendPrtclCounts() {
#if MPI_PARALLEL_ENABLED
  // Get copy of send list on host
  auto sendlist = Kokkos::subview(prtcl_sendlist, std::make_pair(0,npart_send));
  auto sendlist_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sendlist);

  // Sort send list on host by new_rank.
  namespace KE = Kokkos::Experimental;
  std::sort(KE::begin(sendlist_h), KE::end(sendlist_h), SortByRank);

/***/
for (int n=0; n<npart_send; ++n) {
std::cout << "rank="<<global_variable::my_rank<<"  (n,indx,rank,gid)=" << n<<"  "<<sendlist_h(n).prtcl_indx<<"  "<<sendlist_h(n).dest_rank<<"  "<<sendlist_h(n).dest_gid << std::endl;
}
/****/

  // load STL::vector with <sendrank, recvrank,nprtcl_tosend> tuples.
  // Length is nranks_tosendto.  Use vector since initially length is unknown
  counts_thisrank.clear();
  if (npart_send > 0) {
    int &myrank = global_variable::my_rank;
    int rank = sendlist_h(0).dest_rank;
    int nprtcl = 1;

    for (int n=1; n<npart_send; ++n) {
      if (sendlist_h(n).dest_rank == rank) {
        ++nprtcl;
      } else {
        counts_thisrank.emplace_back(std::make_tuple(myrank,rank,nprtcl));
        rank = sendlist_h(n).dest_rank;
        nprtcl = 1;
      }
    }
    counts_thisrank.emplace_back(std::make_tuple(myrank,rank,nprtcl));
  }
  ncounts = counts_thisrank.size();

/***/
{
int ierr = MPI_Barrier(MPI_COMM_WORLD);
for (int n=0; n<ncounts; ++n) {
std::cout << "n="<<n<< "  (sendrank,destrank,npart)=" << std::get<0>(counts_thisrank[n])<<"  "<<std::get<1>(counts_thisrank[n]) << "  "<<std::get<2>(counts_thisrank[n]) << std::endl;
}
}
/****/

  // Share number of ranks to send to with all ranks
  ncounts_eachrank[global_variable::my_rank] = ncounts;
  MPI_Allgather(&ncounts, 1, MPI_INT, ncounts_eachrank.data(), 1, MPI_INT, mpi_comm_part);

/***/
{
int ierr = MPI_Barrier(MPI_COMM_WORLD);
if (global_variable::my_rank == 0) {
for (int n=0; n<global_variable::nranks; ++n) {
std::cout << "n="<<n<<"  counts_eachrank="<<ncounts_eachrank[n]<< std::endl;
}
}
}
/****/


  // Create vector of starting indices of <dest_rank,nprtcl_tosend> pairs over all ranks
  std::vector<int> ncounts_displ;
  ncounts_displ.resize(global_variable::nranks);
  ncounts_displ[0] = 0;
  for (int n=1; n<(global_variable::nranks); ++n) {
    ncounts_displ[n] = ncounts_displ[n-1] + ncounts_eachrank[n-1];
  }
  int ncounts_allranks = ncounts_displ[global_variable::nranks - 1] +
                         ncounts_eachrank[global_variable::nranks - 1];
  counts_allranks.resize(ncounts_allranks);
/***/
{
int ierr = MPI_Barrier(MPI_COMM_WORLD);
if (global_variable::my_rank == 0) {
for (int n=0; n<global_variable::nranks; ++n) {
std::cout << "n="<<n<<"  ncounts_displ="<<ncounts_displ[n]<< std::endl;
}
std::cout << "ncounts_allranks = " << ncounts_allranks << std::endl;
}
}
/****/

  for (int n=0; n<ncounts_eachrank[global_variable::my_rank]; ++n) {
    counts_allranks[n + ncounts_displ[global_variable::my_rank]] = counts_thisrank[n];
  }


  MPI_Datatype mpi_ituple;
  MPI_Type_contiguous(3, MPI_INT, &mpi_ituple);
  MPI_Allgatherv(MPI_IN_PLACE, ncounts_eachrank[global_variable::my_rank],
                   mpi_ituple, counts_allranks.data(), ncounts_eachrank.data(),
                   ncounts_displ.data(), mpi_ituple, mpi_comm_part);
/***/
{
int ierr = MPI_Barrier(MPI_COMM_WORLD);
if (global_variable::my_rank == 0) {
for (int n=0; n<ncounts_allranks; ++n) {
std::cout << "n="<<n<< "  (sendrank,destrank,npart)=" << std::get<0>(counts_allranks[n])<<"  "<<std::get<1>(counts_allranks[n]) << "  "<<std::get<2>(counts_allranks[n]) << std::endl;
}
}
}
/****/

#endif
  return TaskStatus::complete;
}

/*
//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::InitPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::InitPrtclRecv() {
#if MPI_PARALLEL_ENABLED

  // Figure out how many ranks will send to this rank
  // load STL::vector with <origin_rank,nprtcl_recv> tuples. Length is nranks_recvfrom
  ranks_recvfrom.clear();
  if (npart_recv > 0) {
    int rank = sendlist_h(0).dest_rank;
    int nprtcl = 1;

    for (int n=0; n<npart_send; ++n) {
      if (sendlist_h(n).dest_rank == rank) {
        ++nprtcl;
      } else {
        ranks_tosendto.emplace_back(std::make_pair(rank,nprtcl));
        rank = sendlist_h(0).dest_rank;
        nprtcl = 1;
      }
    }
  }
  nranks_tosendto = ranks_tosendto.size();

  // Figure out how many partciles will be received from each rank that sends
  //
  // Allocate receive buffer
  Kokkos::realloc(part_recvbuff, nprtcl_recv);

  // Post non-blocking receives

#endif
}

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
  // Sort DualArray by particle index
  // Use DualArray of tuples to load particles into particle array at blank index spots
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
