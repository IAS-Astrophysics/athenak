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
//! the new GID is on a different rank, then store in sendlist_buf DvceArray: (1) index of
//! particle in prtcl array, (2) destination GID, and (3) destination rank.

namespace particles {

KOKKOS_INLINE_FUNCTION
void UpdateGID(int &newgid, NeighborBlock nghbr, int myrank, int *pcounter,
               DualArray1D<ParticleSendData> slist, int p) {
  newgid = nghbr.gid;
#if MPI_PARALLEL_ENABLED
  if (nghbr.rank != myrank) {
    int index = Kokkos::atomic_fetch_add(pcounter,1);
    slist.d_view(index).prtcl_indx = p;
    slist.d_view(index).dest_gid   = nghbr.gid;
    slist.d_view(index).dest_rank  = nghbr.rank;
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
  auto &pr = pmy_part->prtcl_rdata;
  auto &pi = pmy_part->prtcl_idata;
  int npart = pmy_part->nprtcl_thispack;
  auto mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto meshsize = pmy_part->pmy_pack->pmesh->mesh_size;
  auto myrank = global_variable::my_rank;
  auto nghbr = pmy_part->pmy_pack->pmb->nghbr;
  auto &psendl = sendlist;
  int counter=0;
  int *pcounter = &counter;

  Kokkos::realloc(sendlist, static_cast<int>(0.1*npart));
  par_for("part_update",DevExeSpace(),0,npart, KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    Real x1 = pr(IPX,p);
    Real x2 = pr(IPY,p);
    Real x3 = pr(IPZ,p);

    if (x1 < mbsize.d_view(m).x1min) {
      if (x2 < mbsize.d_view(m).x2min) {
        if (x3 < mbsize.d_view(m).x3min) {
          // corner
          UpdateGID(pi(PGID,p), nghbr.d_view(m,48), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // corner
          UpdateGID(pi(PGID,p), nghbr.d_view(m,52), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,16), myrank, pcounter, psendl, p);
        }
      } else if (x2 > mbsize.d_view(m).x2max) {
        if (x3 < mbsize.d_view(m).x3min) {
          // corner
          UpdateGID(pi(PGID,p), nghbr.d_view(m,50), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // corner
          UpdateGID(pi(PGID,p), nghbr.d_view(m,54), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,20), myrank, pcounter, psendl, p);
        }
      } else {
        if (x3 < mbsize.d_view(m).x3min) {
          // x3x1 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,32), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x3x1 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,36), myrank, pcounter, psendl, p);
        } else {
          // x1 face
          UpdateGID(pi(PGID,p), nghbr.d_view(m,0), myrank, pcounter, psendl, p);
        }
      }

    } else if (x1 > mbsize.d_view(m).x1max) {
      if (x2 < mbsize.d_view(m).x2min) {
        if (x3 < mbsize.d_view(m).x3min) {
          // corner
          UpdateGID(pi(PGID,p), nghbr.d_view(m,49), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // corner
          UpdateGID(pi(PGID,p), nghbr.d_view(m,53), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,18), myrank, pcounter, psendl, p);
        }
      } else if (x2 > mbsize.d_view(m).x2max) {
        if (x3 < mbsize.d_view(m).x3min) {
          // corner
          UpdateGID(pi(PGID,p), nghbr.d_view(m,51), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // corner
          UpdateGID(pi(PGID,p), nghbr.d_view(m,55), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,22), myrank, pcounter, psendl, p);
        }
      } else {
        if (x3 < mbsize.d_view(m).x3min) {
          // x3x1 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,34), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x3x1 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,38), myrank, pcounter, psendl, p);
        } else {
          // x1 face
          UpdateGID(pi(PGID,p), nghbr.d_view(m,4), myrank, pcounter, psendl, p);
        }
      }

    } else {
      if (x2 < mbsize.d_view(m).x2min) {
        if (x3 < mbsize.d_view(m).x3min) {
          // x2x3 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,40), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x2x3 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,44), myrank, pcounter, psendl, p);
        } else {
          // x2 face
          UpdateGID(pi(PGID,p), nghbr.d_view(m,8), myrank, pcounter, psendl, p);
        }
      } else if (x2 > mbsize.d_view(m).x2max) {
        if (x3 < mbsize.d_view(m).x3min) {
          // x2x3 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,42), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x2x3 edge
          UpdateGID(pi(PGID,p), nghbr.d_view(m,46), myrank, pcounter, psendl, p);
        } else {
          // x2 face
          UpdateGID(pi(PGID,p), nghbr.d_view(m,12), myrank, pcounter, psendl, p);
        }
      } else {
        if (x2 < mbsize.d_view(m).x2min) {
          // x3 face
          UpdateGID(pi(PGID,p), nghbr.d_view(m,24), myrank, pcounter, psendl, p);
        } else if (x3 > mbsize.d_view(m).x3max) {
          // x3 face
          UpdateGID(pi(PGID,p), nghbr.d_view(m,28), myrank, pcounter, psendl, p);
        }
      }
    }

    // reset x,y,z positions if particle crosses Mesh boundary using periodic BCs
    if (x1 < meshsize.x1min) {
      pr(IPX,p) += (meshsize.x1max - meshsize.x1min);
    } else if (x1 > meshsize.x1max) {
      pr(IPX,p) -= (meshsize.x1max - meshsize.x1min);
    }
    if (x2 < meshsize.x2min) {
      pr(IPY,p) += (meshsize.x2max - meshsize.x2min);
    } else if (x2 > meshsize.x2max) {
      pr(IPY,p) -= (meshsize.x2max - meshsize.x2min);
    }
    if (x3 < meshsize.x3min) {
      pr(IPZ,p) += (meshsize.x3max - meshsize.x3min);
    } else if (x3 > meshsize.x3max) {
      pr(IPZ,p) -= (meshsize.x3max - meshsize.x3min);
    }
  });
  nprtcl_send = counter;
  Kokkos::resize(sendlist, nprtcl_send);
  // sync sendlist device array with host
  sendlist.template modify<DevExeSpace>();
  sendlist.template sync<HostMemSpace>();

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::CountSendsAndRecvs()
//! \brief

TaskStatus ParticlesBoundaryValues::CountSendsAndRecvs() {
#if MPI_PARALLEL_ENABLED
  // Sort sendlist on host by destrank.
  namespace KE = Kokkos::Experimental;
  std::sort(KE::begin(sendlist.h_view), KE::end(sendlist.h_view), SortByRank);
  // sync sendlist host array with device.  This results in sorted array on device
  sendlist.template modify<HostMemSpace>();
  sendlist.template sync<DevExeSpace>();

/***/
for (int n=0; n<nprtcl_send; ++n) {
std::cout << "rank="<<global_variable::my_rank<<"  (n,indx,rank,gid)=" << n<<"  "<<sendlist.h_view(n).prtcl_indx<<"  "<<sendlist.h_view(n).dest_rank<<"  "<<sendlist.h_view(n).dest_gid << std::endl;
}
/****/

  // load STL::vector with <sendrank, recvrank, nprtcl_tosend> tuples for particles sends
  // from this rank. Length will be nsends; initially this length is unknown
  sends_thisrank.clear();
  if (nprtcl_send > 0) {
    int &myrank = global_variable::my_rank;
    int rank = sendlist.h_view(0).dest_rank;
    int nprtcl = 1;

    for (int n=1; n<nprtcl_send; ++n) {
      if (sendlist.h_view(n).dest_rank == rank) {
        ++nprtcl;
      } else {
        sends_thisrank.emplace_back(std::make_tuple(myrank,rank,nprtcl));
        rank = sendlist.h_view(n).dest_rank;
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

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::PackAndSendPrtcls()
//! \brief

TaskStatus ParticlesBoundaryValues::PackAndSendPrtcls() {
#if MPI_PARALLEL_ENABLED
  // Figure out how many particles will be sent from this ranks
  nprtcl_send=0;
  for (int n=0; n<nsends; ++n) {
    nprtcl_send += std::get<2>(sends_thisrank[n]);
  }

  bool no_errors=true;
  if (nprtcl_send > 0) {
    // Allocate send buffer
    Kokkos::realloc(prtcl_sendbuf, nprtcl_send);

    // sendlist on device is already sorted by destrank in CountSendAndRecvs()
    // Use sendlist on device to load particles into send buffer ordered by dest_rank
    auto &pr = pmy_part->prtcl_rdata;
    par_for("ppack",DevExeSpace(),0,nprtcl_send, KOKKOS_LAMBDA(const int n) {
      prtcl_sendbuf(n).dest_gid = sendlist.d_view(n).dest_gid;
      int p = sendlist.d_view(n).prtcl_indx;
      prtcl_sendbuf(n).x  = pr(IPX,p);
      prtcl_sendbuf(n).y  = pr(IPY,p);
      prtcl_sendbuf(n).z  = pr(IPZ,p);
      prtcl_sendbuf(n).vx = pr(IPVX,p);
      prtcl_sendbuf(n).vy = pr(IPVY,p);
      prtcl_sendbuf(n).vz = pr(IPVZ,p);
    });

    // Post non-blocking sends
    Kokkos::fence();
    int data_start=0;
    send_req.clear();
    for (int n=0; n<nsends; ++n) { send_req.emplace_back(MPI_REQUEST_NULL); }

    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed, get pointer to variables
      int data_size = std::get<2>(sends_thisrank[n])*sizeof(ParticleData);
      int data_end = data_start + std::get<2>(sends_thisrank[n]);
      auto send_ptr = Kokkos::subview(prtcl_sendbuf, std::make_pair(data_start,data_end));
      int drank = std::get<1>(sends_thisrank[n]);

      // Post non-blocking sends
      int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_BYTE, drank, MPI_ANY_TAG,
                           mpi_comm_part, &(send_req[n]));
      if (ierr != MPI_SUCCESS) {no_errors=false;}
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
//! \fn void ParticlesBoundaryValues::RecvAndUnpackPrtcls()
//! \brief

TaskStatus ParticlesBoundaryValues::RecvAndUnpackPrtcls() {
#if MPI_PARALLEL_ENABLED
  // Sort sendlist on host by index in particle array
  namespace KE = Kokkos::Experimental;
  std::sort(KE::begin(sendlist.h_view), KE::end(sendlist.h_view), SortByIndex);
  // sync sendlist host array with device.  This results in sorted array on device
  sendlist.template modify<HostMemSpace>();
  sendlist.template sync<DevExeSpace>();

  // increase size of particle arrays if needed
  if (nprtcl_recv > nprtcl_send) {
    int new_npart = pmy_part->nprtcl_thispack + (nprtcl_recv - nprtcl_send);
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, new_npart);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, new_npart);
  }

  // check that particle communications have all completed
  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<nrecvs; ++n) {
    int test;
    int ierr = MPI_Test(&(recv_req[n]), &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if particle communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  // unpack particles into positions of sent particles
  if (nprtcl_recv > 0) {
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    int npart = pmy_part->nprtcl_thispack;
    par_for("punpack",DevExeSpace(),0,nprtcl_recv, KOKKOS_LAMBDA(const int n) {
      int p;
      if (n < nprtcl_send) {
        p = sendlist.d_view(n).prtcl_indx; // place particles in holes created by sends
      } else {
        p = npart + (n - nprtcl_send);     // place particle at end of arrays
      }
      pi(PGID,p) = prtcl_recvbuf(n).dest_gid;
      pr(IPX, p) = prtcl_recvbuf(n).x;
      pr(IPY, p) = prtcl_recvbuf(n).y;
      pr(IPZ, p) = prtcl_recvbuf(n).z;
      pr(IPVX,p) = prtcl_sendbuf(n).vx;
      pr(IPVY,p) = prtcl_sendbuf(n).vy;
      pr(IPVZ,p) = prtcl_sendbuf(n).vz;
    });

    // At this point have filled npart_recv holes in particle arrays from sends
    // If (nprtcl_recv < nprtcl_send), have to move particles from end of arrays to fill
    // remaining holes
    int nremain = nprtcl_send - nprtcl_recv;
    if (nremain > 0) {
      int nend_pdata = npart-1;
      int nend_sendl = nprtcl_send-1;
      for (int n=0; n<nremain; ++n) {
        if (nend_pdata > sendlist.h_view(nend_sendl).prtcl_indx) {
          // copy particle from end into hole
          auto rdest = Kokkos::subview(pmy_part->prtcl_rdata, Kokkos::ALL, nend_sendl);
          auto rsrc  = Kokkos::subview(pmy_part->prtcl_rdata, Kokkos::ALL, nend_pdata);
          Kokkos::deep_copy(rdest, rsrc);
          auto idest = Kokkos::subview(pmy_part->prtcl_idata, Kokkos::ALL, nend_sendl);
          auto isrc  = Kokkos::subview(pmy_part->prtcl_idata, Kokkos::ALL, nend_pdata);
          Kokkos::deep_copy(idest, isrc);

          // update indices
          nend_pdata--;
          nend_sendl--;
        } else {
          // if indices equal, hole is at end of array, so skip
          nend_pdata--;
          nend_sendl--;
        }
      }
    }
  }

  // Update nparticles_thisrank.  Update cost array (use npart_thismb[nmb]?)
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearPrtclSend()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearPrtclSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  // wait for all non-blocking sends for vars to finish before continuing
  for (int n=0; n<nsends; ++n) {
    int ierr = MPI_Wait(&(send_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
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
//! \fn void ParticlesBoundaryValues::ClearPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearPrtclRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  // wait for all non-blocking receives to finish before continuing
  for (int n=0; n<nrecvs; ++n) {
    int ierr = MPI_Wait(&(recv_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
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

} // end namaspace particles
