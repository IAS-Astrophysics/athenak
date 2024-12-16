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
#include <vector>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/nghbr_index.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::UpdateGID()
//! \brief Updates GID of particles that cross boundary of their parent MeshBlock.  If
//! the new GID is on a different rank, then store in sendlist_buf DvceArray: (1) index of
//! particle in prtcl array, (2) destination GID, and (3) destination rank.

KOKKOS_INLINE_FUNCTION
void UpdateGID(int &newgid, NeighborBlock nghbr, int myrank, int *pcounter,
               DualArray1D<ParticleLocationData> slist, int p) {
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
//! \fn void ParticlesBoundaryValues::MarkForDestruction()
//! \brief Adds particles that cross the user-defined mesh boundariesto a list used then
//! to destroy them. The list uses the same structure as the sendlist for convenience
//! in the following functions.
//! This opeartion should also be performed for single process runs

KOKKOS_INLINE_FUNCTION
void MarkForDestruction(int *pcounter, DualArray1D<ParticleLocationData> dlist, int p) {
    int index = Kokkos::atomic_fetch_add(pcounter,1);
    dlist.d_view(index).prtcl_indx = p;
    // These particles don't actually get sent, thus following information is not needed
    dlist.d_view(index).dest_gid   = 0;
    dlist.d_view(index).dest_rank  = 0;
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
  auto &mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto &mblev = pmy_part->pmy_pack->pmb->mb_lev;
  auto &meshsize = pmy_part->pmy_pack->pmesh->mesh_size;
  auto myrank = global_variable::my_rank;
  auto &nghbr = pmy_part->pmy_pack->pmb->nghbr;
  auto &psendl = sendlist;
  int counter=0;
  Kokkos::View<int> atom_count("atom_count");
  Kokkos::deep_copy(atom_count, counter);
  bool &multi_d = pmy_part->pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_part->pmy_pack->pmesh->three_d;
  // Compatibility with user-defined boundary conditions
  auto &mb_bcs = pmy_part->pmy_pack->pmb->mb_bcs;
  int destroy_count = 0;
  Kokkos::View<int> atom_d_count("atom_d_count");
  Kokkos::deep_copy(atom_d_count, destroy_count);
  auto &pdestroyl = destroylist;

  Kokkos::realloc(sendlist, static_cast<int>(npart));
  Kokkos::realloc(destroylist, static_cast<int>(npart));
  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int mylevel = mblev.d_view(m);
    Real x1 = pr(IPX,p);
    Real x2 = pr(IPY,p);
    Real x3 = pr(IPZ,p);

    // length of MeshBlock in each direction
    Real lx = (mbsize.d_view(m).x1max - mbsize.d_view(m).x1min);
    Real ly = (mbsize.d_view(m).x2max - mbsize.d_view(m).x2min);
    Real lz = (mbsize.d_view(m).x3max - mbsize.d_view(m).x3min);

    // integer offset of particle relative to center of MeshBlock (-1,0,+1)
    int ix = static_cast<int>((x1 - mbsize.d_view(m).x1min + lx)/lx) - 1;
    int iy = static_cast<int>((x2 - mbsize.d_view(m).x2min + ly)/ly) - 1;
    int iz = static_cast<int>((x3 - mbsize.d_view(m).x3min + lz)/lz) - 1;

    // sublock indices for faces and edges with S/AMR
    int fx = (x1 < 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max))? 0 : 1;
    int fy = (x2 < 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max))? 0 : 1;
    int fz = (x3 < 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max))? 0 : 1;
    fy = multi_d ? fy : 0;
    fz = three_d ? fz : 0;

    // only update particle GID if it has crossed MeshBlock boundary
    if ((abs(ix) + abs(iy) + abs(iz)) != 0) {
      // The way GIDs are determined currently is not reliable in SMR cases
      // due to nighbours across edges and corners being uninitialized.
      // Need extra check
      bool send_to_coarser = false;
      if (ix < 0) {
        // level might be initizialized to -1 on some neighbours
        for (int iop = 0; iop <= 3; ++iop) {
          if (nghbr.d_view(m,iop).lev < mylevel && nghbr.d_view(m,iop).lev > 0) {
            send_to_coarser = true;
          }
        }
      } else if (ix > 0) {
         for (int iop = 4; iop <= 7; ++iop) {
          if (nghbr.d_view(m,iop).lev < mylevel && nghbr.d_view(m,iop).lev > 0) {
            send_to_coarser = true;
          }
         }
      }
      if (iy < 0) {
        for (int iop = 8; iop <= 11; ++iop) {
          if (nghbr.d_view(m,iop).lev < mylevel && nghbr.d_view(m,iop).lev > 0) {
            send_to_coarser = true;
          }
        }
      } else if (iy > 0) {
        for (int iop = 12; iop <= 15; ++iop) {
          if (nghbr.d_view(m,iop).lev < mylevel && nghbr.d_view(m,iop).lev > 0) {
            send_to_coarser = true;
          }
        }
      }
      if (iz < 0) {
        for (int iop = 24; iop <= 27; ++iop) {
          if (nghbr.d_view(m,iop).lev < mylevel && nghbr.d_view(m,iop).lev > 0) {
            send_to_coarser = true;
          }
        }
      } else if (iz > 0) {
          for (int iop = 28; iop <= 31; ++iop) {
            if (nghbr.d_view(m,iop).lev < mylevel && nghbr.d_view(m,iop).lev > 0) {
              send_to_coarser = true;
            }
        }
      }
      int indx = 0;
      bool check_boundary =
        ( mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user && iz < 0)
        || ( mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user && iz > 0)
        || ( mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user && iy < 0)
        || ( mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user && iy > 0)
        || ( mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user && ix < 0)
        || ( mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user && ix > 0);
      // Add particle to destruction list
      // At the time of sending the particles that need to be destroyed
      // are treated like those that have been sent
      // without actually being sent (i.e. they're remove from arrays)
      if (check_boundary) {
        MarkForDestruction(&atom_d_count(), pdestroyl, p);
      } else {
        if (iz == 0) {
          if (iy == 0) {
            // x1 face
            indx = NeighborIndex(ix,0,0,0,0);           // neighbor at same level
            if (nghbr.d_view(m,indx).lev > mylevel) {       // neighbor at finer level
              indx = NeighborIndex(ix,0,0,fy,fz);
            }
            while (nghbr.d_view(m,indx).gid < 0) {indx++;}  // neighbor at coarser level
            UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, &atom_count(), psendl, p);
          } else if (ix == 0) {
            // x2 face
            indx = NeighborIndex(0,iy,0,0,0);
            if (nghbr.d_view(m,indx).lev > mylevel) {
              indx = NeighborIndex(0,iy,0,fx,fz);
            }
            while (nghbr.d_view(m,indx).gid < 0) {indx++;}
            UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, &atom_count(), psendl, p);
          } else {
            // x1x2 edge
            indx = NeighborIndex(ix,iy,0,0,0);
            if (nghbr.d_view(m,indx).lev > mylevel) {
              indx = NeighborIndex(ix,iy,0,fz,0);
            }
            while (nghbr.d_view(m,indx).gid < 0) {indx++;}
            // Using SMR some edge and corner neighbours are uninitialized,
            // thus check if the index has increased over the appropriate range
            // and try to communicate through faces to a coarser meshblock
            // Communication to coarser meshblocks should always go through faces
            if (indx > 23 || send_to_coarser) {
              bool found_coarser = false;
              // First try through x face
              indx = NeighborIndex(ix,0,0,0,0);
              while (nghbr.d_view(m,indx).gid < 0) {indx++;}
              if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              // If all faces in the x direction are on a finer level this should
              // have already been covered by previous logic, thus check y
              if ( !found_coarser ) {
                indx = NeighborIndex(0,iy,0,0,0);
                while (nghbr.d_view(m,indx).gid < 0) {indx++;}
                if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              }
            }
            UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, &atom_count(), psendl, p);
          }
        } else if (iy == 0) {
          if (ix == 0) {
            // x3 face
            indx = NeighborIndex(0,0,iz,0,0);
            if (nghbr.d_view(m,indx).lev > mylevel) {
              indx = NeighborIndex(0,0,iz,fx,fy);
            }
            while (nghbr.d_view(m,indx).gid < 0) {indx++;}
            UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, &atom_count(), psendl, p);
          } else {
            // x3x1 edge
            indx = NeighborIndex(ix,0,iz,0,0);
            if (nghbr.d_view(m,indx).lev > mylevel) {
              indx = NeighborIndex(ix,0,iz,fy,0);
            }
            while (nghbr.d_view(m,indx).gid < 0) {indx++;}
            if (indx > 39 || send_to_coarser) {
              bool found_coarser = false;
              indx = NeighborIndex(ix,0,0,0,0);
              while (nghbr.d_view(m,indx).gid < 0) {indx++;}
              if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              if ( !found_coarser ) {
                indx = NeighborIndex(0,0,iz,0,0);
                while (nghbr.d_view(m,indx).gid < 0) {indx++;}
                if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              }
            }
            UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, &atom_count(), psendl, p);
          }
        } else {
          if (ix == 0) {
            // x2x3 edge
            indx = NeighborIndex(0,iy,iz,0,0);
            if (nghbr.d_view(m,indx).lev > mylevel) {
              indx = NeighborIndex(0,iy,iz,fx,0);
            }
            while (nghbr.d_view(m,indx).gid < 0) {indx++;}
            if (indx > 47 || send_to_coarser) {
              bool found_coarser = false;
              indx = NeighborIndex(0,iy,0,0,0);
              while (nghbr.d_view(m,indx).gid < 0) {indx++;}
              if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              if ( !found_coarser ) {
                indx = NeighborIndex(0,0,iz,0,0);
                while (nghbr.d_view(m,indx).gid < 0) {indx++;}
                if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              }
            }
            UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, &atom_count(), psendl, p);
          } else {
            // corners
            indx = NeighborIndex(ix,iy,iz,0,0);
            if (nghbr.d_view(m,indx).gid < 0 || send_to_coarser) {
              bool found_coarser = false;
              indx = NeighborIndex(ix,0,0,0,0);
              while (nghbr.d_view(m,indx).gid < 0) {indx++;}
              if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              if ( !found_coarser ) {
                indx = NeighborIndex(0,iy,0,0,0);
                while (nghbr.d_view(m,indx).gid < 0) {indx++;}
                if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              }
              if ( !found_coarser ) {
                indx = NeighborIndex(0,0,iz,0,0);
                while (nghbr.d_view(m,indx).gid < 0) {indx++;}
                if (nghbr.d_view(m,indx).lev < mylevel) { found_coarser = true; }
              }
            }
            UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, &atom_count(), psendl, p);
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
      }
    }
  });
  Kokkos::deep_copy(counter, atom_count);
  Kokkos::deep_copy(destroy_count, atom_d_count);
  nprtcl_send = counter;
  nprtcl_destroy = destroy_count;
  Kokkos::resize(sendlist, nprtcl_send);
  Kokkos::resize(destroylist, nprtcl_destroy);
  // sync sendlist device array with host
  sendlist.template modify<DevExeSpace>();
  sendlist.template sync<HostMemSpace>();
  // sync destroylist device array with host
  destroylist.template modify<DevExeSpace>();
  destroylist.template sync<HostMemSpace>();

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

  // load STL::vector of ParticleMessageData with <sendrank, recvrank, nprtcls> for sends
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
        sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
        rank = sendlist.h_view(n).dest_rank;
        nprtcl = 1;
      }
    }
    sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
  }
  nsends = sends_thisrank.size();

  // Share number of ranks to send to amongst all ranks
  nsends_eachrank[global_variable::my_rank] = nsends;
  MPI_Allgather(&nsends, 1, MPI_INT, nsends_eachrank.data(), 1, MPI_INT, mpi_comm_part);

  // Now share ParticleMessageData amongst all ranks
  // First create vector of starting indices in full vector
  std::vector<int> nsends_displ;
  nsends_displ.resize(global_variable::nranks);
  nsends_displ[0] = 0;
  for (int n=1; n<(global_variable::nranks); ++n) {
    nsends_displ[n] = nsends_displ[n-1] + nsends_eachrank[n-1];
  }
  int nsends_allranks = nsends_displ[global_variable::nranks - 1] +
                        nsends_eachrank[global_variable::nranks - 1];
  // Load ParticleMessageData on this rank into full vector
  sends_allranks.resize(nsends_allranks, ParticleMessageData(0,0,0));
  for (int n=0; n<nsends_eachrank[global_variable::my_rank]; ++n) {
    sends_allranks[n + nsends_displ[global_variable::my_rank]] = sends_thisrank[n];
  }

  // Share tuples using MPI derived data type for tuple of 3*int
  MPI_Datatype mpi_ituple;
  MPI_Type_contiguous(3, MPI_INT, &mpi_ituple);
  MPI_Allgatherv(MPI_IN_PLACE, nsends_eachrank[global_variable::my_rank],
                   mpi_ituple, sends_allranks.data(), nsends_eachrank.data(),
                   nsends_displ.data(), mpi_ituple, mpi_comm_part);
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::InitPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::InitPrtclRecv() {
  // Moved assignment here for better compatibility with destruction logic
  nprtcl_recv=0;
#if MPI_PARALLEL_ENABLED
  // load STL::vector of ParticleMessageData with <sendrank,recvrank,nprtcl_recv> for
  // receives // on this rank. Length will be nrecvs, initially this length is unknown
  recvs_thisrank.clear();

  int nsends_allranks = sends_allranks.size();
  for (int n=0; n<nsends_allranks; ++n) {
    if (sends_allranks[n].recvrank == global_variable::my_rank) {
      recvs_thisrank.emplace_back(sends_allranks[n]);
    }
  }
  nrecvs = recvs_thisrank.size();

  // Figure out how many particles will be received from all ranks
  for (int n=0; n<nrecvs; ++n) {
    nprtcl_recv += recvs_thisrank[n].nprtcls;
  }

  // Allocate receive buffer
  Kokkos::realloc(prtcl_rrecvbuf, (pmy_part->nrdata)*nprtcl_recv);
  Kokkos::realloc(prtcl_irecvbuf, (pmy_part->nidata)*nprtcl_recv);

  // Post non-blocking receives
  bool no_errors=true;
  rrecv_req.clear();
  irecv_req.clear();
  for (int n=0; n<nrecvs; ++n) {
    rrecv_req.emplace_back(MPI_REQUEST_NULL);
    irecv_req.emplace_back(MPI_REQUEST_NULL);
  }

  // Init receives for Reals
  int data_start=0;
  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = (pmy_part->nrdata)*(recvs_thisrank[n].nprtcls);
    int data_end = data_start + (pmy_part->nrdata)*(recvs_thisrank[n].nprtcls - 1);
    auto recv_ptr = Kokkos::subview(prtcl_rrecvbuf, std::make_pair(data_start, data_end));
    int drank = recvs_thisrank[n].sendrank;
    int tag = 0; // 0 for Reals, 1 for ints

    // Post non-blocking receive
    int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                         mpi_comm_part, &(rrecv_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    data_start += data_size;
  }
  // Init receives for ints
  data_start=0;
  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = (pmy_part->nidata)*(recvs_thisrank[n].nprtcls);
    int data_end = data_start + (pmy_part->nidata)*(recvs_thisrank[n].nprtcls - 1);
    auto recv_ptr = Kokkos::subview(prtcl_irecvbuf, std::make_pair(data_start, data_end));
    int drank = recvs_thisrank[n].sendrank;
    int tag = 1; // 0 for Reals, 1 for ints

    // Post non-blocking receive
    int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_INT, drank, tag,
                         mpi_comm_part, &(irecv_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    data_start += data_size;
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
    nprtcl_send += sends_thisrank[n].nprtcls;
  }

  bool no_errors=true;
  if (nprtcl_send > 0) {
    // Allocate send buffer
    Kokkos::realloc(prtcl_rsendbuf, (pmy_part->nrdata)*nprtcl_send);
    Kokkos::realloc(prtcl_isendbuf, (pmy_part->nidata)*nprtcl_send);

    // sendlist on device is already sorted by destrank in CountSendAndRecvs()
    // Use sendlist on device to load particles into send buffer ordered by dest_rank
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rsendbuf = prtcl_rsendbuf;
    auto &isendbuf = prtcl_isendbuf;
    par_for("ppack",DevExeSpace(),0,(nprtcl_send-1), KOKKOS_LAMBDA(const int n) {
      int p = sendlist.d_view(n).prtcl_indx;
      for (int i=0; i<nidata; ++i) {
        isendbuf(nidata*n + i) = pi(i,p);
      }
      for (int i=0; i<nrdata; ++i) {
        rsendbuf(nrdata*n + i) = pr(i,p);
      }
    });

    // Post non-blocking sends
    Kokkos::fence();
    rsend_req.clear();
    isend_req.clear();
    for (int n=0; n<nsends; ++n) {
      rsend_req.emplace_back(MPI_REQUEST_NULL);
      isend_req.emplace_back(MPI_REQUEST_NULL);
    }

    // Send Reals
    int data_start=0;
    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed, get pointer to variables
      int data_size = nrdata*(sends_thisrank[n].nprtcls);
      int data_end = data_start + nrdata*(sends_thisrank[n].nprtcls - 1);
      auto send_ptr = Kokkos::subview(prtcl_rsendbuf,std::make_pair(data_start,data_end));
      int drank = sends_thisrank[n].recvrank;
      int tag = 0; // 0 for Reals, 1 for ints

      // Post non-blocking sends
      int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                           mpi_comm_part, &(rsend_req[n]));
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      data_start += data_size;
    }
    // Send ints
    data_start=0;
    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed, get pointer to variables
      int data_size = nidata*(sends_thisrank[n].nprtcls);
      int data_end = data_start + nidata*(sends_thisrank[n].nprtcls - 1);
      auto send_ptr = Kokkos::subview(prtcl_isendbuf,std::make_pair(data_start,data_end));
      int drank = sends_thisrank[n].recvrank;
      int tag = 1; // 0 for Reals, 1 for ints

      // Post non-blocking sends
      int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_INT, drank, tag,
                           mpi_comm_part, &(isend_req[n]));
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      data_start += data_size;
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
  namespace KE = Kokkos::Experimental;
  //particle destruction must happen also for single process runs
  std::sort(KE::begin(destroylist.h_view), KE::end(destroylist.h_view), SortByIndex);
  // sync destroylist host array with device.  This results in sorted array on device
  destroylist.template modify<HostMemSpace>();
  destroylist.template sync<DevExeSpace>();

  //In single process runs, size of particle arrays can only be reduced
  int &npart = pmy_part->nprtcl_thispack;
  int new_npart = npart + (nprtcl_recv - nprtcl_send - nprtcl_destroy);

  // nremain defined outside compiler instructions for end logic
  int nremain = 0;
  int nremain_d = nprtcl_send + nprtcl_destroy - nprtcl_recv;

#if MPI_PARALLEL_ENABLED
  // Sort sendlist on host by index in particle array
  std::sort(KE::begin(sendlist.h_view), KE::end(sendlist.h_view), SortByIndex);
  // sync sendlist host array with device.  This results in sorted array on device
  sendlist.template modify<HostMemSpace>();
  sendlist.template sync<DevExeSpace>();

  // increase size of particle arrays if needed
  if (nprtcl_recv > (nprtcl_send + nprtcl_destroy) ) {
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, new_npart);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, new_npart);
  }

  // check that particle communications have all completed
  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<nrecvs; ++n) {
    int test;
    int ierr = MPI_Test(&(rrecv_req[n]), &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
    ierr = MPI_Test(&(irecv_req[n]), &test, MPI_STATUS_IGNORE);
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

  // unpack particles into positions of sent or destroyed particles
  if (nprtcl_recv > 0) {
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rrecvbuf = prtcl_rrecvbuf;
    auto &irecvbuf = prtcl_irecvbuf;
    par_for("punpack",DevExeSpace(),0,(nprtcl_recv-1), KOKKOS_LAMBDA(const int n) {
      int p;
      if (n < nprtcl_send ) {
        p = sendlist.d_view(n).prtcl_indx;  // place particles in holes created by send
      } else if (n < nprtcl_send + nprtcl_destroy ) {
        // place particles in holes created by destroy
        p = destroylist.d_view(n-nprtcl_send).prtcl_indx;
      } else {
        // place particle at end of arrays
        p = npart + (n - nprtcl_send - nprtcl_destroy );
      }
      for (int i=0; i<nidata; ++i) {
        pi(i,p) = irecvbuf(nidata*n + i);
      }
      for (int i=0; i<nrdata; ++i) {
        pr(i,p) = rrecvbuf(nrdata*n + i);
      }
    });
  }

  // At this point have filled npart_recv holes in particle arrays from sends
  // If (nprtcl_recv < nprtcl_send), have to move particles from end of arrays to fill
  // remaining holes
  nremain = nprtcl_send - nprtcl_recv;
  if (nremain > 0) {
    int i_last_hole = nprtcl_send-1;
    int i_next_hole = nprtcl_recv;
    for (int n=1; n<=nremain; ++n) {
      int nend = npart-n;
      --nremain_d;
      if (nend > sendlist.h_view(i_last_hole).prtcl_indx) {
        // copy particle from end into hole
        int next_hole = sendlist.h_view(i_next_hole).prtcl_indx;
        auto rdest = Kokkos::subview(pmy_part->prtcl_rdata, Kokkos::ALL, next_hole);
        auto rsrc  = Kokkos::subview(pmy_part->prtcl_rdata, Kokkos::ALL, nend);
        Kokkos::deep_copy(rdest, rsrc);
        auto idest = Kokkos::subview(pmy_part->prtcl_idata, Kokkos::ALL, next_hole);
        auto isrc  = Kokkos::subview(pmy_part->prtcl_idata, Kokkos::ALL, nend);
        Kokkos::deep_copy(idest, isrc);
        i_next_hole += 1;
      } else {
        // this index contains a hole, so do nothing except find new index of last hole
        i_last_hole -= 1;
      }
    }
  }

  // Update nparticles_thisrank.  Update cost array (use npart_thismb[nmb]?)
  MPI_Allgather(&new_npart,1,MPI_INT,(pmy_part->pmy_pack->pmesh->nprtcl_eachrank),1,
                MPI_INT,MPI_COMM_WORLD);
#endif
  // If there are still holes due to particle destruction
  //Destroy particles by moving remaining particles in their places
  if (nremain_d > 0) {
    int i_last_hole = nprtcl_destroy-1;
    // If the holes left by sending have been entirely covered by receives
    // and there are more receives left, start from where you left off in the
    // (nprtcl_recv>0) branch, otherwise simply destroy all particles in the list
    int i_next_hole = nprtcl_recv-nprtcl_send < 0 ? 0 : nprtcl_recv-nprtcl_send;
    for (int n=1; n<=nremain_d; ++n) {
      //At this point already nremain particles might have been removed, check for that
      int nend = nremain > 0 ? npart-nremain-n : npart - n;
      if (nend > destroylist.h_view(i_last_hole).prtcl_indx) {
        // copy particle from end into hole
        int next_hole = destroylist.h_view(i_next_hole).prtcl_indx;
        auto rdest = Kokkos::subview(pmy_part->prtcl_rdata, Kokkos::ALL, next_hole);
        auto rsrc  = Kokkos::subview(pmy_part->prtcl_rdata, Kokkos::ALL, nend);
        Kokkos::deep_copy(rdest, rsrc);
        auto idest = Kokkos::subview(pmy_part->prtcl_idata, Kokkos::ALL, next_hole);
        auto isrc  = Kokkos::subview(pmy_part->prtcl_idata, Kokkos::ALL, nend);
        Kokkos::deep_copy(idest, isrc);
        i_next_hole += 1;
      } else {
        // this index contains a hole, so do nothing except find new index of last hole
        i_last_hole -= 1;
      }
    }
  }
  if (nprtcl_send + nprtcl_destroy - nprtcl_recv > 0) {
    // shrink size of particle data arrays
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, new_npart);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, new_npart);
  }
  pmy_part->nprtcl_thispack = new_npart;
  pmy_part->pmy_pack->pmesh->nprtcl_thisrank = new_npart;
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
    int ierr = MPI_Wait(&(rsend_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    ierr = MPI_Wait(&(isend_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rsend_req.clear();
  isend_req.clear();
#endif
  nsends=0;
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
    int ierr = MPI_Wait(&(rrecv_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    ierr = MPI_Wait(&(irecv_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rrecv_req.clear();
  irecv_req.clear();
#endif
  nrecvs=0;
  return TaskStatus::complete;
}

} // namespace particles
