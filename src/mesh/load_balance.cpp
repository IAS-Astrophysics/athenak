//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file load_balance.cpp
//! \brief Contains various Mesh and MeshRefinement functions associated with
//! load balancing when MPI is used, both for uniform grids and with SMR/AMR.

#include <iostream>
#include <limits> // numeric_limits<>
#include <algorithm> // max
#include <utility> // make_pair
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "particles/particles.hpp" 

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn void Mesh::LoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb)
//! \brief Calculate distribution of MeshBlocks across ranks based on input cost list
//! input: clist = cost of each MB (array of length nmbtotal)
//!        nb = number of MeshBlocks
//! output: rlist = rank to which each MB is assigned (array of length nmbtotal)
//!         slist = starting grid ID (gid) for MB on each rank (array of length nrank)
//!         nlist = number of MBs on each rank (array of length nrank)
//! With multiple ranks in MPI, this function is needed even on a uniform mesh and not
//! just for SMR/AMR, which is why it is part of the Mesh and not MeshRefinement class.

void Mesh::LoadBalance(float *clist, int *rlist, int *slist, int *nlist, int nb) {
  float min_cost = std::numeric_limits<float>::max();
  float max_cost = 0.0, totalcost = 0.0;
  // find min/max and total cost in clist
  for (int i=0; i<nb; i++) {
    totalcost += clist[i];
    min_cost = std::min(min_cost,clist[i]);
    max_cost = std::max(max_cost,clist[i]);
  }

  int j = (global_variable::nranks) - 1;
  float targetcost = totalcost/global_variable::nranks;
  float mycost = 0.0;
  // create rank list from the end: the master MPI rank should have less load
  for (int i=nb-1; i>=0; i--) {
    if (targetcost == 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "There is at least one process which has no MeshBlock"
                << std::endl << "Decrease the number of processes or use smaller "
                << "MeshBlocks." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    mycost += clist[i];
    rlist[i] = j;
    if (mycost >= targetcost && j>0) {
      j--;
      totalcost -= mycost;
      mycost = 0.0;
      targetcost = totalcost/(j+1);
    }
  }
  slist[0] = 0;
  j = 0;
  for (int i=1; i<nb; i++) { // make the list of nbstart and nblocks
    if (rlist[i] != rlist[i-1]) {
      nlist[j] = i-slist[j];
      slist[++j] = i;
    }
  }
  nlist[j] = nb-slist[j];

#if MPI_PARALLEL_ENABLED
  if (nb % global_variable::nranks != 0
     && !adaptive && max_cost == min_cost && global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Number of MeshBlocks cannot be divided evenly by number of MPI ranks. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::InitRecvAMR()
//! \brief Allocates and initializes receive buffers, and posts non-blocking receives,
//! for communicating MeshBlocks during load balancing. Equivalent to some of the work
//! done inside MPI_PARALLEL block in the Mesh::RedistributeAndRefineMeshBlocks()
//! function in amr_loadbalance.cpp in Athena++

void MeshRefinement::InitRecvAMR(int nleaf) {
#if MPI_PARALLEL_ENABLED

  InitPartRecv();

  // Step 1. (InitRecvAMR)
  // loop over new MBs on this rank, count number of MeshBlocks received by this rank
  nmb_recv = 0;
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {          // old MB was de-refined
      for (int l=0; l<nleaf; l++) {
        // recv whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) ||
            (pmy_mesh->rank_eachmb[oldm+l] != global_variable::my_rank)) {
          nmb_recv++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {  // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        nmb_recv++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        nmb_recv++;
      }
    }
  }
  if (nmb_recv == 0) return;  // nothing to do

  // allocate array of recv buffers
  //Kokkos::realloc(recvbuf, nmb_recv);
  if ((int)recvbuf.h_view.extent(0) < nmb_recv) {
    Kokkos::realloc(recvbuf, nmb_recv);
  }
  recv_req = new MPI_Request[nmb_recv];
  for (int n=0; n<nmb_recv; ++n) {
    recv_req[n] = MPI_REQUEST_NULL;
  }

  // count number of cell- and face-centered variables communicated depending on physics
  int ncc_tosend=0, nfc_tosend=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->phydro->nhydro +
                   pmy_mesh->pmb_pack->phydro->nscalars);
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->pmhd->nmhd  +
                   pmy_mesh->pmb_pack->pmhd->nscalars);
    nfc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->pz4c != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->pz4c->nz4c);
  }

  // Step 2. (InitRecvAMR)
  // loop over new MBs on this rank, initialize recv buffers
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  auto &nx1 = indcs.nx1, &nx2 = indcs.nx2, &nx3 = indcs.nx3;
  auto &ng = indcs.ng;
  int il = cis - ng, iu = cie + ng;
  int jl = cjs,      ju = cje;
  int kl = cks,      ku = cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }

  int rb_idx = 0;   // recv buffer index
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {        // old MB was de-refined
      for (int l=0; l<nleaf; l++) {
        // recv whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) ||
            (pmy_mesh->rank_eachmb[oldm+l] != global_variable::my_rank)) {
          LogicalLocation &lloc = pmy_mesh->lloc_eachmb[oldm+l];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          recvbuf.h_view(rb_idx).bis = cis + ox1*cnx1;
          recvbuf.h_view(rb_idx).bie = cie + ox1*cnx1;
          recvbuf.h_view(rb_idx).bjs = cjs + ox2*cnx2;
          recvbuf.h_view(rb_idx).bje = cje + ox2*cnx2;
          recvbuf.h_view(rb_idx).bks = cks + ox3*cnx3;
          recvbuf.h_view(rb_idx).bke = cke + ox3*cnx3;
          recvbuf.h_view(rb_idx).cntcc = cnx1*cnx2*cnx3;
          recvbuf.h_view(rb_idx).cntfc = 3*cnx1*cnx2*cnx3 + cnx2*cnx3 +
                                          cnx1*cnx3 + cnx1*cnx2;
          recvbuf.h_view(rb_idx).cnt   = ncc_tosend*(recvbuf.h_view(rb_idx).cntcc) +
                                         nfc_tosend*(recvbuf.h_view(rb_idx).cntfc);
          recvbuf.h_view(rb_idx).lid   = newm - nmbs;
          recvbuf.h_view(rb_idx).use_coarse = false;
          if (rb_idx > 0) {
            recvbuf.h_view(rb_idx).offset = recvbuf.h_view((rb_idx-1)).offset +
                                             recvbuf.h_view((rb_idx-1)).cnt;
          } else {
            recvbuf.h_view(rb_idx).offset = 0;
          }
          rb_idx++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {   // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        recvbuf.h_view(rb_idx).bis = is;
        recvbuf.h_view(rb_idx).bie = ie;
        recvbuf.h_view(rb_idx).bjs = js;
        recvbuf.h_view(rb_idx).bje = je;
        recvbuf.h_view(rb_idx).bks = ks;
        recvbuf.h_view(rb_idx).bke = ke;
        recvbuf.h_view(rb_idx).cntcc = nx1*nx2*nx3;
        recvbuf.h_view(rb_idx).cntfc = 3*nx1*nx2*nx3 + nx2*nx3 + nx1*nx3 + nx1*nx2;
        recvbuf.h_view(rb_idx).cnt = ncc_tosend*(recvbuf.h_view(rb_idx).cntcc) +
                                     nfc_tosend*(recvbuf.h_view(rb_idx).cntfc);
        recvbuf.h_view(rb_idx).lid = newm - nmbs;
        recvbuf.h_view(rb_idx).use_coarse = false;
        if (rb_idx > 0) {
          recvbuf.h_view(rb_idx).offset = recvbuf.h_view((rb_idx-1)).offset +
                                          recvbuf.h_view((rb_idx-1)).cnt;
        } else {
          recvbuf.h_view(rb_idx).offset = 0;
        }
        rb_idx++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        recvbuf.h_view(rb_idx).bis = il; // note il:iu etc. includes ghost zones
        recvbuf.h_view(rb_idx).bie = iu;
        recvbuf.h_view(rb_idx).bjs = jl;
        recvbuf.h_view(rb_idx).bje = ju;
        recvbuf.h_view(rb_idx).bks = kl;
        recvbuf.h_view(rb_idx).bke = ku;
        recvbuf.h_view(rb_idx).cntcc = (iu-il+1)*(ju-jl+1)*(ku-kl+1);
        recvbuf.h_view(rb_idx).cntfc = (iu-il+2)*(ju-jl+1)*(ku-kl+1) +
             (iu-il+1)*(ju-jl+2)*(ku-kl+1) + (iu-il+1)*(ju-jl+1)*(ku-kl+2);
        recvbuf.h_view(rb_idx).cnt = ncc_tosend*(recvbuf.h_view(rb_idx).cntcc) +
                                     nfc_tosend*(recvbuf.h_view(rb_idx).cntfc);
        recvbuf.h_view(rb_idx).lid = newm - nmbs;
        recvbuf.h_view(rb_idx).use_coarse = true;
        if (rb_idx > 0) {
          recvbuf.h_view(rb_idx).offset = recvbuf.h_view((rb_idx-1)).offset +
                                          recvbuf.h_view((rb_idx-1)).cnt;
        } else {
          recvbuf.h_view(rb_idx).offset = 0;
        }
        rb_idx++;
      }
    }
  }
  // Sync dual array, reallocate receive data array
  recvbuf.template modify<HostMemSpace>();
  recvbuf.template sync<DevExeSpace>();
  {
    int ndata = recvbuf.h_view((nmb_recv-1)).offset + recvbuf.h_view((nmb_recv-1)).cnt;
    //Kokkos::realloc(recv_data, ndata);
    if ((int)recv_data.extent(0) < ndata) {
      Kokkos::realloc(recv_data, ndata);
    }
  }

  // Step 3. (InitRecvAMR)
  // loop over new MBs on this rank, post non-blocking recvs
  // Receive requests will only be accessed on host, so no need to sync after this step.
  rb_idx = 0;   // recv buffer index
  bool no_errors=true;
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {        // old MB was de-refined
      for (int l=0; l<nleaf; l++) {
        // recv whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) ||
            (pmy_mesh->rank_eachmb[oldm+l] != global_variable::my_rank)) {
          LogicalLocation &lloc = pmy_mesh->lloc_eachmb[oldm+l];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          int vs = recvbuf.h_view(rb_idx).offset;
          int ve = vs + recvbuf.h_view(rb_idx).cnt;
          auto pdata = Kokkos::subview(recv_data, std::make_pair(vs,ve));
          // create tag using local ID of *receiving* MeshBlock, post receive
          int tag = CreateAMR_MPI_Tag(newm-nmbs, ox1, ox2, ox3);
          // post non-blocking receive
          int ierr = MPI_Irecv(pdata.data(), recvbuf.h_view(rb_idx).cnt,
                     MPI_ATHENA_REAL, pmy_mesh->rank_eachmb[oldm+l], tag, amr_comm,
                     &(recv_req[rb_idx]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          rb_idx++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {   // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        int vs = recvbuf.h_view(rb_idx).offset;
        int ve = vs + recvbuf.h_view(rb_idx).cnt;
        auto pdata = Kokkos::subview(recv_data, std::make_pair(vs,ve));
        // create tag using local ID of *receiving* MeshBlock, post receive
        int tag = CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0);
        // post non-blocking receive
        int ierr = MPI_Irecv(pdata.data(), recvbuf.h_view(rb_idx).cnt, MPI_ATHENA_REAL,
                   pmy_mesh->rank_eachmb[oldm], tag, amr_comm,
                   &(recv_req[rb_idx]));
        if (ierr != MPI_SUCCESS) {no_errors=false;}
        rb_idx++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        int vs = recvbuf.h_view(rb_idx).offset;
        int ve = vs + recvbuf.h_view(rb_idx).cnt;
        auto pdata = Kokkos::subview(recv_data, std::make_pair(vs,ve));
        // create tag using local ID of *receiving* MeshBlock, post receive
        int tag = CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0);
        // post non-blocking receive
        int ierr = MPI_Irecv(pdata.data(), recvbuf.h_view(rb_idx).cnt, MPI_ATHENA_REAL,
                   pmy_mesh->rank_eachmb[oldm], tag, amr_comm,
                   &(recv_req[rb_idx]));
        if (ierr != MPI_SUCCESS) {no_errors=false;}
        rb_idx++;
      }
    }
  }

  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives with AMR" << std::endl;
    std::exit(EXIT_FAILURE);
  }

#endif

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAndSendAMR()
//! \brief Allocates and initializes send buffers for communicating MeshBlocks during
//! load balancing, calls function to pack data into buffers, and posts non-blocking sends
//! Equivalent to some of the work done inside MPI_PARALLEL block in the
//! Mesh::RedistributeAndRefineMeshBlocks() function in amr_loadbalance.cpp in Athena++

void MeshRefinement::PackAndSendAMR(int nleaf) {
#if MPI_PARALLEL_ENABLED
  if (pmy_mesh->pmb_pack->ppart != nullptr) {
    PackAMRBuffersParticles();
  }

  // Step 1. (PackAndSendAMR)
  // loop over old MBs on this rank, count number of MeshBlocks to send on this rank
  nmb_send = 0;
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {          // old MB was refined
      for (int l=0; l<nleaf; l++) {
        // send if refined MB changes rank, or if any leaf on different rank than root
        if ((new_rank_eachmb[newm] != global_variable::my_rank) ||
            (new_rank_eachmb[newm + l] != global_variable::my_rank)) {
          nmb_send++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {  // old MB on same level
      if (new_rank_eachmb[newm] != global_variable::my_rank) {
        nmb_send++;
      }
    } else {                                        // old MB was de-refined
      // send if root MB changes rank, or if any leaf on different rank than root
      if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
          (new_rank_eachmb[newm] != global_variable::my_rank)) {
        nmb_send++;
      }
    }
  }

  if (nmb_send == 0) return;  // nothing to do

  // allocate array of send buffers
  //Kokkos::realloc(sendbuf, nmb_send);
  if ((int)sendbuf.h_view.extent(0) < nmb_send) {
    Kokkos::realloc(sendbuf, nmb_send);
  }
  send_req = new MPI_Request[nmb_send];
  for (int n=0; n<nmb_send; ++n) {
    send_req[n] = MPI_REQUEST_NULL;
  }

  // count number of cell- and face-centered variables communicated depending on physics
  int ncc_tosend=0, nfc_tosend=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->phydro->nhydro  +
                   pmy_mesh->pmb_pack->phydro->nscalars);
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->pmhd->nmhd  +
                   pmy_mesh->pmb_pack->pmhd->nscalars);
    nfc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->pz4c != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->pz4c->nz4c);
  }

  // Step 2. (PackAndSendAMR)
  // loop over old MBs on this rank, initialize send buffers
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  auto &nx1 = indcs.nx1, &nx2 = indcs.nx2, &nx3 = indcs.nx3;
  auto &ng = indcs.ng;
  int il = cis - ng, iu = cie + ng;
  int jl = cjs,      ju = cje;
  int kl = cks,      ku = cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }

  int sb_idx = 0;   // send buffer index
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {  // old MB was refined
      for (int l=0; l<nleaf; l++) {
        // send if refined MB changes rank, or if any leaf on different rank than root
        if ((new_rank_eachmb[newm] != global_variable::my_rank) ||
            (new_rank_eachmb[newm + l] != global_variable::my_rank)) {
          LogicalLocation &lloc = new_lloc_eachmb[newm+l];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          sendbuf.h_view(sb_idx).bis = il + ox1*cnx1;  // il:iu etc. includes ghost zones
          sendbuf.h_view(sb_idx).bie = iu + ox1*cnx1;
          sendbuf.h_view(sb_idx).bjs = jl + ox2*cnx2;
          sendbuf.h_view(sb_idx).bje = ju + ox2*cnx2;
          sendbuf.h_view(sb_idx).bks = kl + ox3*cnx3;
          sendbuf.h_view(sb_idx).bke = ku + ox3*cnx3;
          sendbuf.h_view(sb_idx).cntcc = (iu-il+1)*(ju-jl+1)*(ku-kl+1);
          sendbuf.h_view(sb_idx).cntfc = (iu-il+2)*(ju-jl+1)*(ku-kl+1) +
               (iu-il+1)*(ju-jl+2)*(ku-kl+1) + (iu-il+1)*(ju-jl+1)*(ku-kl+2);
          sendbuf.h_view(sb_idx).cnt   = ncc_tosend*(sendbuf.h_view(sb_idx).cntcc) +
                                         nfc_tosend*(sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid   = oldm - ombs;
          sendbuf.h_view(sb_idx).use_coarse = false;
          if (sb_idx > 0) {
            sendbuf.h_view(sb_idx).offset = sendbuf.h_view((sb_idx-1)).offset +
                                            sendbuf.h_view((sb_idx-1)).cnt;
          } else {
            sendbuf.h_view(sb_idx).offset = 0;
          }
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (old_lloc.level == new_lloc.level) { // old MB on same level
        if (new_rank_eachmb[newm] != global_variable::my_rank) {
          sendbuf.h_view(sb_idx).bis = is;
          sendbuf.h_view(sb_idx).bie = ie;
          sendbuf.h_view(sb_idx).bjs = js;
          sendbuf.h_view(sb_idx).bje = je;
          sendbuf.h_view(sb_idx).bks = ks;
          sendbuf.h_view(sb_idx).bke = ke;
          sendbuf.h_view(sb_idx).cntcc = nx1*nx2*nx3;
          sendbuf.h_view(sb_idx).cntfc = 3*nx1*nx2*nx3 + nx2*nx3 + nx1*nx3 + nx1*nx2;
          sendbuf.h_view(sb_idx).cnt = ncc_tosend*(sendbuf.h_view(sb_idx).cntcc) +
                                       nfc_tosend*(sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid = oldm - ombs;
          sendbuf.h_view(sb_idx).use_coarse = false;
          if (sb_idx > 0) {
            sendbuf.h_view(sb_idx).offset = sendbuf.h_view((sb_idx-1)).offset +
                                            sendbuf.h_view((sb_idx-1)).cnt;
          } else {
            sendbuf.h_view(sb_idx).offset = 0;
          }
          sb_idx++;
        }
      } else {                                // old MB was de-refined
        // send whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
            (new_rank_eachmb[newm] != global_variable::my_rank)) {
          sendbuf.h_view(sb_idx).bis = cis;
          sendbuf.h_view(sb_idx).bie = cie;
          sendbuf.h_view(sb_idx).bjs = cjs;
          sendbuf.h_view(sb_idx).bje = cje;
          sendbuf.h_view(sb_idx).bks = cks;
          sendbuf.h_view(sb_idx).bke = cke;
          sendbuf.h_view(sb_idx).cntcc = cnx1*cnx2*cnx3;
          sendbuf.h_view(sb_idx).cntfc = 3*cnx1*cnx2*cnx3 + cnx2*cnx3 + cnx1*cnx3
                                          + cnx1*cnx2;
          sendbuf.h_view(sb_idx).use_coarse = true;
          sendbuf.h_view(sb_idx).cnt = ncc_tosend*(sendbuf.h_view(sb_idx).cntcc) +
                                       nfc_tosend*(sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid = oldm - ombs;
          if (sb_idx > 0) {
            sendbuf.h_view(sb_idx).offset = sendbuf.h_view((sb_idx-1)).offset +
                                            sendbuf.h_view((sb_idx-1)).cnt;
          } else {
            sendbuf.h_view(sb_idx).offset = 0;
          }
          sb_idx++;
        }
      }
    }
  }
  // Sync dual array, reallocate send data array
  sendbuf.template modify<HostMemSpace>();
  sendbuf.template sync<DevExeSpace>();
  {
    int ndata = sendbuf.h_view((nmb_send-1)).offset + sendbuf.h_view((nmb_send-1)).cnt;
    //Kokkos::realloc(send_data, ndata);
    if ((int)send_data.extent(0) < ndata) {
      Kokkos::realloc(send_data, ndata);
    }
  }

  // Step 3. (PackAndSendAMR)
  // Pack data into send buffers in parallel
  hydro::Hydro* phydro = pmy_mesh->pmb_pack->phydro;
  mhd::MHD* pmhd = pmy_mesh->pmb_pack->pmhd;
  z4c::Z4c* pz4c = pmy_mesh->pmb_pack->pz4c;

  int ncc_sent = 0, nfc_sent = 0;
  if (phydro != nullptr) {
    PackAMRBuffersCC(phydro->u0, phydro->coarse_u0, ncc_sent, nfc_sent);
    ncc_sent += phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    PackAMRBuffersCC(pmhd->u0, pmhd->coarse_u0, ncc_sent, nfc_sent);
    ncc_sent += pmhd->nmhd + pmhd->nscalars;
    PackAMRBuffersFC(pmhd->b0, pmhd->coarse_b0, ncc_sent, nfc_sent);
    nfc_sent += 1;
  }
  if (pz4c != nullptr) {
    PackAMRBuffersCC(pz4c->u0, pz4c->coarse_u0, ncc_sent, nfc_sent);
    ncc_sent += pz4c->nz4c;
  }

  // Step 4. (PackAndSendAMR)
  // loop over old MBs on this rank, send data using MPI non-blocking sends
  // Send requests will only be accessed on host, so no need to sync after this step.
  Kokkos::fence();
  bool no_errors=true;
  sb_idx = 0;     // send buffer index
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {      // old MB was refined
      for (int l=0; l<nleaf; l++) {
        // send if refined MB changes rank, or if any leaf on different rank than root
        if ((new_rank_eachmb[newm] != global_variable::my_rank) ||
            (new_rank_eachmb[newm + l] != global_variable::my_rank)) {
          int vs = sendbuf.h_view(sb_idx).offset;
          int ve = vs + sendbuf.h_view(sb_idx).cnt;
          auto pdata = Kokkos::subview(send_data, std::make_pair(vs,ve));
          // create tag using local ID of *receiving* MeshBlock
          int lid = (newm + l) - new_gids_eachrank[new_rank_eachmb[newm+l]];
          int tag = CreateAMR_MPI_Tag(lid, 0, 0, 0);
          // post non-blocking send
          int ierr = MPI_Isend(pdata.data(), sendbuf.h_view(sb_idx).cnt, MPI_ATHENA_REAL,
                     new_rank_eachmb[newm+l], tag, amr_comm,
                     &(send_req[sb_idx]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (old_lloc.level == new_lloc.level) {   // old MB at same level
        if (new_rank_eachmb[newm] != global_variable::my_rank) {
          int vs = sendbuf.h_view(sb_idx).offset;
          int ve = vs + sendbuf.h_view(sb_idx).cnt;
          auto pdata = Kokkos::subview(send_data, std::make_pair(vs,ve));
          // create tag using local ID of *receiving* MeshBlock
          int lid = newm - new_gids_eachrank[new_rank_eachmb[newm]];
          int tag = CreateAMR_MPI_Tag(lid, 0, 0, 0);
          // post non-blocking send
          int ierr = MPI_Isend(pdata.data(), sendbuf.h_view(sb_idx).cnt, MPI_ATHENA_REAL,
                     new_rank_eachmb[newm], tag, amr_comm,
                     &(send_req[sb_idx]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          sb_idx++;
        }
      } else {                                  // old MB was de-refined
        // send whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
            (new_rank_eachmb[newm] != global_variable::my_rank)) {
          int vs = sendbuf.h_view(sb_idx).offset;
          int ve = vs + sendbuf.h_view(sb_idx).cnt;
          auto pdata = Kokkos::subview(send_data, std::make_pair(vs,ve));
          // create tag using local ID of *receiving* MeshBlock
          int ox1 = ((old_lloc.lx1 & 1) == 1);
          int ox2 = ((old_lloc.lx2 & 1) == 1);
          int ox3 = ((old_lloc.lx3 & 1) == 1);
          int lid = newm - new_gids_eachrank[new_rank_eachmb[newm]];
          int tag = CreateAMR_MPI_Tag(lid, ox1, ox2, ox3);
          // post non-blocking send
          int ierr = MPI_Isend(pdata.data(), sendbuf.h_view(sb_idx).cnt, MPI_ATHENA_REAL,
                     new_rank_eachmb[newm], tag, amr_comm,
                     &(send_req[sb_idx]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          sb_idx++;
        }
      }
    }
  }

  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking receives with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAMRBuffersCC()
//! \brief Packs cell-centered data into AMR communication buffers for all MBs being sent
//! Equivalent to PrepareSendSameLevel(), PrepareSendCoarseToFineAMR(), and
//! PrepareSendFineToCoarseAMR() functions in amr_loadbalance.cpp

void MeshRefinement::PackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                                      int ncc, int nfc) {
#if MPI_PARALLEL_ENABLED
  auto &sbuf = sendbuf;
  auto &sdata = send_data;
  // Outer loop over (# of MeshBlocks sent)*(# of variables)
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nnv = nmb_send*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int n = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - n*nvar);

    const int il = sbuf.d_view(n).bis;
    const int jl = sbuf.d_view(n).bjs;
    const int kl = sbuf.d_view(n).bks;
    const int ni = sbuf.d_view(n).bie - il + 1;
    const int nj = sbuf.d_view(n).bje - jl + 1;
    const int nk = sbuf.d_view(n).bke - kl + 1;
    const int nkji = nk*nj*ni;
    const int nji  = nj*ni;
    const int m  = sbuf.d_view(n).lid;
    const int offset = sbuf.d_view(n).offset +
                       (ncc*sbuf.d_view(n).cntcc + nfc*sbuf.d_view(n).cntfc);

    // Middle loop over k,j,i
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + il;
      k += kl;
      j += jl;
      if (sbuf.d_view(n).use_coarse) {
        // if de-refinement, load data from coarse_a
        sdata(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = ca(m,v,k,j,i);
      } else {
        // if refinement or same level, load data from a
        sdata(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = a(m,v,k,j,i);
      }
    });
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAMRBuffersFC()
//! \brief Packs face-centered data into AMR communication buffers for all MBs being sent

void MeshRefinement::PackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb,
                                      int ncc, int nfc) {
#if MPI_PARALLEL_ENABLED
  auto &sbuf = sendbuf;
  auto &sdata = send_data;
  // Outer loop over (# of MeshBlocks sent)*(3 compnts of field)
  int nn = 3*nmb_send;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nn, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int n = (tmember.league_rank())/3;
    const int v = (tmember.league_rank() - 3*n);

    const int il = sbuf.d_view(n).bis;
    const int jl = sbuf.d_view(n).bjs;
    const int kl = sbuf.d_view(n).bks;
    const int m  = sbuf.d_view(n).lid;
    const int nicc = sbuf.d_view(n).bie - il + 1;
    const int njcc = sbuf.d_view(n).bje - jl + 1;
    const int nkcc = sbuf.d_view(n).bke - kl + 1;

    // pack x1 component
    if (v==0) {
      const int offset = sbuf.d_view(n).offset +
                         (ncc*sbuf.d_view(n).cntcc + nfc*sbuf.d_view(n).cntfc);
      const int ni = nicc + 1;  // add b.x1f at (ie+1)
      const int nj = njcc;
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x1f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x1f(m,k,j,i);
        }
      });

    // pack x2 component
    } else if (v==1) {
      const int offset = sbuf.d_view(n).offset +
                         (ncc*sbuf.d_view(n).cntcc + nfc*sbuf.d_view(n).cntfc) +
                         (nicc+1)*njcc*nkcc;
      const int ni = nicc;
      const int nj = njcc + 1;  // add b.x2f at (je+1)
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x2f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x2f(m,k,j,i);
        }
      });

    // pack x3 component
    } else {
      const int offset = sbuf.d_view(n).offset +
                         (ncc*sbuf.d_view(n).cntcc + nfc*sbuf.d_view(n).cntfc) +
                         (nicc+1)*njcc*nkcc + nicc*(njcc+1)*nkcc;
      const int ni = nicc;
      const int nj = njcc;
      const int nk = nkcc + 1;  // add b.x3f at (ke+1)
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x3f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x3f(m,k,j,i);
        }
      });
    }
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearRecvAndUnpackAMR()
//! \brief Checks non-blocking receives have finished, calls function to unpack buffers,
//! deletes receive buffers. Equivalent to some of the work done inside MPI_PARALLEL block
//! in the Mesh::RedistributeAndRefineMeshBlocks() function in amr_loadbalance.cpp

void MeshRefinement::ClearRecvAndUnpackAMR() {
#if MPI_PARALLEL_ENABLED
  if (pmy_mesh->pmb_pack->ppart != nullptr) {
    // Wait for particle receives to finish
    bool no_errors=true;
    for (int n=0; n<prtcl_nrecvs; ++n) {
      int ierr = MPI_Wait(&(prtcl_rrecv_req[n]), MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) no_errors=false;
      ierr = MPI_Wait(&(prtcl_irecv_req[n]), MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) no_errors = false;
    }
    // Quit if MPI error detected
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in particle communication with AMR"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (nmb_recv != 0) {
    // Wait for all receives to finish
    bool no_errors=true;
    for (int n=0; n<nmb_recv; ++n) {
      int ierr = MPI_Wait(&(recv_req[n]), MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) {no_errors=false;}
    }
    // Quit if MPI error detected
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in posting non-blocking receives with AMR"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    delete [] recv_req;
  }

  if (pmy_mesh->pmb_pack->ppart != nullptr) {
    UnpackAMRBuffersParticles();
  }

  if (nmb_recv == 0) return;

  // Unpack data
  hydro::Hydro* phydro = pmy_mesh->pmb_pack->phydro;
  mhd::MHD* pmhd = pmy_mesh->pmb_pack->pmhd;
  z4c::Z4c* pz4c = pmy_mesh->pmb_pack->pz4c;

  int ncc_recv=0, nfc_recv=0;

  if (phydro != nullptr) {
    UnpackAMRBuffersCC(phydro->u0, phydro->coarse_u0, ncc_recv, nfc_recv);
    ncc_recv += phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    UnpackAMRBuffersCC(pmhd->u0, pmhd->coarse_u0, ncc_recv, nfc_recv);
    ncc_recv += pmhd->nmhd + pmhd->nscalars;
    UnpackAMRBuffersFC(pmhd->b0, pmhd->coarse_b0, ncc_recv, nfc_recv);
    nfc_recv += 1;
  }
  if (pz4c != nullptr) {
    UnpackAMRBuffersCC(pz4c->u0, pz4c->coarse_u0, ncc_recv, nfc_recv);
    ncc_recv += pz4c->nz4c;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UnpackAMRBuffersCC()
//! \brief Unpacks face-centered data from AMR communication buffers into appropriate
//! coarse or fine arrays for all MBs received during load balancing.
//! Equivalent to FinishRecvSameLevel(), FinishRecvCoarseToFineAMR(), and
//! FinishRecvFineToCoarseAMR() functions in amr_loadbalance.cpp

void MeshRefinement::UnpackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                                        int ncc, int nfc) {
#if MPI_PARALLEL_ENABLED
  auto &rbuf = recvbuf;
  auto &rdata = recv_data;
  // Outer loop over (# of MeshBlocks recv)*(# of variables)
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nnv = nmb_recv*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int n = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - n*nvar);

    const int il = rbuf.d_view(n).bis;
    const int jl = rbuf.d_view(n).bjs;
    const int kl = rbuf.d_view(n).bks;
    const int ni = rbuf.d_view(n).bie - il + 1;
    const int nj = rbuf.d_view(n).bje - jl + 1;
    const int nk = rbuf.d_view(n).bke - kl + 1;
    const int nkji = nk*nj*ni;
    const int nji  = nj*ni;
    const int m  = rbuf.d_view(n).lid;
    const int offset = rbuf.d_view(n).offset +
                       (ncc*rbuf.d_view(n).cntcc + nfc*rbuf.d_view(n).cntfc);

    // Middle loop over k,j,i
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + il;
      k += kl;
      j += jl;
      if (rbuf.d_view(n).use_coarse) {
        // if refinement, load data into coarse_a
        ca(m,v,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
      } else {
        // if de-refinement or same level, load data into a
        a(m,v,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
      }
    });
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UnpackAMRBuffersFC()
//! \brief Unpacks face-centered data from AMR communication buffers into appropriate
//! coarse or fine arrays for all MBs received during load balancing.

void MeshRefinement::UnpackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb,
                                        int ncc, int nfc) {
#if MPI_PARALLEL_ENABLED
  auto &rbuf = recvbuf;
  auto &rdata = recv_data;
  // Outer loop over (# of MeshBlocks recv)*(3 compnts of field)
  int nnv = 3*nmb_recv;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int n = (tmember.league_rank())/3;
    const int v = (tmember.league_rank() - n*3);

    const int il = rbuf.d_view(n).bis;
    const int jl = rbuf.d_view(n).bjs;
    const int kl = rbuf.d_view(n).bks;
    const int m  = rbuf.d_view(n).lid;
    const int nicc = rbuf.d_view(n).bie - il + 1;
    const int njcc = rbuf.d_view(n).bje - jl + 1;
    const int nkcc = rbuf.d_view(n).bke - kl + 1;

    // unpack x1 component
    if (v==0) {
      const int offset = rbuf.d_view(n).offset +
                         (ncc*rbuf.d_view(n).cntcc + nfc*rbuf.d_view(n).cntfc);
      const int ni = nicc + 1;  // add b.x1f at (ie+1)
      const int nj = njcc;
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x1f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x1f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        }
      });

    // unpack x2 component
    } else if (v==1) {
      const int offset = rbuf.d_view(n).offset +
                         (ncc*rbuf.d_view(n).cntcc + nfc*rbuf.d_view(n).cntfc) +
                         (nicc+1)*njcc*nkcc;
      const int ni = nicc;
      const int nj = njcc + 1;  // add b.x2f at (je+1)
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x2f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x2f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        }
      });

    // unpack x3 component
    } else {
      const int offset = rbuf.d_view(n).offset +
                         (ncc*rbuf.d_view(n).cntcc + nfc*rbuf.d_view(n).cntfc) +
                         (nicc+1)*njcc*nkcc + nicc*(njcc+1)*nkcc;
      const int ni = nicc;
      const int nj = njcc;
      const int nk = nkcc + 1;  // add b.x3f at (ke+1)
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x3f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x3f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        }
      });
    }
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearSendAMR()
//! \brief Checks all non-blocking sends completed, deletes send buffers.

void MeshRefinement::ClearSendAMR() {
#if MPI_PARALLEL_ENABLED
  bool no_errors = true; 
  if (pmy_mesh->pmb_pack->ppart != nullptr) {
    for (int n = 0; n < prtcl_nsends; ++n) {
      if (prtcl_rsend_req[n] != MPI_REQUEST_NULL) {
        int ierr = MPI_Wait(&(prtcl_rsend_req[n]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
      if (prtcl_isend_req[n] != MPI_REQUEST_NULL) {
        int ierr = MPI_Wait(&(prtcl_isend_req[n]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
    // Clear particle request vectors
    prtcl_rsend_req.clear();
    prtcl_isend_req.clear();
    prtcl_nsends = 0;
  }

  if (nmb_send == 0) return;

  for (int n=0; n<nmb_send; ++n) {
    int ierr = MPI_Wait(&(send_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in clearing non-blocking sends with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  delete [] send_req;
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAMRBuffersParticles()
//! \brief Packs particle data into AMR communication buffers for all MBs being sent

void MeshRefinement::PackAMRBuffersParticles() {
#if MPI_PARALLEL_ENABLED
  // Figure out how many particles will be sent from this ranks
  nprtcl_send = 0;
  for (int n=0; n<prtcl_nsends; ++n) {
    nprtcl_send += prtcl_sends_thisrank[n].nprtcls;
  }
  if (nprtcl_send == 0) return;

  auto *ppart = pmy_mesh->pmb_pack->ppart;
  int nrdata = ppart->nrdata;
  int nidata = ppart->nidata;
  auto &pr = ppart->prtcl_rdata;
  auto &pi = ppart->prtcl_idata;
  auto &rsendbuf = prtcl_rsendbuf;
  auto &isendbuf = prtcl_isendbuf;
  auto &sendlist_ = prtcl_sendlist;
  bool no_errors = true;

  // Allocate send buffer
  Kokkos::realloc(prtcl_rsendbuf, nrdata*nprtcl_send);
  Kokkos::realloc(prtcl_isendbuf, nidata*nprtcl_send);

  // sendlist on device is already sorted by destrank in CountSendAndRecvs()
  // Use sendlist on device to load particles into send buffer ordered by dest_rank
  par_for("amr_ppack",DevExeSpace(),0,(nprtcl_send-1), KOKKOS_LAMBDA(const int n) {
    int p = sendlist_.d_view(n).prtcl_indx;
    for (int i=0; i<nidata; ++i) {
      isendbuf(nidata*n + i) = pi(i,p);
    }
    for (int i=0; i<nrdata; ++i) {
      rsendbuf(nrdata*n + i) = pr(i,p);
    }
  });

  // Ensure all packing is complete before MPI sends
  Kokkos::fence();

  // Initialize MPI request vectors
  prtcl_rsend_req.clear();
  prtcl_isend_req.clear();
  for (int n = 0; n < prtcl_nsends; ++n) {
    prtcl_rsend_req.emplace_back(MPI_REQUEST_NULL);
    prtcl_isend_req.emplace_back(MPI_REQUEST_NULL);
  }

  // Send Real particle data
  int data_start = 0;
  for (int n = 0; n < prtcl_nsends; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = nrdata*(prtcl_sends_thisrank[n].nprtcls);
    int data_end = data_start + data_size;
    auto send_ptr = Kokkos::subview(prtcl_rsendbuf,std::make_pair(data_start,data_end));
    int drank = prtcl_sends_thisrank[n].recvrank;
    int tag = 0; // 0 for Reals, 1 for ints

    // Post non-blocking sends
    int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                         par_comm, &(prtcl_rsend_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    data_start += data_size;
  }

  // Send int particle data
  data_start = 0;
  for (int n = 0; n < prtcl_nsends; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = nidata*(prtcl_sends_thisrank[n].nprtcls);
    int data_end = data_start + data_size;
    auto send_ptr = Kokkos::subview(prtcl_isendbuf,std::make_pair(data_start,data_end));
    int drank = prtcl_sends_thisrank[n].recvrank;
    int tag = 1; // 0 for Reals, 1 for ints

    // Post non-blocking sends
    int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_INT, drank, tag,
                         par_comm, &(prtcl_isend_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    data_start += data_size;
  }

  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UnpackAMRBuffersParticles()
//! \brief Unpacks particle data from AMR communication buffers

void MeshRefinement::UnpackAMRBuffersParticles() {
#if MPI_PARALLEL_ENABLED
  auto *ppart = pmy_mesh->pmb_pack->ppart;

  namespace KE = Kokkos::Experimental;

  int &npart = ppart->nprtcl_thispack;
  int new_npart = npart + (nprtcl_recv - nprtcl_send);
  int nremain = 0;

  // Sort particle sendlist by index for hole-filling (already done in CountParticlesPerMeshBlock)
  std::sort(KE::begin(prtcl_sendlist.h_view), KE::end(prtcl_sendlist.h_view), SortByIndex);
  // sync sendlist host array with device.  This results in sorted array on device
  prtcl_sendlist.template modify<HostMemSpace>();
  prtcl_sendlist.template sync<DevExeSpace>();

  // increase size of particle arrays if needed
  if (nprtcl_recv > nprtcl_send) {
    Kokkos::resize(ppart->prtcl_idata, ppart->nidata, new_npart);
    Kokkos::resize(ppart->prtcl_rdata, ppart->nrdata, new_npart);
  }

  // unpack particles into positions of sent particles or at end of arrays
  if (nprtcl_recv > 0) {
    int nrdata = ppart->nrdata;
    int nidata = ppart->nidata;
    auto &pr = ppart->prtcl_rdata;
    auto &pi = ppart->prtcl_idata;
    auto &rrecvbuf = prtcl_rrecvbuf;
    auto &irecvbuf = prtcl_irecvbuf;
    int nprtcl_send_ = nprtcl_send;
    auto &sendlist_ = prtcl_sendlist;
    auto &gid_start = pmy_mesh->pmb_pack->gids;

    par_for("amr_punpack",DevExeSpace(),0,(nprtcl_recv-1), KOKKOS_LAMBDA(const int n) {
      int p;
      if (n < nprtcl_send_) {
        p = sendlist_.d_view(n).prtcl_indx; // place particles in holes created by sends
      } else {
        p = npart + (n - nprtcl_send_);     // place particle at end of arrays
      }
      for (int i=0; i<nidata; ++i) {
        pi(i,p) = irecvbuf(nidata*n + i);
      }
      for (int i=0; i<nrdata; ++i) {
        pr(i,p) = rrecvbuf(nrdata*n + i);
      }
    });
  }

  // At this point have filled nprtcl_recv holes in particle arrays from sends
  // If (nprtcl_recv < nprtcl_send), have to move particles from end of arrays to fill
  // remaining holes
  nremain = nprtcl_send - nprtcl_recv;
  if (nremain > 0) {
    int i_last_hole = nprtcl_send-1;
    int i_next_hole = nprtcl_recv;
    for (int n=1; n<=nremain; ++n) {
      int nend = npart-n;
      if (nend > prtcl_sendlist.h_view(i_last_hole).prtcl_indx) {
        // copy particle from end into hole
        int next_hole = prtcl_sendlist.h_view(i_next_hole).prtcl_indx;
        auto rdest = Kokkos::subview(ppart->prtcl_rdata, Kokkos::ALL, next_hole);
        auto rsrc  = Kokkos::subview(ppart->prtcl_rdata, Kokkos::ALL, nend);
        Kokkos::deep_copy(rdest, rsrc);
        auto idest = Kokkos::subview(ppart->prtcl_idata, Kokkos::ALL, next_hole);
        auto isrc  = Kokkos::subview(ppart->prtcl_idata, Kokkos::ALL, nend);
        Kokkos::deep_copy(idest, isrc);
        i_next_hole += 1;
      } else {
        // this index contains a hole, so do nothing except find new index of last hole
        i_last_hole -= 1;
      }
    }
  }

  // Update nparticles_thisrank.  Update cost array (use npart_thismb[nmb]?)
  MPI_Allgather(&new_npart,1,MPI_INT,(pmy_mesh->nprtcl_eachrank),1,
                  MPI_INT,MPI_COMM_WORLD);

  // shrink size of particle data arrays if needed
  if (nprtcl_send - nprtcl_recv > 0) {
    Kokkos::resize(ppart->prtcl_idata, ppart->nidata, new_npart);
    Kokkos::resize(ppart->prtcl_rdata, ppart->nrdata, new_npart);
  }
  
  // Update particle counts
  ppart->nprtcl_thispack = new_npart;
  pmy_mesh->pmb_pack->pmesh->nprtcl_thisrank = new_npart;

  // Update global particle count across all ranks
  pmy_mesh->CountParticles();

  prtcl_rrecv_req.clear();
  prtcl_irecv_req.clear();
  prtcl_nrecvs = 0;
#endif  

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CountParticleSendsAndRecvs()
//! \brief

void MeshRefinement::CountParticleSendsAndRecvs() {
#if MPI_PARALLEL_ENABLED
  // Sort sendlist on host by destrank.
  namespace KE = Kokkos::Experimental;
  std::sort(KE::begin(prtcl_sendlist.h_view), KE::end(prtcl_sendlist.h_view), SortByRank);
  // sync sendlist host array with device.  This results in sorted array on device
  prtcl_sendlist.template modify<HostMemSpace>();
  prtcl_sendlist.template sync<DevExeSpace>();

  prtcl_sends_thisrank.clear();
  if (nprtcl_send > 0) {
    int &myrank = global_variable::my_rank;
    int rank = prtcl_sendlist.h_view(0).dest_rank;
    int nprtcl = 1;

    for (int n=1; n<nprtcl_send; ++n) {
      if (prtcl_sendlist.h_view(n).dest_rank == rank) {
        ++nprtcl;
      } else {
        prtcl_sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
        rank = prtcl_sendlist.h_view(n).dest_rank;
        nprtcl = 1;
      }
    }
    prtcl_sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
  }
  prtcl_nsends = prtcl_sends_thisrank.size();

  // Share number of ranks to send to amongst all ranks
  prtcl_nsends_eachrank[global_variable::my_rank] = prtcl_nsends;
  MPI_Allgather(&prtcl_nsends, 1, MPI_INT, prtcl_nsends_eachrank.data(), 
		1, MPI_INT, par_comm);

  // Now share ParticleMessageData amongst all ranks
  // First create vector of starting indices in full vector
  std::vector<int> nsends_displ;
  nsends_displ.resize(global_variable::nranks);
  nsends_displ[0] = 0;
  for (int n=1; n<(global_variable::nranks); ++n) {
    nsends_displ[n] = nsends_displ[n-1] + prtcl_nsends_eachrank[n-1];
  }
  int nsends_allranks = nsends_displ[global_variable::nranks - 1] +
                        prtcl_nsends_eachrank[global_variable::nranks - 1];
  // Load ParticleMessageData on this rank into full vector
  prtcl_sends_allranks.resize(nsends_allranks, ParticleMessageData(0,0,0));
  for (int n=0; n<prtcl_nsends_eachrank[global_variable::my_rank]; ++n) {
    prtcl_sends_allranks[n + nsends_displ[global_variable::my_rank]] = prtcl_sends_thisrank[n];
  }

  // Share tuples using MPI derived data type for tuple of 3*int
  MPI_Datatype mpi_ituple;
  MPI_Type_contiguous(3, MPI_INT, &mpi_ituple);
  MPI_Type_commit(&mpi_ituple);
  MPI_Allgatherv(MPI_IN_PLACE, prtcl_nsends_eachrank[global_variable::my_rank],
                   mpi_ituple, prtcl_sends_allranks.data(), prtcl_nsends_eachrank.data(),
                   nsends_displ.data(), mpi_ituple, par_comm);
  MPI_Type_free(&mpi_ituple);

#endif

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CreateParticleLists()
//! \brief

void MeshRefinement::CreateParticleLists() {
#if MPI_PARALLEL_ENABLED
  auto *ppart = pmy_mesh->pmb_pack->ppart;

  nprtcl_send = 0;

  auto &pr = ppart->prtcl_rdata;
  auto &pi = ppart->prtcl_idata;
  int npart = ppart->nprtcl_thispack;
  int myrank = global_variable::my_rank;

  // Create device copy of arrays needed for mapping
  int old_nmb = pmy_mesh->nmb_total; 
  
  // Calculate total number of new MeshBlocks from the new arrays
  int new_nmb = 0;
  for (int n = 0; n < global_variable::nranks; n++) {
    new_nmb += new_nmb_eachrank[n];
  }
  
  DualArray1D<int> old_to_new("oldtonew_device", old_nmb);
  for (int i = 0; i < old_nmb; i++) {
    old_to_new.h_view(i) = oldtonew[i];
  }
  old_to_new.template modify<HostMemSpace>();
  old_to_new.template sync<DevExeSpace>();

  DualArray1D<int> new_rank("new_rank_device", new_nmb);
  for (int i = 0; i < new_nmb; i++) {
    new_rank.h_view(i) = new_rank_eachmb[i];
  }
  new_rank.template modify<HostMemSpace>();
  new_rank.template sync<DevExeSpace>();
 
  // Set particle sendlist to maximum length first
  Kokkos::realloc(prtcl_sendlist, npart);
  auto sendlist_d = prtcl_sendlist.d_view;

  int counter = 0;
  Kokkos::View<int> atom_count("atom_count");
  Kokkos::deep_copy(atom_count, counter);

  par_for("create_part_list", DevExeSpace(), 0, (npart-1),
          KOKKOS_LAMBDA(const int p) {
    
    int old_gid  = pi(PGID, p); 
    int new_gid  = old_to_new.d_view(old_gid);
    int dest_rank = new_rank.d_view(new_gid);

    // Update particle's GID to new value
    pi(PGID, p) = new_gid;

    // If particle needs to move to different rank, add to send list
    if (dest_rank != myrank) {
      int index = Kokkos::atomic_fetch_add(&atom_count(), 1);
      sendlist_d(index).prtcl_indx = p;
      sendlist_d(index).dest_gid   = new_gid; 
      sendlist_d(index).dest_rank  = dest_rank;
    }
  });

  Kokkos::deep_copy(counter, atom_count);
  nprtcl_send = counter;
  Kokkos::resize(prtcl_sendlist, nprtcl_send);

  prtcl_sendlist.template modify<DevExeSpace>();
  prtcl_sendlist.template sync<HostMemSpace>();
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::InitPartRecv()
//! \brief

void MeshRefinement::InitPartRecv() {
#if MPI_PARALLEL_ENABLED
  // Post particle receives 
  auto *ppart = pmy_mesh->pmb_pack->ppart;
  bool no_errors = true;
  if (ppart != nullptr) {
    CreateParticleLists();
    CountParticleSendsAndRecvs();
    
    prtcl_recvs_thisrank.clear();
    int nsends_allranks = prtcl_sends_allranks.size();
    for (int n=0; n<nsends_allranks; ++n) {
      if (prtcl_sends_allranks[n].recvrank == global_variable::my_rank) {
        prtcl_recvs_thisrank.emplace_back(prtcl_sends_allranks[n]);
      }
    }
    prtcl_nrecvs = prtcl_recvs_thisrank.size();

    nprtcl_recv = 0;
    for (int n=0; n<prtcl_nrecvs; ++n) {
      nprtcl_recv += prtcl_recvs_thisrank[n].nprtcls;
    }

    if (nprtcl_recv > 0) {
      // Allocate particle receive buffers
      Kokkos::realloc(prtcl_rrecvbuf, ppart->nrdata * nprtcl_recv);
      Kokkos::realloc(prtcl_irecvbuf, ppart->nidata * nprtcl_recv);

      // Initialize MPI request vectors
      prtcl_rrecv_req.clear();
      prtcl_irecv_req.clear();
      for (int n = 0; n < prtcl_nrecvs; ++n) {
        prtcl_rrecv_req.emplace_back(MPI_REQUEST_NULL);
        prtcl_irecv_req.emplace_back(MPI_REQUEST_NULL);
      }

      // Post non-blocking receives for Real data
      int data_start = 0;
      for (int n = 0; n < prtcl_nrecvs; ++n) {
        // calculate amount of data to be passed, get pointer to variables
        int data_size = (ppart->nrdata)*(prtcl_recvs_thisrank[n].nprtcls);
        int data_end = data_start + data_size;
        auto recv_ptr = Kokkos::subview(prtcl_rrecvbuf,
                                        std::make_pair(data_start, data_end));
        int drank = prtcl_recvs_thisrank[n].sendrank;
        int tag = 0; // 0 for Reals, 1 for ints

        int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                             par_comm, &(prtcl_rrecv_req[n]));
        if (ierr != MPI_SUCCESS) no_errors = false;
        data_start += data_size;
      }

      // Post non-blocking receives for int data  
      data_start = 0;
      for (int n = 0; n < prtcl_nrecvs; ++n) {
        // calculate amount of data to be passed, get pointer to variables
        int data_size = (ppart->nidata)*(prtcl_recvs_thisrank[n].nprtcls);
        int data_end = data_start + data_size;
        auto recv_ptr = Kokkos::subview(prtcl_irecvbuf,
                                        std::make_pair(data_start, data_end));
        int drank = prtcl_recvs_thisrank[n].sendrank;
        int tag = 1; // 0 for Reals, 1 for ints

        int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_INT, drank, tag,
                             par_comm, &(prtcl_irecv_req[n]));
        if (ierr != MPI_SUCCESS) no_errors = false;

        data_start += data_size;
      }
    }
  }

  // Check for particle MPI errors
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting particle receives with AMR" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  return;
}
