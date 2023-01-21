//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file load_balance.cpp
//! \brief Contains various Mesh and MeshRefinement functions associated with
//! load balancing when MPI is used, both for uniform grids and with SMR/AMR.

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

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
  // count number of MeshBlocks received on this rank (loop over new MBs)
  nmb_recv = 0;
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {   // de-refinement
      for (int l=0; l<nleaf; l++) {
        if (pmy_mesh->rank_eachmb[oldm+l] != global_variable::my_rank) {
          nmb_recv++;
        }
      }
    } else {
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        nmb_recv++;
      }
    }
  }
  if (nmb_recv == 0) return;  // nothing to do

  // allocate array of recv buffers
  recv_buf = new AMRBuffer[nmb_recv];

  // count number of cell- and face-centered data elements to be communicated depending
  // on physics enrolled
  auto &indcs = pmy_mesh->mb_indcs;
  int cntcc_same = (indcs.nx1)*(indcs.nx2)*(indcs.nx3);
  int cntcc_coar = (indcs.cnx1)*(indcs.cnx2)*(indcs.cnx3);
  int cntfc_same = (indcs.nx1+1)*(indcs.nx2  )*(indcs.nx3  ) +
                   (indcs.nx1  )*(indcs.nx2+1)*(indcs.nx3  ) +
                   (indcs.nx1  )*(indcs.nx2  )*(indcs.nx3+1);
  int cntfc_coar = (indcs.cnx1+1)*(indcs.cnx2  )*(indcs.cnx3  ) +
                   (indcs.cnx1  )*(indcs.cnx2+1)*(indcs.cnx3  ) +
                   (indcs.cnx1  )*(indcs.cnx2  )*(indcs.cnx3+1);
  int nvarcc=0, nvarfc=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    nvarcc += (pmy_mesh->pmb_pack->phydro->nhydro);
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    nvarcc += (pmy_mesh->pmb_pack->pmhd->nmhd);
    nvarfc += 1;
  }

  // loop over new MBs on this rank, initialize recv buffers, and post non-blocking recvs
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  int rb_idx = 0;     // recv buffer index
  bool no_errors=true;
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {   // de-refinement
      for (int l=0; l<nleaf; l++) {
        if (pmy_mesh->rank_eachmb[oldm+l] != global_variable::my_rank) {
          // initialize AMRBuffer data for de-refinement
          LogicalLocation &lloc = pmy_mesh->lloc_eachmb[oldm+l];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          recv_buf[rb_idx].bis = is + (ox1  )*cnx1;
          recv_buf[rb_idx].bie = is + (ox1+1)*cnx1 - 1;
          recv_buf[rb_idx].bjs = js + (ox2  )*cnx2;
          recv_buf[rb_idx].bje = js + (ox2+1)*cnx2 - 1;
          recv_buf[rb_idx].bks = ks + (ox3  )*cnx3;
          recv_buf[rb_idx].bke = ks + (ox3+1)*cnx3 - 1;
          recv_buf[rb_idx].cntcc = cntcc_coar;
          recv_buf[rb_idx].cntfc = cntfc_coar;
          recv_buf[rb_idx].cnt   = (nvarcc*cntcc_coar + nvarfc*cntfc_coar);
          recv_buf[rb_idx].lid   = newm - nmbs;
          recv_buf[rb_idx].derefine = true;
          Kokkos::realloc(recv_buf[rb_idx].vars, recv_buf[rb_idx].cnt);

          // create tag using local ID of *receiving* MeshBlock, post receive
          int tag = CreateAMR_MPI_Tag(newm-nmbs, ox1, ox2, ox3);
          int ierr = MPI_Irecv(recv_buf[rb_idx].vars.data(), recv_buf[rb_idx].cnt,
                     MPI_ATHENA_REAL, pmy_mesh->rank_eachmb[oldm+l], tag, amr_comm,
                     &(recv_buf[rb_idx].req));
/***
if (global_variable::my_rank == 1) {
std::cout <<"rank="<<global_variable::my_rank<<" m="<<m<<" Recv="<<rb_idx<<" size="<<recv_buf[rb_idx].data_size<<" tag="<<tag<<" from rank="<<pmy_mesh->rank_eachmb[oldm+l]<<" ox1/2/3="<<ox1<<" "<<ox2<<" "<<ox3<<std::endl;
}
**/
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          rb_idx++;
        }
      }
    } else {   // same level or refinement
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        if (old_lloc.level == new_lloc.level) { // same level
          recv_buf[rb_idx].bis = is;
          recv_buf[rb_idx].bie = ie;
          recv_buf[rb_idx].bjs = js;
          recv_buf[rb_idx].bje = je;
          recv_buf[rb_idx].bks = ks;
          recv_buf[rb_idx].bke = ke;
          recv_buf[rb_idx].cntcc = cntcc_same ;
          recv_buf[rb_idx].cntfc = cntfc_same ;
          recv_buf[rb_idx].cnt   = (nvarcc*cntcc_same + nvarfc*cntfc_same);
          recv_buf[rb_idx].lid   = newm - nmbs;
          Kokkos::realloc(recv_buf[rb_idx].vars, recv_buf[rb_idx].cnt);
        } else {                                // refinement
          recv_buf[rb_idx].bis = cis;
          recv_buf[rb_idx].bie = cie;
          recv_buf[rb_idx].bjs = cjs;
          recv_buf[rb_idx].bje = cje;
          recv_buf[rb_idx].bks = cks;
          recv_buf[rb_idx].bke = cke;
          recv_buf[rb_idx].cntcc = cntcc_coar ;
          recv_buf[rb_idx].cntfc = cntfc_coar ;
          recv_buf[rb_idx].cnt   = (nvarcc*cntcc_coar + nvarfc*cntfc_coar);
          recv_buf[rb_idx].lid   = newm - nmbs;
          recv_buf[rb_idx].refine = true;
          Kokkos::realloc(recv_buf[rb_idx].vars, recv_buf[rb_idx].cnt);
        }
        // create tag using local ID of *receiving* MeshBlock, post receive
        int tag = CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0);
        int ierr = MPI_Irecv(recv_buf[rb_idx].vars.data(), recv_buf[rb_idx].cnt,
                   MPI_ATHENA_REAL, pmy_mesh->rank_eachmb[oldm], tag, amr_comm,
                   &(recv_buf[rb_idx].req));
/***
if (global_variable::my_rank == 1) {
std::cout <<"rank="<<global_variable::my_rank<<" m="<<m<<" Recv="<<rb_idx<<" size="<<recv_buf[rb_idx].data_size<<" tag="<<tag<<" from rank="<<pmy_mesh->rank_eachmb[oldm]<<std::endl;
}
***/
        if (ierr != MPI_SUCCESS) {no_errors=false;}
        rb_idx++;
      }
    }
  }

  // Quit if MPI error detected
  if (no_errors == false) {
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
  // count number of MeshBlocks to send on this rank (loop over old MBs)
  nmb_send = 0;
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {   // refinement
      for (int l=0; l<nleaf; l++) {
        if (new_rank_eachmb[newm + l] != global_variable::my_rank) {
          nmb_send++;
        }
      }
    } else {
      if (new_rank_eachmb[newm] != global_variable::my_rank) {
        nmb_send++;
      }
    }
  }
  if (nmb_send == 0) return;  // nothing to do

  // allocate array of send buffers
  send_buf = new AMRBuffer[nmb_send];

  // count number of cell- and face-centered data elements to be communicated depending
  // on physics enrolled
  auto &indcs = pmy_mesh->mb_indcs;
  int cntcc_same = (indcs.nx1)*(indcs.nx2)*(indcs.nx3);
  int cntcc_coar = (indcs.cnx1)*(indcs.cnx2)*(indcs.cnx3);
  int cntfc_same = (indcs.nx1+1)*(indcs.nx2  )*(indcs.nx3  ) +
                   (indcs.nx1  )*(indcs.nx2+1)*(indcs.nx3  ) +
                   (indcs.nx1  )*(indcs.nx2  )*(indcs.nx3+1);
  int cntfc_coar = (indcs.cnx1+1)*(indcs.cnx2  )*(indcs.cnx3  ) +
                   (indcs.cnx1  )*(indcs.cnx2+1)*(indcs.cnx3  ) +
                   (indcs.cnx1  )*(indcs.cnx2  )*(indcs.cnx3+1);
  int nvarcc=0, nvarfc=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    nvarcc += (pmy_mesh->pmb_pack->phydro->nhydro);
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    nvarcc += (pmy_mesh->pmb_pack->pmhd->nmhd);
    nvarfc += 1;
  }

  // loop over old MBs on this rank, initialize send buffers
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  int sb_idx = 0;     // send buffer index
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {   // refinement
      for (int l=0; l<nleaf; l++) {
        if (new_rank_eachmb[newm+l] != global_variable::my_rank) {
          LogicalLocation &lloc = new_lloc_eachmb[newm+l];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          send_buf[sb_idx].bis = is + (ox1  )*cnx1;
          send_buf[sb_idx].bie = is + (ox1+1)*cnx1 - 1;
          send_buf[sb_idx].bjs = js + (ox2  )*cnx2;
          send_buf[sb_idx].bje = js + (ox2+1)*cnx2 - 1;
          send_buf[sb_idx].bks = ks + (ox3  )*cnx3;
          send_buf[sb_idx].bke = ks + (ox3+1)*cnx3 - 1;
          send_buf[sb_idx].cntcc = cntcc_coar;
          send_buf[sb_idx].cntfc = cntfc_coar;
          send_buf[sb_idx].cnt   = (nvarcc*cntcc_coar + nvarfc*cntfc_coar);
          send_buf[sb_idx].lid   = oldm - ombs;
          send_buf[sb_idx].refine = true;
          Kokkos::realloc(send_buf[sb_idx].vars, send_buf[sb_idx].cnt);
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (new_rank_eachmb[newm] != global_variable::my_rank) {
        if (old_lloc.level == new_lloc.level) { // same level
          send_buf[sb_idx].bis = is;
          send_buf[sb_idx].bie = ie;
          send_buf[sb_idx].bjs = js;
          send_buf[sb_idx].bje = je;
          send_buf[sb_idx].bks = ks;
          send_buf[sb_idx].bke = ke;
          send_buf[sb_idx].cntcc = cntcc_same;
          send_buf[sb_idx].cntcc = cntfc_same;
          send_buf[sb_idx].cnt   = (nvarcc*cntcc_same + nvarfc*cntfc_same);
          send_buf[sb_idx].lid   = oldm - ombs;
          Kokkos::realloc(send_buf[sb_idx].vars,(cntcc_same + cntfc_same));
        } else {                                // de-refinement
          send_buf[sb_idx].bis = cis;
          send_buf[sb_idx].bie = cie;
          send_buf[sb_idx].bjs = cjs;
          send_buf[sb_idx].bje = cje;
          send_buf[sb_idx].bks = cks;
          send_buf[sb_idx].bke = cke;
          send_buf[sb_idx].cntcc = cntcc_coar;
          send_buf[sb_idx].cntfc = cntfc_coar;
          send_buf[sb_idx].cnt   = (nvarcc*cntcc_coar + nvarfc*cntfc_coar);
          send_buf[sb_idx].lid = oldm - ombs;
          send_buf[sb_idx].derefine = true;
          Kokkos::realloc(send_buf[sb_idx].vars, send_buf[sb_idx].cnt);
        }
        sb_idx++;
      }
    }
  }

  // Pack data into send buffers in parallel
  hydro::Hydro* phydro = pmy_mesh->pmb_pack->phydro;
  mhd::MHD* pmhd = pmy_mesh->pmb_pack->pmhd;

  nvarcc = 0; nvarfc = 0;
  if (phydro != nullptr) {
    PackAMRBuffersCC(phydro->u0, phydro->coarse_u0, nvarcc, nvarfc);
    nvarcc += phydro->nhydro;
  }
  if (pmhd != nullptr) {
    PackAMRBuffersCC(pmhd->u0, pmhd->coarse_u0, nvarcc, nvarfc);
    nvarcc += pmhd->nmhd;
    PackAMRBuffersFC(pmhd->b0, pmhd->coarse_b0, nvarcc, nvarfc);
    nvarfc += 1;
  }

  // Send data using MPI (loop over old MBs)
  bool no_errors=true;
  sb_idx = 0;     // send buffer index
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {   // refinement
      for (int l=0; l<nleaf; l++) {
        if (new_rank_eachmb[newm+l] != global_variable::my_rank) {
          // create tag using local ID of *receiving* MeshBlock
          LogicalLocation &lloc = pmy_mesh->lloc_eachmb[oldm+l];
          int ox1 = ((new_lloc.lx1 & 1) == 1);
          int ox2 = ((new_lloc.lx2 & 1) == 1);
          int ox3 = ((new_lloc.lx3 & 1) == 1);
          int lid = (newm + l) - new_gids_eachrank[new_rank_eachmb[newm+l]];
          int tag = CreateAMR_MPI_Tag(lid, 0, 0, 0);
          // post non-blocking send
          int ierr = MPI_Isend(send_buf[sb_idx].vars.data(), send_buf[sb_idx].cnt,
                     MPI_ATHENA_REAL, new_rank_eachmb[newm+l], tag, amr_comm,
                     &(send_buf[sb_idx].req));
/***
if (global_variable::my_rank == 0) {
std::cout << "Send="<<sb_idx<<" size="<<data_size_coar<<" tag="<<tag<<" rank="<<new_rank_eachmb[newm+l]<<" ox1/2/3="<<ox1<<" "<<ox2<<" "<<ox3<<std::endl;
}
**/
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (new_rank_eachmb[newm] != global_variable::my_rank) {
        if (old_lloc.level == new_lloc.level) {   // same level
          // create tag using local ID of *receiving* MeshBlock
          int lid = newm - new_gids_eachrank[new_rank_eachmb[newm]];
          int tag = CreateAMR_MPI_Tag(lid, 0, 0, 0);
          // post non-blocking send
          int ierr = MPI_Isend(send_buf[sb_idx].vars.data(), send_buf[sb_idx].cnt,
                     MPI_ATHENA_REAL, new_rank_eachmb[newm], tag, amr_comm,
                     &(send_buf[sb_idx].req));
/***
if (global_variable::my_rank == 0) {
std::cout << "Send="<<sb_idx<<" size="<<send_buf[sb_idx].data_size<<" tag="<<tag<<" rank="<<new_rank_eachmb[newm]<<std::endl;
}
***/
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          sb_idx++;
        } else {                                // de-refinement
          // create tag using local ID of *receiving* MeshBlock
          int ox1 = ((old_lloc.lx1 & 1) == 1);
          int ox2 = ((old_lloc.lx2 & 1) == 1);
          int ox3 = ((old_lloc.lx3 & 1) == 1);
          int lid = newm - new_gids_eachrank[new_rank_eachmb[newm]];
          int tag = CreateAMR_MPI_Tag(lid, ox1, ox2, ox3);
          // post non-blocking send
          int ierr = MPI_Isend(send_buf[sb_idx].vars.data(), send_buf[sb_idx].cnt,
                     MPI_ATHENA_REAL, new_rank_eachmb[newm], tag, amr_comm,
                     &(send_buf[sb_idx].req));
/***
if (global_variable::my_rank == 0) {
std::cout << "Send="<<sb_idx<<" size="<<send_buf[sb_idx].data_size<<" tag="<<tag<<" rank="<<new_rank_eachmb[newm]<<" ox1/2/3="<<ox1<<" "<<ox2<<" "<<ox3<<std::endl;
}
***/
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          sb_idx++;
        }
      }
    }
  }

  // Quit if MPI error detected
  if (no_errors == false) {
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
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &sbuf = send_buf;

  // Outer loop over (# of MeshBlocks sent)*(# of variables)
  int nnv = nmb_send*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int n = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - n*nvar);

    const int il = sbuf[n].bis;
    const int jl = sbuf[n].bjs;
    const int kl = sbuf[n].bks;
    const int ni = sbuf[n].bie - il + 1;
    const int nj = sbuf[n].bje - jl + 1;
    const int nk = sbuf[n].bke - kl + 1;
    const int nkji = nk*nj*ni;
    const int nji  = nj*ni;
    const int m  = sbuf[n].lid;
    const int offset = ncc*sbuf[n].cntcc + nfc*sbuf[n].cntfc;

    // Middle loop over k,j,i
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + il;
      k += kl;
      j += jl;
      if (sbuf[n].derefine) {
        // if de-refinement, load data from coarse_a
        sbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = ca(m,v,k,j,i);
      } else {
        // if refinement or same level, load data from a
        sbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = a(m,v,k,j,i);
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
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RecvAndUnpackAMR()
//! \brief Checks non-blocking receives have finished, calls function to unpack buffers,
//! deletes receive buffers. Equivalent to some of the work done inside MPI_PARALLEL block
//! in the Mesh::RedistributeAndRefineMeshBlocks() function in amr_loadbalance.cpp

void MeshRefinement::RecvAndUnpackAMR() {
#if MPI_PARALLEL_ENABLED
  // Wait for all receives to finish
  bool no_errors=true;
  for (int n=0; n<nmb_recv; ++n) {
    int ierr = MPI_Wait(&(recv_buf[n].req), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (no_errors == false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking receives with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Unpack data
  hydro::Hydro* phydro = pmy_mesh->pmb_pack->phydro;
  mhd::MHD* pmhd = pmy_mesh->pmb_pack->pmhd;
  int nvarcc=0, nvarfc=0;
  if (phydro != nullptr) {
    UnpackAMRBuffersCC(phydro->u0, phydro->coarse_u0, nvarcc, nvarfc);
    nvarcc += phydro->nhydro;
  }
  if (pmhd != nullptr) {
    UnpackAMRBuffersCC(pmhd->u0, pmhd->coarse_u0, nvarcc, nvarfc);
    nvarcc += pmhd->nmhd;
    UnpackAMRBuffersFC(pmhd->b0, pmhd->coarse_b0, nvarcc, nvarfc);
    nvarfc += 1;
  }

  // recv buffers no longer needed, clean-up
  delete [] recv_buf;
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
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &rbuf = recv_buf;

  // Outer loop over (# of MeshBlocks sent)*(# of variables)
  int nnv = nmb_recv*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int n = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - n*nvar);

    const int il = rbuf[n].bis;
    const int jl = rbuf[n].bjs;
    const int kl = rbuf[n].bks;
    const int ni = rbuf[n].bie - il + 1;
    const int nj = rbuf[n].bje - jl + 1;
    const int nk = rbuf[n].bke - kl + 1;
    const int nkji = nk*nj*ni;
    const int nji  = nj*ni;
    const int m  = rbuf[n].lid;
    const int offset = ncc*rbuf[n].cntcc + nfc*rbuf[n].cntfc;

    // Middle loop over k,j,i
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + il;
      k += kl;
      j += jl;
      if (rbuf[n].refine) {
        // if refinement, load data into coarse_a
        ca(m,v,k,j,i) = rbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
      } else {
        // if de-refinement or same level, load data into a
        a(m,v,k,j,i) = rbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
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
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearSendAMR()
//! \brief Checks all non-blocking sends completed, deletes send buffers.

void MeshRefinement::ClearSendAMR() {
  bool no_errors=true;
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<nmb_send; ++n) {
    int ierr = MPI_Wait(&(send_buf[n].req), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (no_errors == false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking receives with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // send buffers no longer needed, clean-up
  delete [] send_buf;
#endif
  return;
}
