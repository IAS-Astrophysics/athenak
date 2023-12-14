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
  // count number of MeshBlocks received by this rank (loop over new MBs on this rank)
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
  recv_buf = new AMRBuffer[nmb_recv];

  // count number of cell- and face-centered variables communicated depending on physics
  int nvarcc=0, nvarfc=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    nvarcc += (pmy_mesh->pmb_pack->phydro->nhydro);
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    nvarcc += (pmy_mesh->pmb_pack->pmhd->nmhd);
    nvarfc += 1;
  }

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

  // loop over new MBs on this rank, initialize recv buffers, and post non-blocking recvs
  int rb_idx = 0;   // recv buffer index
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
          recv_buf[rb_idx].bis = cis + ox1*cnx1;
          recv_buf[rb_idx].bie = cie + ox1*cnx1;
          recv_buf[rb_idx].bjs = cjs + ox2*cnx2;
          recv_buf[rb_idx].bje = cje + ox2*cnx2;
          recv_buf[rb_idx].bks = cks + ox3*cnx3;
          recv_buf[rb_idx].bke = cke + ox3*cnx3;
          recv_buf[rb_idx].cntcc = cnx1*cnx2*cnx3;
          recv_buf[rb_idx].cntfc = 3*cnx1*cnx2*cnx3 + cnx2*cnx3 + cnx1*cnx3 + cnx1*cnx2;
          recv_buf[rb_idx].cnt   = nvarcc*recv_buf[rb_idx].cntcc +
                                   nvarfc*recv_buf[rb_idx].cntfc;
          recv_buf[rb_idx].lid   = newm - nmbs;
          recv_buf[rb_idx].derefine = true;
          Kokkos::realloc(recv_buf[rb_idx].vars, recv_buf[rb_idx].cnt);
          // create tag using local ID of *receiving* MeshBlock, post receive
          int tag = CreateAMR_MPI_Tag(newm-nmbs, ox1, ox2, ox3);
          int ierr = MPI_Irecv(recv_buf[rb_idx].vars.data(), recv_buf[rb_idx].cnt,
                     MPI_ATHENA_REAL, pmy_mesh->rank_eachmb[oldm+l], tag, amr_comm,
                     &(recv_buf[rb_idx].req));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          rb_idx++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {   // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        recv_buf[rb_idx].bis = is;
        recv_buf[rb_idx].bie = ie;
        recv_buf[rb_idx].bjs = js;
        recv_buf[rb_idx].bje = je;
        recv_buf[rb_idx].bks = ks;
        recv_buf[rb_idx].bke = ke;
        recv_buf[rb_idx].cntcc = nx1*nx2*nx3;
        recv_buf[rb_idx].cntfc = 3*nx1*nx2*nx3 + nx2*nx3 + nx1*nx3 + nx1*nx2;
        recv_buf[rb_idx].cnt = nvarcc*recv_buf[rb_idx].cntcc +
                               nvarfc*recv_buf[rb_idx].cntfc;
        recv_buf[rb_idx].lid = newm - nmbs;
        Kokkos::realloc(recv_buf[rb_idx].vars, recv_buf[rb_idx].cnt);
        // create tag using local ID of *receiving* MeshBlock, post receive
        int tag = CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0);
        int ierr = MPI_Irecv(recv_buf[rb_idx].vars.data(), recv_buf[rb_idx].cnt,
                   MPI_ATHENA_REAL, pmy_mesh->rank_eachmb[oldm], tag, amr_comm,
                   &(recv_buf[rb_idx].req));
        if (ierr != MPI_SUCCESS) {no_errors=false;}
        rb_idx++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        recv_buf[rb_idx].bis = il; // note il:iu etc. includes ghost zones
        recv_buf[rb_idx].bie = iu;
        recv_buf[rb_idx].bjs = jl;
        recv_buf[rb_idx].bje = ju;
        recv_buf[rb_idx].bks = kl;
        recv_buf[rb_idx].bke = ku;
        recv_buf[rb_idx].cntcc = (iu-il+1)*(ju-jl+1)*(ku-kl+1);
        recv_buf[rb_idx].cntfc = (iu-il+2)*(ju-jl+1)*(ku-kl+1) +
             (iu-il+1)*(ju-jl+2)*(ku-kl+1) + (iu-il+1)*(ju-jl+1)*(ku-kl+2);
        recv_buf[rb_idx].refine = true;
        recv_buf[rb_idx].cnt = nvarcc*recv_buf[rb_idx].cntcc +
                               nvarfc*recv_buf[rb_idx].cntfc;
        recv_buf[rb_idx].lid = newm - nmbs;
        Kokkos::realloc(recv_buf[rb_idx].vars, recv_buf[rb_idx].cnt);
        // create tag using local ID of *receiving* MeshBlock, post receive
        int tag = CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0);
        int ierr = MPI_Irecv(recv_buf[rb_idx].vars.data(), recv_buf[rb_idx].cnt,
                   MPI_ATHENA_REAL, pmy_mesh->rank_eachmb[oldm], tag, amr_comm,
                   &(recv_buf[rb_idx].req));
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
  // count number of MeshBlocks to send on this rank (loop over old MBs on this rank)
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
  send_buf = new AMRBuffer[nmb_send];

  // count number of cell- and face-centered variables communicated depending on physics
  int nvarcc=0, nvarfc=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    nvarcc += (pmy_mesh->pmb_pack->phydro->nhydro);
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    nvarcc += (pmy_mesh->pmb_pack->pmhd->nmhd);
    nvarfc += 1;
  }

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

  // loop over old MBs on this rank, initialize send buffers
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
          send_buf[sb_idx].bis = il + ox1*cnx1;  // note il:iu etc. includes ghost zones
          send_buf[sb_idx].bie = iu + ox1*cnx1;
          send_buf[sb_idx].bjs = jl + ox2*cnx2;
          send_buf[sb_idx].bje = ju + ox2*cnx2;
          send_buf[sb_idx].bks = kl + ox3*cnx3;
          send_buf[sb_idx].bke = ku + ox3*cnx3;
          send_buf[sb_idx].cntcc = (iu-il+1)*(ju-jl+1)*(ku-kl+1);
          send_buf[sb_idx].cntfc = (iu-il+2)*(ju-jl+1)*(ku-kl+1) +
               (iu-il+1)*(ju-jl+2)*(ku-kl+1) + (iu-il+1)*(ju-jl+1)*(ku-kl+2);
          send_buf[sb_idx].cnt   = nvarcc*send_buf[sb_idx].cntcc +
                                   nvarfc*send_buf[sb_idx].cntfc;
          send_buf[sb_idx].lid   = oldm - ombs;
          send_buf[sb_idx].refine = true;
          Kokkos::realloc(send_buf[sb_idx].vars, send_buf[sb_idx].cnt);
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (old_lloc.level == new_lloc.level) { // old MB on same level
        if (new_rank_eachmb[newm] != global_variable::my_rank) {
          send_buf[sb_idx].bis = is;
          send_buf[sb_idx].bie = ie;
          send_buf[sb_idx].bjs = js;
          send_buf[sb_idx].bje = je;
          send_buf[sb_idx].bks = ks;
          send_buf[sb_idx].bke = ke;
          send_buf[sb_idx].cntcc = nx1*nx2*nx3;
          send_buf[sb_idx].cntfc = 3*nx1*nx2*nx3 + nx2*nx3 + nx1*nx3 + nx1*nx2;
          send_buf[sb_idx].cnt = nvarcc*send_buf[sb_idx].cntcc +
                                 nvarfc*send_buf[sb_idx].cntfc;
          send_buf[sb_idx].lid = oldm - ombs;
          Kokkos::realloc(send_buf[sb_idx].vars, send_buf[sb_idx].cnt);
          sb_idx++;
        }
      } else {                                // old MB was de-refined
        // send whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
            (new_rank_eachmb[newm] != global_variable::my_rank)) {
          send_buf[sb_idx].bis = cis;
          send_buf[sb_idx].bie = cie;
          send_buf[sb_idx].bjs = cjs;
          send_buf[sb_idx].bje = cje;
          send_buf[sb_idx].bks = cks;
          send_buf[sb_idx].bke = cke;
          send_buf[sb_idx].cntcc = cnx1*cnx2*cnx3;
          send_buf[sb_idx].cntfc = 3*cnx1*cnx2*cnx3 + cnx2*cnx3 + cnx1*cnx3 + cnx1*cnx2;
          send_buf[sb_idx].derefine = true;
          send_buf[sb_idx].cnt = nvarcc*send_buf[sb_idx].cntcc +
                                 nvarfc*send_buf[sb_idx].cntfc;
          send_buf[sb_idx].lid = oldm - ombs;
          Kokkos::realloc(send_buf[sb_idx].vars, send_buf[sb_idx].cnt);
          sb_idx++;
        }
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

  // Send data using MPI (loop over old MBs on this rank)
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
          // create tag using local ID of *receiving* MeshBlock
          // LogicalLocation &lloc = pmy_mesh->lloc_eachmb[oldm+l];
          // int ox1 = ((new_lloc.lx1 & 1) == 1);
          // int ox2 = ((new_lloc.lx2 & 1) == 1);
          // int ox3 = ((new_lloc.lx3 & 1) == 1);
          int lid = (newm + l) - new_gids_eachrank[new_rank_eachmb[newm+l]];
          int tag = CreateAMR_MPI_Tag(lid, 0, 0, 0);
          // post non-blocking send
          int ierr = MPI_Isend(send_buf[sb_idx].vars.data(), send_buf[sb_idx].cnt,
                     MPI_ATHENA_REAL, new_rank_eachmb[newm+l], tag, amr_comm,
                     &(send_buf[sb_idx].req));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (old_lloc.level == new_lloc.level) {   // old MB at same level
        if (new_rank_eachmb[newm] != global_variable::my_rank) {
          // create tag using local ID of *receiving* MeshBlock
          int lid = newm - new_gids_eachrank[new_rank_eachmb[newm]];
          int tag = CreateAMR_MPI_Tag(lid, 0, 0, 0);
          // post non-blocking send
          int ierr = MPI_Isend(send_buf[sb_idx].vars.data(), send_buf[sb_idx].cnt,
                     MPI_ATHENA_REAL, new_rank_eachmb[newm], tag, amr_comm,
                     &(send_buf[sb_idx].req));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          sb_idx++;
        }
      } else {                                  // old MB was de-refined
        // send whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
            (new_rank_eachmb[newm] != global_variable::my_rank)) {
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
  auto &sbuf = send_buf;
  // Outer loop over (# of MeshBlocks sent)*(# of variables)
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
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
#if MPI_PARALLEL_ENABLED
  auto &sbuf = send_buf;
  // Outer loop over (# of MeshBlocks sent)*(3 compnts of field)
  int nn = 3*nmb_send;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nn, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int n = (tmember.league_rank())/3;
    const int v = (tmember.league_rank() - 3*n);

    const int il = sbuf[n].bis;
    const int jl = sbuf[n].bjs;
    const int kl = sbuf[n].bks;
    const int m  = sbuf[n].lid;
    const int nicc = sbuf[n].bie - il + 1;
    const int njcc = sbuf[n].bje - jl + 1;
    const int nkcc = sbuf[n].bke - kl + 1;

    // pack x1 component
    if (v==0) {
      const int offset = (ncc*sbuf[n].cntcc + nfc*sbuf[n].cntfc);
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
        if (sbuf[n].derefine) {
          // if de-refinement, load data from coarse_a
          sbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x1f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x1f(m,k,j,i);
        }
      });

    // pack x2 component
    } else if (v==1) {
      const int offset = (ncc*sbuf[n].cntcc + nfc*sbuf[n].cntfc) + (nicc+1)*njcc*nkcc;
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
        if (sbuf[n].derefine) {
          // if de-refinement, load data from coarse_a
          sbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x2f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x2f(m,k,j,i);
        }
      });

    // pack x3 component
    } else {
      const int offset = (ncc*sbuf[n].cntcc + nfc*sbuf[n].cntfc)
                          + (nicc+1)*njcc*nkcc + nicc*(njcc+1)*nkcc;
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
        if (sbuf[n].derefine) {
          // if de-refinement, load data from coarse_a
          sbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x3f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x3f(m,k,j,i);
        }
      });
    }
  }); // end par_for_outer
#endif
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
  if (!(no_errors)) {
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
  auto &rbuf = recv_buf;
  // Outer loop over (# of MeshBlocks recv)*(# of variables)
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
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
#if MPI_PARALLEL_ENABLED
  auto &rbuf = recv_buf;
  // Outer loop over (# of MeshBlocks recv)*(3 compnts of field)
  int nnv = 3*nmb_recv;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int n = (tmember.league_rank())/3;
    const int v = (tmember.league_rank() - n*3);

    const int il = rbuf[n].bis;
    const int jl = rbuf[n].bjs;
    const int kl = rbuf[n].bks;
    const int m  = rbuf[n].lid;
    const int nicc = rbuf[n].bie - il + 1;
    const int njcc = rbuf[n].bje - jl + 1;
    const int nkcc = rbuf[n].bke - kl + 1;

    // unpack x1 component
    if (v==0) {
      const int offset = (ncc*rbuf[n].cntcc + nfc*rbuf[n].cntfc);
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
        if (rbuf[n].refine) {
          // if refinement, load data into coarse_a
          cb.x1f(m,k,j,i) = rbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x1f(m,k,j,i) = rbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        }
      });

    // unpack x2 component
    } else if (v==1) {
      const int offset = (ncc*rbuf[n].cntcc + nfc*rbuf[n].cntfc) + (nicc+1)*njcc*nkcc;
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
        if (rbuf[n].refine) {
          // if refinement, load data into coarse_a
          cb.x2f(m,k,j,i) = rbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x2f(m,k,j,i) = rbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        }
      });

    // unpack x3 component
    } else {
      const int offset = (ncc*rbuf[n].cntcc + nfc*rbuf[n].cntfc)
                          + (nicc+1)*njcc*nkcc + nicc*(njcc+1)*nkcc;
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
        if (rbuf[n].refine) {
          // if refinement, load data into coarse_a
          cb.x3f(m,k,j,i) = rbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x3f(m,k,j,i) = rbuf[n].vars(offset + (i-il + ni*(j-jl + nj*(k-kl))));
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
  bool no_errors=true;
  for (int n=0; n<nmb_send; ++n) {
    int ierr = MPI_Wait(&(send_buf[n].req), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in clearing non-blocking sends with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // send buffers no longer needed, clean-up
  delete [] send_buf;
#endif
  return;
}
