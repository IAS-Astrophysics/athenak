//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file load_balance.cpp
//! \brief File containing various Mesh and MeshRefinement functions associated with
//! load balancing with MPI, both for uniform grids and with SMR/AMR.

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn void Mesh::LoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb)
//! \brief Calculate distribution of MeshBlocks based on input cost list
//! input: clist = cost of each MB (array of length nmbtotal)
//!        nb = number of MeshBlocks
//! output: rlist = rank to which each MB is assigned (array of length nmbtotal)
//!         slist =
//!         nlist =
//! This function is needed even on a uniform mesh with MPI, and not just for SMR/AMR,
//! which is why it is part of the Mesh and not MeshRefinement class.

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

/****

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::InitRecvAMR()
//! \brief

void MeshRefinement::InitRecvAMR() {
#if MPI_PARALLEL_ENABLED
  auto &new_nmb = new_nmb_eachrank[global_variable::my_rank];
  auto &new_gids = new_gids_eachrank[global_variable::my_rank];
  // loop over all new MBs on this rank, and check if data is coming from another rank
  for (int m=0; m<new_nmb; i++) {
    if (rank_eachmb[newtoold[m+gids]] != global_variable::my_rank) {
      // post non-blocking recvs for de-refinement

      // post non-blocking recvs for refinement

      // post non-blocking recvs for MBs moved without refinement
      MPI_Request new_request;
      MPI
      int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                           vars_comm, &(recv_buf[n].vars_req[m]));
      if (ierr != MPI_SUCCESS) {no_errors=false;}
    }
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAndSendAMR()
//! \brief

void MeshRefinement::PackAndSendAMR() {

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RecvAndUnpackAMR()
//! \brief

void MeshRefinement::RecvAndUnpackAMR() {

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearSendAMR()
//! \brief

void MeshRefinement::ClearSendAMR() {

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearRecvAMR()
//! \brief

void MeshRefinement::ClearRecvAMR() {

  return;
}
*****/
