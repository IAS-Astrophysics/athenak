//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file load_balance.cpp
//  \brief 

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// \!fn void Mesh::CalculateLoadBalance(double *clist, int *rlist, int *slist,
//                                      int *nlist, int nb)
// \brief Calculate distribution of MeshBlocks based on the cost list
// input: clist = cost of each MB (array of length nmbtotal)
//        nb = number of MeshBlocks
// output: rlist = rank to which each MB is assigned (array of length nmbtotal)
//         slist = 
//         nlist = 

void Mesh::LoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb)
{
  double min_cost = std::numeric_limits<double>::max();
  double max_cost = 0.0, totalcost = 0.0;

  // find min/max and total cost in clist
  for (int i=0; i<nb; i++) {
    totalcost += clist[i];
    min_cost = std::min(min_cost,clist[i]);
    max_cost = std::max(max_cost,clist[i]);
  }

  int j = (global_variable::nranks) - 1;
  double targetcost = totalcost/global_variable::nranks;
  double mycost = 0.0;
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
     && !adaptive && !lb_flag_ && max_cost == min_cost && global_variable::my_rank == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Number of MeshBlocks cannot be divided evenly by number of MPI ranks. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ResetLoadBalanceCounters()
// \brief reset counters and flags for load balancing

void Mesh::ResetLoadBalanceCounters()
{
  if (lb_automatic_) {
    for (int m=0; m<pmb_pack->nmb_thispack; ++m) {
      costlist[pmb_pack->pmb->mbgid.h_view(m)] = std::numeric_limits<double>::min();
      pmb_pack->pmb->mbcost(m) = std::numeric_limits<double>::min();
    }
  }
  lb_flag_ = false;
  cyc_since_lb_ = 0;
}
