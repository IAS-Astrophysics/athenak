#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file globals.hpp
//  \brief namespace containing external global variables

#include <vector>

#include "config.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace global_variable {
extern int my_rank, nranks;
extern int node_id, rank_in_node, ranks_per_node, nnodes;
extern std::vector<int> rank_to_node;
#if MPI_PARALLEL_ENABLED
extern MPI_Comm node_comm;
#endif
}

#endif // GLOBALS_HPP_
