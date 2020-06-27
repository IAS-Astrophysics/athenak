#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.


#include "config.hpp"
#include "globals.hpp"

// type alias that allows code to run with either floats or doubles
#if SINGLE_PRECISION_ENABLED

using Real = float;
#if MPI_PARALLEL_ENABLED
#define MPI_ATHENA_REAL MPI_FLOAT
#endif

#else

using Real = double;
#if MPI_PARALLEL_ENABLED
#define MPI_ATHENA_REAL MPI_DOUBLE
#endif

#endif // SINGLE_PRECISION_ENABLED


#endif // ATHENA_HPP_
