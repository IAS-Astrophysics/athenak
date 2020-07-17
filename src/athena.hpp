#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

#include "config.hpp"
#include "globals.hpp"

//----------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------
// general purpose macros (never modified)

#define SQR(x) ( (x)*(x) )
#define SIGN(x) ( ((x) < 0.0) ? -1.0 : 1.0 )

// data types only used in Hydro modules (defined here to avoid recursive dependencies)
namespace hydro {
// constants that enumerate Hydro physics options
enum class HydroEOS {adiabatic, isothermal};
enum class HydroRiemannSolver {advection, llf, hlle, hllc, roe};

// constants that determine array index of Hydro variables
enum ConsIndex {IDN=0, IM1=1, IM2=2, IM3=3, IEN=4};
enum PrimIndex {IVX=1, IVY=2, IVZ=3, IPR=4};
} // namespace hydro

#endif // ATHENA_HPP_
