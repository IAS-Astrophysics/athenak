#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

#include <Kokkos_Core.hpp>
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

// data types only used in physics modules (defined here to avoid recursive dependencies)

namespace hydro {
// constants that determine array index of Hydro variables
enum ConsIndex {IDN=0, IM1=1, IM2=2, IM3=3, IEN=4};
enum PrimIndex {IVX=1, IVY=2, IVZ=3, IPR=4};
} // namespace hydro

//----------------------------------------------------------------------------------------
// define default Kokkos execution and memory spaces

using DevExecSpace = Kokkos::DefaultExecutionSpace;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace = Kokkos::HostSpace;
using ScratchMemSpace = DevExecSpace::scratch_memory_space;
using LayoutWrapper = Kokkos::LayoutRight;   // increments last index fastest

// alias template declarations for construction of 1D...6D AthenaArrays as Kokkos::View
template <typename T>
using AthenaArray1D = Kokkos::View<T *, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthenaArray2D = Kokkos::View<T **, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthenaArray3D = Kokkos::View<T ***, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthenaArray4D = Kokkos::View<T ****, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthenaArray5D = Kokkos::View<T *****, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthenaArray6D = Kokkos::View<T ******, LayoutWrapper, DevMemSpace>;

// alias template declarations for construction of HostArrays for, e.g. outputs
template <typename T>
using HostArray1D = Kokkos::View<T *, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray2D = Kokkos::View<T **, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray3D = Kokkos::View<T ***, LayoutWrapper, HostMemSpace>;

// alias template declarations for construction of 1D...2D scratch arrays as Kokkos::View
template <typename T>
using AthenaScratch1D = Kokkos::View<T *, LayoutWrapper, ScratchMemSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using AthenaScratch2D = Kokkos::View<T **, LayoutWrapper, ScratchMemSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// type alias for Kokkos thread teams
using TeamMember_t = Kokkos::TeamPolicy<>::member_type;

#endif // ATHENA_HPP_
