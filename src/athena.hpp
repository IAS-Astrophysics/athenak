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
// following have to be variables since they depend on nhydro which is set at run time
// values are set in MHD constructor (yes, this is a bit sketchy)
int IBX, IBY, IBZ;
} // namespace hydro

// integer constants to specify physics module (maximum of 16 set by number of bits used
// to encode ID in BoundaryValues::CreateMPItag)
enum PhysicsID {Hydro_ID, Scalars_ID, MHD_ID};

// integer constants to specify reconstruction methods
enum ReconstructionMethod {dc, plm, ppm};

// constants that enumerate time evolution options
enum TimeEvolution {stationary, kinematic, dynamic};

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

template <typename T>
using AthenaArray2DSlice = Kokkos::View<T **, Kokkos::LayoutStride, DevMemSpace>;

// alias template declarations for construction of HostArrays for, e.g. outputs
template <typename T>
using HostArray1D = Kokkos::View<T *, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray2D = Kokkos::View<T **, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray3D = Kokkos::View<T ***, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray4D = Kokkos::View<T ****, LayoutWrapper, HostMemSpace>;

// alias template declarations for construction of 1D...2D scratch arrays as Kokkos::View
template <typename T>
using AthenaScratch1D = Kokkos::View<T *, LayoutWrapper, ScratchMemSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using AthenaScratch2D = Kokkos::View<T **, LayoutWrapper, ScratchMemSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// type alias for Kokkos thread teams
using TeamMember_t = Kokkos::TeamPolicy<>::member_type;

//----------------------------------------------------------------------------------------
// struct for storing face-, edge-, and corner centered variables
// (currently only face-centered fields implemented)

template <typename T>
struct FaceArray3D {
  AthenaArray3D<T> x1f, x2f, x3f;
  FaceArray3D(const std::string &label, int n3, int n2, int n1) :
    x1f(label + ".x1f", n3, n2, n1),
    x2f(label + ".x2f", n3, n2, n1),
    x3f(label + ".x3f", n3, n2, n1) {}
  ~FaceArray3D() = default;
};

//----------------------------------------------------------------------------------------
// wrappers for Kokkos::parallel_for
// Currently these wrappers all implement a 1D range policy, since experiments in
// K-Athena and Parthenon indicate these, in general, are faster then MD range policy.

// 3D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(const std::string &name, DevExecSpace exec_space,
                    const int &kl, const int &ku, const int &jl, const int &ju,
                    const int &il, const int &iu, const Function &function)
{ 
  // compute total number of elements and call Kokkos::parallel_for()
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int ni = iu - il + 1;
  const int nkji = nk * nj * ni;
  const int nji  = nj * ni;
  Kokkos::parallel_for(name, Kokkos::RangePolicy<>(exec_space, 0, nkji),
                       KOKKOS_LAMBDA(const int &idx)
  { 
    // compute n,k,j,i indices of thread and call function
    int k = (idx)/nji;
    int j = (idx - k*nji)/ni;
    int i = (idx - k*nji - j*ni) + il;
    k += kl;
    j += jl;
    function(k, j, i);
  });
}

// 4D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(const std::string &name, DevExecSpace exec_space,
                    const int &nl, const int &nu, const int &kl, const int &ku,
                    const int &jl, const int &ju, const int &il, const int &iu,
                    const Function &function) {
  // compute total number of elements and call Kokkos::parallel_for()
  const int nn = nu - nl + 1;
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int ni = iu - il + 1;
  const int nnkji = nn * nk * nj * ni;
  const int nkji  = nk * nj * ni;
  const int nji   = nj * ni;
  Kokkos::parallel_for(name, Kokkos::RangePolicy<>(exec_space, 0, nnkji),
                       KOKKOS_LAMBDA(const int &idx)
  {
    // compute n,k,j,i indices of thread and call function
    int n = (idx)/nkji;
    int k = (idx - n*nkji)/nji;
    int j = (idx - n*nkji - k*nji)/ni;
    int i = (idx - n*nkji - k*nji - j*ni) + il;
    n += nl;
    k += kl;
    j += jl;
    function(n, k, j, i);
  });
}

// 1D outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(const std::string &name, DevExecSpace exec_space,
                          size_t scr_size, const int scr_level,
                          const int kl, const int ku, const Function &function)
{
  const int nk = ku + 1 - kl;
  Kokkos::TeamPolicy<> policy(exec_space, nk, Kokkos::AUTO);
  Kokkos::parallel_for(name, policy.set_scratch_size(scr_level,Kokkos::PerTeam(scr_size)),
                       KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    const int k = tmember.league_rank() + kl;
    function(tmember, k);
  });
}

// 2D outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(const std::string &name, DevExecSpace exec_space,
                          size_t scr_size, const int scr_level,
                          const int kl, const int ku, const int jl, const int ju,
                          const Function &function)
{ 
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int nkj = nk*nj;
  Kokkos::TeamPolicy<> policy(exec_space, nkj, Kokkos::AUTO);
  Kokkos::parallel_for(name, policy.set_scratch_size(scr_level,Kokkos::PerTeam(scr_size)),
                       KOKKOS_LAMBDA(TeamMember_t tmember)
  { 
    const int k = tmember.league_rank()/nj + kl;
    const int j = tmember.league_rank()%nj + jl;
    function(tmember, k, j);
  });
}

// 3D outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(const std::string &name, DevExecSpace exec_space,
                          size_t scr_size, const int scr_level,
                          const int nl, const int nu, const int kl, const int ku,
                          const int jl, const int ju, const Function &function)
{
  const int nn = nu - nl + 1;
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int nkj  = nk*nj;
  const int nnkj = nn*nk*nj;
  Kokkos::TeamPolicy<> policy(exec_space, nnkj, Kokkos::AUTO);
  Kokkos::parallel_for(name, policy.set_scratch_size(scr_level,Kokkos::PerTeam(scr_size)),
                       KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    int n = (tmember.league_rank())/nkj;
    int k = (tmember.league_rank() - n*nkj)/nj;
    int j = (tmember.league_rank() - n*nkj - k*nj) + jl;
    n += nl;
    k += kl;
    function(tmember, n, k, j);
  });
}


// 1D inner parallel loop using TeamVectorRange
template <typename Function>
KOKKOS_INLINE_FUNCTION void par_for_inner(TeamMember_t tmember, const int il,const int iu,
                                          const Function &function)
{
  // Note Kokkos::TeamVectorRange only iterates from ibegin to iend-1, so must pass iu+1
  Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, il, iu+1), function);
}

#endif // ATHENA_HPP_
