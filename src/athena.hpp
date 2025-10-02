#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Macros.hpp>
#include "config.hpp"

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

// number of bits used to store MeshBlock local ID in MPI message tags.  Thus maximum
// number of MBs per rank is 2^(NUM_BITS_LID). This limit is required because the MPI
// standard requires signed int tag, with MPI_TAG_UB>=2^15-1 = 32,767 (inclusive). In fact
// virtually every library allows this limit to be exceeded. As of 2025, the Intel MPI
// library provided the most stringent limit of 20 bits per tag. Six bits are needed to
// store the buffer ID, leaving NUM_BITS_LID=14
#define NUM_BITS_LID 14

#define SQR(x) ( (x)*(x) )
#define SIGN(x) ( ((x) < 0.0) ? -1.0 : 1.0 )
#define ONE_3RD  0.3333333333333333
#define TWO_3RDS 0.6666666666666667
#define FOUR_3RDS 1.333333333333333

// data types only used in physics modules (defined here to avoid recursive dependencies)

// constants that determine array index of Hydro/MHD variables
// array indices for conserved: density, momemtum, total energy
enum VariableIndex {IDN=0, IM1=1, IVX=1, IM2=2, IVY=2, IM3=3, IVZ=3, IEN=4,
                    ITM=4, IPR=4, IYF=5};
// array indices for components of magnetic field
enum BFieldIndex {IBX=0, IBY=1, IBZ=2, NMAG=3};
// array indices for metric matrices in GR
enum MetricIndex {I00=0, I01=1, I02=2, I03=3, I11=4, I12=5, I13=6, I22=7, I23=8, I33=9,
                  NMETRIC=10};
// array indices for particle arrays
enum ParticlesIndex {PGID=0, PTAG=1, IPX=0, IPVX=1, IPY=2, IPVY=3, IPZ=4, IPVZ=5};

// integer constants to specify spatial reconstruction methods
enum ReconstructionMethod {dc, plm, ppm4, ppmx, wenoz};

// constants that enumerate time evolution options
enum TimeEvolution {tstatic, kinematic, dynamic};

// constants that enumerate Physics Modules implemented in code
enum PhysicsModule {HydroDynamics, MagnetoHydroDynamics,
                    SpaceTimeDynamics, UserDefined}; //SpaceTimeDynamics = Z4c

// structs to store primitive/conserved variables in one-dimension
// (density, velocity/momentum, internal/total energy, [transverse magnetic field])
struct HydPrim1D {
  Real d, vx, vy, vz, e;
};
struct HydCons1D {
  Real d, mx, my, mz, e;
};
struct MHDPrim1D {
  Real d, vx, vy, vz, e, bx, by, bz;
};
struct MHDCons1D {
  Real d, mx, my, mz, e, bx, by, bz;
};

//----------------------------------------------------------------------------------------
// define default Kokkos execution and memory spaces

using DevExeSpace = Kokkos::DefaultExecutionSpace;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace = Kokkos::HostSpace;
using ScratchMemSpace = DevExeSpace::scratch_memory_space;
using LayoutWrapper = Kokkos::LayoutRight;                // increments last index fastest
using TeamMember_t = Kokkos::TeamPolicy<>::member_type;   // for Kokkos thread teams

//----------------------------------------------------------------------------------------
// alias template declarations for various array types (formerly AthenaArrays)
// mostly used to store cell-centered variables (volume averaged)

// template declarations for construction of Kokkos::View on device
template <typename T>
using DvceArray1D = Kokkos::View<T *, LayoutWrapper, DevMemSpace>;
template <typename T>
using DvceArray2D = Kokkos::View<T **, LayoutWrapper, DevMemSpace>;
template <typename T>
using DvceArray3D = Kokkos::View<T ***, LayoutWrapper, DevMemSpace>;
template <typename T>
using DvceArray4D = Kokkos::View<T ****, LayoutWrapper, DevMemSpace>;
template <typename T>
using DvceArray5D = Kokkos::View<T *****, LayoutWrapper, DevMemSpace>;
template <typename T>
using DvceArray6D = Kokkos::View<T ******, LayoutWrapper, DevMemSpace>;

// template declarations for construction of Kokkos::View on host
template <typename T>
using HostArray1D = Kokkos::View<T *, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray2D = Kokkos::View<T **, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray3D = Kokkos::View<T ***, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray4D = Kokkos::View<T ****, LayoutWrapper, HostMemSpace>;
template <typename T>
using HostArray5D = Kokkos::View<T *****, LayoutWrapper, HostMemSpace>;

// template declarations for construction of Kokkos::DualViews
template <typename T>
using DualArray1D = Kokkos::DualView<T *, LayoutWrapper, DevMemSpace>;
template <typename T>
using DualArray2D = Kokkos::DualView<T **, LayoutWrapper, DevMemSpace>;
template <typename T>
using DualArray3D = Kokkos::DualView<T ***, LayoutWrapper, DevMemSpace>;
template <typename T>
using DualArray4D = Kokkos::DualView<T ****, LayoutWrapper, DevMemSpace>;
template <typename T>
using DualArray5D = Kokkos::DualView<T *****, LayoutWrapper, DevMemSpace>;

// template declarations for construction of Kokkos::View in scratch memory
template <typename T>
using ScrArray1D = Kokkos::View<T *, LayoutWrapper, ScratchMemSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using ScrArray2D = Kokkos::View<T **, LayoutWrapper, ScratchMemSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

//----------------------------------------------------------------------------------------
// struct for storing face-centered (area-averaged) variables, e.g. magnetic field
//                 ___________
//                 |x3f[k+1,j,i]
//                 | \    X    \
//                 |  \_________\
//          x1f[k,j,i] |         |
//                 \ X |  x2f[k,j,i]
//   x2 x3          \  |    X    |
//    \ |            \ |         |
//     \|__x1         \|_________|

template <typename T>
struct DvceFaceFld4D {
  DvceArray4D<T> x1f, x2f, x3f;  // name indicates both direction and location
  DvceFaceFld4D(const std::string &label, int nmb, int n3, int n2, int n1) :
    x1f(label + ".x1f", nmb, n3, n2, n1+1),
    x2f(label + ".x2f", nmb, n3, n2+1, n1),
    x3f(label + ".x3f", nmb, n3+1, n2, n1) {}
  ~DvceFaceFld4D() = default;
};

template <typename T>
struct DvceFaceFld5D {
  DvceArray5D<T> x1f, x2f, x3f;  // name indicates both direction and location
  DvceFaceFld5D(const std::string &label, int nmb, int nvar, int n3, int n2, int n1) :
    x1f(label + ".x1f", nmb, nvar, n3, n2, n1+1),
    x2f(label + ".x2f", nmb, nvar, n3, n2+1, n1),
    x3f(label + ".x3f", nmb, nvar, n3+1, n2, n1) {}
  ~DvceFaceFld5D() = default;
};

template <typename T>
struct HostFaceFld4D {
  HostArray4D<T> x1f, x2f, x3f;  // name indicates both direction and location
  HostFaceFld4D(const std::string &label, int nmb, int n3, int n2, int n1) :
    x1f(label + ".x1f", nmb, n3, n2, n1+1),
    x2f(label + ".x2f", nmb, n3, n2+1, n1),
    x3f(label + ".x3f", nmb, n3+1, n2, n1) {}
  ~HostFaceFld4D() = default;
};

//----------------------------------------------------------------------------------------
// struct for storing edge-centered (line-averaged) variables, e.g. EMF
//             _____________
//             |\           \
//             | \           \
//             |  \___________\
//             |   |           |
//             \   |           |
//    x2e[k,j,i]*  *x3e[k,j,i] |
//               \ |           |
//                \|_____*_____|
//                    x1e[k,j,i]

template <typename T>
struct DvceEdgeFld4D {
  DvceArray4D<T> x1e, x2e, x3e;   // name refers to direction NOT location
  DvceEdgeFld4D(const std::string &label, int nmb, int n3, int n2, int n1) :
    x1e(label + ".x1e", nmb, n3+1, n2+1, n1),
    x2e(label + ".x2e", nmb, n3+1, n2, n1+1),
    x3e(label + ".x3e", nmb, n3, n2+1, n1+1) {}
  ~DvceEdgeFld4D() = default;
};

//----------------------------------------------------------------------------------------
// wrappers for Kokkos::parallel_for
// These wrappers implement a variety of parallel execution strategies, including
// 1D-range, and thread teams for use with inner vector threads. Experiments in K-Athena
// and Parthenon indicate that 1D-range policy is generally faster than multidimensional
// MD-range policy, so the latter is not used.
//------------------------------
// 1D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(const std::string &name, DevExeSpace exec_space,
                    const int &il, const int &iu, const Function &function) {
  // compute total number of elements and call Kokkos::parallel_for()
  const int ni = iu - il + 1;
  Kokkos::parallel_for(name, Kokkos::RangePolicy<>(exec_space, 0, ni),
  KOKKOS_LAMBDA(const int &idx) {
    // compute i indices of thread and call function
    int i = (idx) + il;
    function(i);
  });
}

//------------------------------
// 2D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(const std::string &name, DevExeSpace exec_space,
                    const int &jl, const int &ju,
                    const int &il, const int &iu, const Function &function) {
  // compute total number of elements and call Kokkos::parallel_for()
  const int nj = ju - jl + 1;
  const int ni = iu - il + 1;
  const int nji  = nj * ni;
  Kokkos::parallel_for(name, Kokkos::RangePolicy<>(exec_space, 0, nji),
  KOKKOS_LAMBDA(const int &idx) {
    // compute j,i indices of thread and call function
    int j = (idx)/ni;
    int i = (idx - j*ni) + il;
    j += jl;
    function(j, i);
  });
}

//------------------------------
// 3D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(const std::string &name, DevExeSpace exec_space,
                    const int &kl, const int &ku, const int &jl, const int &ju,
                    const int &il, const int &iu, const Function &function) {
  // compute total number of elements and call Kokkos::parallel_for()
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int ni = iu - il + 1;
  const int nkji = nk * nj * ni;
  const int nji  = nj * ni;
  Kokkos::parallel_for(name, Kokkos::RangePolicy<>(exec_space, 0, nkji),
  KOKKOS_LAMBDA(const int &idx) {
    // compute k,j,i indices of thread and call function
    int k = (idx)/nji;
    int j = (idx - k*nji)/ni;
    int i = (idx - k*nji - j*ni) + il;
    k += kl;
    j += jl;
    function(k, j, i);
  });
}

//------------------------------
// 4D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(const std::string &name, DevExeSpace exec_space,
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
  KOKKOS_LAMBDA(const int &idx) {
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

//------------------------------
// 5D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(const std::string &name, DevExeSpace exec_space,
                    const int &ml, const int &mu,
                    const int &nl, const int &nu, const int &kl, const int &ku,
                    const int &jl, const int &ju, const int &il, const int &iu,
                    const Function &function) {
  // compute total number of elements and call Kokkos::parallel_for()
  const int nm = mu - ml + 1;
  const int nn = nu - nl + 1;
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int ni = iu - il + 1;
  const int nmnkji = nm * nn * nk * nj * ni;
  const int nnkji  = nn * nk * nj * ni;
  const int nkji   = nk * nj * ni;
  const int nji    = nj * ni;
  Kokkos::parallel_for(name, Kokkos::RangePolicy<>(exec_space, 0, nmnkji),
  KOKKOS_LAMBDA(const int &idx) {
    // compute m,n,k,j,i indices of thread and call function
    int m = (idx)/nnkji;
    int n = (idx - m*nnkji)/nkji;
    int k = (idx - m*nnkji - n*nkji)/nji;
    int j = (idx - m*nnkji - n*nkji - k*nji)/ni;
    int i = (idx - m*nnkji - n*nkji - k*nji - j*ni) + il;
    m += ml;
    n += nl;
    k += kl;
    j += jl;
    function(m, n, k, j, i);
  });
}

//------------------------------------------
// 1D outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(const std::string &name, DevExeSpace exec_space,
                          size_t scr_size, const int scr_level,
                          const int kl, const int ku, const Function &function) {
  const int nk = ku - kl + 1;
  Kokkos::TeamPolicy<> policy(exec_space, nk, Kokkos::AUTO);
  Kokkos::parallel_for(name, policy.set_scratch_size(scr_level,Kokkos::PerTeam(scr_size)),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int k = tmember.league_rank() + kl;
    function(tmember, k);
  });
}

//------------------------------------------
// 2D outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(const std::string &name, DevExeSpace exec_space,
                          size_t scr_size, const int scr_level,
                          const int kl, const int ku, const int jl, const int ju,
                          const Function &function) {
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int nkj = nk*nj;
  Kokkos::TeamPolicy<> policy(exec_space, nkj, Kokkos::AUTO);
  Kokkos::parallel_for(name, policy.set_scratch_size(scr_level,Kokkos::PerTeam(scr_size)),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int k = tmember.league_rank()/nj + kl;
    const int j = tmember.league_rank()%nj + jl;
    function(tmember, k, j);
  });
}

//------------------------------------------
// 3D outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(const std::string &name, DevExeSpace exec_space,
                          size_t scr_size, const int scr_level,
                          const int nl, const int nu, const int kl, const int ku,
                          const int jl, const int ju, const Function &function) {
  const int nn = nu - nl + 1;
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int nkj  = nk*nj;
  const int nnkj = nn*nk*nj;
  Kokkos::TeamPolicy<> policy(exec_space, nnkj, Kokkos::AUTO);
  Kokkos::parallel_for(name, policy.set_scratch_size(scr_level,Kokkos::PerTeam(scr_size)),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    int n = (tmember.league_rank())/nkj;
    int k = (tmember.league_rank() - n*nkj)/nj;
    int j = (tmember.league_rank() - n*nkj - k*nj) + jl;
    n += nl;
    k += kl;
    function(tmember, n, k, j);
  });
}

//------------------------------------------
// 4D outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(const std::string &name, DevExeSpace exec_space,
                          size_t scr_size, const int scr_level,
                          const int ml, const int mu,
                          const int nl, const int nu, const int kl, const int ku,
                          const int jl, const int ju, const Function &function) {
  const int nm = mu - ml + 1;
  const int nn = nu - nl + 1;
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int nkj   = nk*nj;
  const int nnkj  = nn*nk*nj;
  const int nmnkj = nm*nn*nk*nj;
  Kokkos::TeamPolicy<> policy(exec_space, nmnkj, Kokkos::AUTO);
  Kokkos::parallel_for(name, policy.set_scratch_size(scr_level,Kokkos::PerTeam(scr_size)),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    int m = (tmember.league_rank())/nnkj;
    int n = (tmember.league_rank() - m*nnkj)/nkj;
    int k = (tmember.league_rank() - m*nnkj - n*nkj)/nj;
    int j = (tmember.league_rank() - m*nnkj - n*nkj - k*nj) + jl;
    m += ml;
    n += nl;
    k += kl;
    function(tmember, m, n, k, j);
  });
}

//---------------------------------------------
// 1D inner parallel loop using TeamVectorRange
template <typename Function>
KOKKOS_INLINE_FUNCTION void par_for_inner(TeamMember_t tmember, const int il,const int iu,
                                          const Function &function) {
  // Note Kokkos::TeamVectorRange only iterates from ibegin to iend-1, so must pass iu+1
  Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, il, iu+1), function);
}

#define NREDUCTION_VARIABLES 20
//----------------------------------------------------------------------------------------
//! \struct summed_array_type
// Following code is copied from Kokkos wiki pages on building custom reducers.  It allows
// an arbitrary number (set by the compile time constant NREDUCTION_VARIABLES above) of
// sum reductions to be computed simultaneously.  Used for history outputs, etc.

namespace array_sum {  // namespace helps with name resolution in reduction identity
template< class ScalarType, int N >
struct array_type {
  ScalarType the_array[N];
  KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
  array_type() {
    for (int i = 0; i < N; i++ ) { the_array[i] = 0; }
  }
  KOKKOS_INLINE_FUNCTION   // Copy Constructor
  array_type(const array_type & rhs) {
    for (int i = 0; i < N; i++ ) {
      the_array[i] = rhs.the_array[i];
    }
  }
  KOKKOS_INLINE_FUNCTION   // add operator
  array_type& operator += (const array_type& src) {
    for ( int i = 0; i < N; i++ ) {
       the_array[i]+=src.the_array[i];
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION   // volatile add operator
  void operator += (const volatile array_type& src) volatile {
    for ( int i = 0; i < N; i++ ) {
      the_array[i]+=src.the_array[i];
    }
  }
};
// Number of reductions templated by (NHISTORY_VARIABLES)
typedef array_type<Real,(NREDUCTION_VARIABLES)> GlobalSum;  // simplifies code below
} // namespace array_sum

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
template<>
struct reduction_identity< array_sum::GlobalSum > {
  KOKKOS_FORCEINLINE_FUNCTION static array_sum::GlobalSum sum() {
    return array_sum::GlobalSum();
  }
};
}

#endif // ATHENA_HPP_
