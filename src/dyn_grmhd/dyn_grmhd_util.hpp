#ifndef DYN_GRMHD_DYN_GRMHD_UTIL_HPP_
#define DYN_GRMHD_DYN_GRMHD_UTIL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_util.hpp
//  \brief Utility functions for use with dynamic hydro

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "reconstruct/plm.hpp"

namespace dyngr {

template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void ExtractPrimitives(Real prim_pt[NPRIM], const DvceArray5D<Real>& prim,
                       const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
                       const int& nhyd, const int& nscal,
                       const int m, const int k, const int j, const int i) {
  Real mb = eos.ps.GetEOS().GetBaryonMass();
  prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
  prim_pt[PVX] = prim(m, IVX, k, j, i);
  prim_pt[PVY] = prim(m, IVY, k, j, i);
  prim_pt[PVZ] = prim(m, IVZ, k, j, i);
  for (int s = 0; s < nscal; s++) {
    prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
  }
  prim_pt[PPR] = prim(m, IPR, k, j, i);
  prim_pt[PTM] = eos.ps.GetEOS().GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR],
                                                     &prim_pt[PYF]);
}

KOKKOS_INLINE_FUNCTION
void ExtractBField(Real bu_pt[NMAG], const DvceArray5D<Real> bcc,
                   int ibx, int iby, int ibz,
                   const int m, const int k, const int j, const int i) {
  bu_pt[iby] = bcc(m, iby, k, j, i);
  bu_pt[ibz] = bcc(m, ibz, k, j, i);
}

KOKKOS_INLINE_FUNCTION
void InsertFluxes(const Real flux_pt[NCONS], const DvceArray5D<Real>& flx,
                  const int m, const int k, const int j, const int i) {
  flx(m, IDN, k, j, i) = flux_pt[CDN];
  flx(m, IM1, k, j, i) = flux_pt[CSX];
  flx(m, IM2, k, j, i) = flux_pt[CSY];
  flx(m, IM3, k, j, i) = flux_pt[CSZ];
  flx(m, IEN, k, j, i) = flux_pt[CTA];
}

// Finite-difference operator used for well-balancing scheme to get 1/2\partial_x q
// at the cell interface.
// width: half-width of stencil with respect to the interface
// dir: x = 0, y = 1, z = 2
// sign: +1 computes at +1/2, -1 computes at -1/2
// Type: a data structure for a scalar field
template<int width, int dir, int sign, typename Type>
KOKKOS_INLINE_FUNCTION
Real HalfDifferenceInterface(Type& quant, int const m,
    int const k, int const j, int const i) {
  constexpr int shifti = sign*(dir == 0);
  constexpr int shiftj = sign*(dir == 1);
  constexpr int shiftk = sign*(dir == 2);
  if constexpr (width == 1) {
    // (f(i+1) - f(i))/2
    return 0.5*(quant(m,k+shiftk,j+shiftj,i+shifti) - quant(m,k,j,i));
  } else if (width == 2) {
    // (f(i-1) - 9 f(i) + 9 f(i) - f(i+2))/12
    return (quant(m,k-shiftk,j-shiftj,i-shifti) -
           9.0*(quant(m,k,j,i) - quant(m,k+shiftk,j+shiftj,i+shifti)) -
           quant(m,k+2*shiftk,j+2*shiftj,i+2*shifti))/12.0;
  } else if (width == 3) {
    // (-f(i-2) + 10 f(i-1) - 55 f(i) + 55(i+1) - 10 f(i+2) + f(i+3))/60
    return (-quant(m,k-2*shiftk,j-2*shiftj,i-2*shifti) +
            10.0*(quant(m,k-shiftk,j-shiftj,i-shifti) -
                  quant(m,k+2*shiftk,j+2*shiftj,i+2*shifti)) -
            55.0*(quant(m,k,j,i) - quant(m,k+shiftk,j+shiftj,i+shifti)) +
            quant(m,k+3*shiftk,j+3*shiftj,i+3*shifti))/60.0;
  }
  static_assert(width >= 1 && width <= 3, "Unimplemented operator requested.");
  return 0.;
}

// Finite-difference operator used for well-balancing scheme to get 1/2\partial_x q
// at the cell interface.
// width: half-width of stencil with respect to the interface
// dir: x = 0, y = 1, z = 2
// sign: +1 computes at +1/2, -1 computes at -1/2
// Type: a data structure for a scalar field
template<int width, int dir, int sign, typename Type>
KOKKOS_INLINE_FUNCTION
Real HalfDifferenceInterface(Type& quant, int const m, int const a, int const b,
    int const k, int const j, int const i) {
  constexpr int shifti = sign*(dir == 0);
  constexpr int shiftj = sign*(dir == 1);
  constexpr int shiftk = sign*(dir == 2);
  if constexpr (width == 1) {
    // (f(i+1) - f(i))/2
    return 0.5*(quant(m,a,b,k+shiftk,j+shiftj,i+shifti) - quant(m,a,b,k,j,i));
  } else if (width == 2) {
    // (f(i-1) - 9 f(i) + 9 f(i+1) - f(i+2))/12
    return (quant(m,a,b,k-shiftk,j-shiftj,i-shifti) -
           9.0*(quant(m,a,b,k,j,i) - quant(m,a,b,k+shiftk,j+shiftj,i+shifti)) -
           quant(m,a,b,k+2*shiftk,j+2*shiftj,i+2*shifti))/12.0;
  } else if (width == 3) {
    // (-f(i-2) + 10 f(i-1) - 55 f(i) + 55(i+1) - 10 f(i+2) + f(i+3))/60
    return (-quant(m,a,b,k-2*shiftk,j-2*shiftj,i-2*shifti) +
            10.0*(quant(m,a,b,k-shiftk,j-shiftj,i-shifti) -
                  quant(m,a,b,k+2*shiftk,j+2*shiftj,i+2*shifti)) -
            55.0*(quant(m,a,b,k,j,i) - quant(m,a,b,k+shiftk,j+shiftj,i+shifti)) +
            quant(m,a,b,k+3*shiftk,j+3*shiftj,i+3*shifti))/60.0;
  }
  static_assert(width >= 1 && width <= 3, "Unimplemented operator requested.");
  return 0.;
}

// Interpolation operator used to get quantities at cell interfaces.
// width: half-width of stencil with respect to the interface
// dir: x = 0, y = 1, z = 2
// sign: +1 computes at +1/2, -1 computes at -1/2
// Type: a data structure for a scalar field
template<int width, int dir, int sign, typename Type>
KOKKOS_INLINE_FUNCTION
Real GetAtInterface(Type& quant, int const m, int const k, int const j, int const i) {
  constexpr int shifti = sign*(dir == 0);
  constexpr int shiftj = sign*(dir == 1);
  constexpr int shiftk = sign*(dir == 2);
  if constexpr (width == 1) {
    // (f(i) + f(i+1))/2
    return 0.5*(quant(m,k,j,i) + quant(m,k+shiftk,j+shiftj,i+shifti));
  } else if (width == 2) {
    // (-f(i-1) + 9 f(i) + 9 f(i+1) - f(i+2))/16
    return 0.0625*(-quant(m,k-shiftk,j-shiftj,i-shifti) +
               9.0*(quant(m,k,j,i) + quant(m,k+shiftk,j+shiftj,i+shifti)) - 
                    quant(m,k+2*shiftk,j+2*shiftj,i+2*shifti));
  } else if (width == 3) {
    // (3 f(i-2) - 25 f(i-1) + 150 f(i) + 150 f(i+1) - 25 f(i+2) + 3 f(i+3))/256
    return 0.00390625*(3.0*(quant(m,k-2*shiftk,j-2*shiftj,i-2*shifti) +
                            quant(m,k+3*shiftk,j+3*shiftj,i+3*shifti)) -
                       25.0*(quant(m,k-shiftk,j-shiftj,i-shifti) +
                             quant(m,k+2*shiftk,j+2*shiftj,i+2*shifti)) +
                       150.0*(quant(m,k,j,i) + quant(m,k+shiftk,j+shiftj,i+shifti)));
  }
  static_assert(width >= 1 && width <= 3, "Unimplemented operator requested.");
  return 0.;
}

// Interpolation operator used to get quantities at cell interfaces.
// width: half-width of stencil with respect to the interface
// dir: x = 0, y = 1, z = 2
// sign: +1 computes at +1/2, -1 computes at -1/2
// Type: a data structure for a scalar field
template<int width, int dir, int sign, typename Type>
KOKKOS_INLINE_FUNCTION
Real GetAtInterface(Type& quant, int const m, int const v, int const k, int const j,
    int const i) {
  constexpr int shifti = sign*(dir == 0);
  constexpr int shiftj = sign*(dir == 1);
  constexpr int shiftk = sign*(dir == 2);
  if constexpr (width == 1) {
    // (f(i) + f(i+1))/2
    return 0.5*(quant(m,v,k,j,i) + quant(m,v,k+shiftk,j+shiftj,i+shifti));
  } else if (width == 2) {
    // (-f(i-1) + 9 f(i) + 9 f(i+1) - f(i+2))/16
    return 0.0625*(-quant(m,v,k-shiftk,j-shiftj,i-shifti) +
               9.0*(quant(m,v,k,j,i) + quant(m,v,k+shiftk,j+shiftj,i+shifti)) - 
                    quant(m,v,k+2*shiftk,j+2*shiftj,i+2*shifti));
  } else if (width == 3) {
    // (3 f(i-2) - 25 f(i-1) + 150 f(i) + 150 f(i+1) - 25 f(i+2) + 3 f(i+3))/256
    return 0.00390625*(3.0*(quant(m,v,k-2*shiftk,j-2*shiftj,i-2*shifti) +
                            quant(m,v,k+3*shiftk,j+3*shiftj,i+3*shifti)) -
                       25.0*(quant(m,v,k-shiftk,j-shiftj,i-shifti) +
                             quant(m,v,k+2*shiftk,j+2*shiftj,i+2*shifti)) +
                       150.0*(quant(m,v,k,j,i) + quant(m,v,k+shiftk,j+shiftj,i+shifti)));
  }
  static_assert(width >= 1 && width <= 3, "Unimplemented operator requested.");
  return 0.;
}

} // namespace dyngr

#endif  // DYN_GRMHD_DYN_GRMHD_UTIL_HPP_
