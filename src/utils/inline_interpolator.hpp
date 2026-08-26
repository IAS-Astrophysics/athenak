#ifndef UTILS_INLINE_INTERPOLATOR_HPP_
#define UTILS_INLINE_INTERPOLATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file inline_interpolator.hpp
//! \brief The following is a wrapper of the Lagrange interpolator
//!             to be used inside a Kokkos kernel.
#include <cmath>
#include <iostream>
#include <list>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"

template <int NGHOST>
struct IndAndWghts {
  int ii0, ii1, ii2, ii3;
  Real wght1[2 * NGHOST];
  Real wght2[2 * NGHOST];
  Real wght3[2 * NGHOST];
  // Derivatives of the Lagrange basis functions, for InterpolateLagrangeDeriv() below.
  Real dwght1[2 * NGHOST];
  Real dwght2[2 * NGHOST];
  Real dwght3[2 * NGHOST];
  bool point_exist;
};

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
IndAndWghts<NGHOST> IndicesAndWeights(
  const RegionIndcs &indcs,
  const DualArray1D<RegionSize> &size,
  Real rcoords[3],
  int nmb
) {
  IndAndWghts<NGHOST> result;

  // **INTERPOLATION INDICES**
  result.ii0 = result.ii1 = result.ii2 = result.ii3 = -1;
  result.point_exist = false;

  for (int m = 0; m < nmb; ++m) {
    // extract MeshBlock bounds
    auto mb = size.d_view(m);

    // save MeshBlock and zone indicies for nearest position to spherical patch
    // center if this angle position resides in this MeshBlock
    if (
      (rcoords[0] >= mb.x1min && rcoords[0] < mb.x1max)
      && (rcoords[1] >= mb.x2min && rcoords[1] < mb.x2max)
      && (rcoords[2] >= mb.x3min && rcoords[2] < mb.x3max)) {
      result.point_exist = true;
      result.ii0 = m;
      result.ii1 = static_cast<int>(Kokkos::floor((rcoords[0]
                    - (mb.x1min + mb.dx1 / 2.0)) / mb.dx1));
      result.ii2 = static_cast<int>(Kokkos::floor((rcoords[1]
                    - (mb.x2min + mb.dx2 / 2.0)) / mb.dx2));
      result.ii3 = static_cast<int>(Kokkos::floor((rcoords[2]
                    - (mb.x3min + mb.dx3 / 2.0)) / mb.dx3));

      break;
    }
  }

  // **INTERPOLATION WEIGHTS**
  constexpr int NSTENCIL = 2 * NGHOST;
  if (result.ii0 == -1) {
    for (int i = 0; i < NSTENCIL; ++i) {
      result.wght1[i] = 0.;
      result.wght2[i] = 0.;
      result.wght3[i] = 0.;
      result.dwght1[i] = 0.;
      result.dwght2[i] = 0.;
      result.dwght3[i] = 0.;
    }
  } else {
    // extract MeshBlock bounds
    auto mb0 = size.d_view(result.ii0);

    // stencil node coordinates in each direction
    Real xn1[NSTENCIL], xn2[NSTENCIL], xn3[NSTENCIL];
    for (int i = 0; i < NSTENCIL; ++i) {
      xn1[i] = CellCenterX(result.ii1 - NGHOST + i + 1, indcs.nx1, mb0.x1min, mb0.x1max);
      xn2[i] = CellCenterX(result.ii2 - NGHOST + i + 1, indcs.nx2, mb0.x2min, mb0.x2max);
      xn3[i] = CellCenterX(result.ii3 - NGHOST + i + 1, indcs.nx3, mb0.x3min, mb0.x3max);
    }

    // set interpolation weights, i.e. the Lagrange basis functions
    //   L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
    for (int i = 0; i < NSTENCIL; ++i) {
      result.wght1[i] = 1.;
      result.wght2[i] = 1.;
      result.wght3[i] = 1.;
      for (int j = 0; j < NSTENCIL; ++j) {
        if (j != i) {
          result.wght1[i] *= (rcoords[0] - xn1[j]) / (xn1[i] - xn1[j]);
          result.wght2[i] *= (rcoords[1] - xn2[j]) / (xn2[i] - xn2[j]);
          result.wght3[i] *= (rcoords[2] - xn3[j]) / (xn3[i] - xn3[j]);
        }
      }
    }

    // set the derivatives of the Lagrange basis functions,
    //   L_i'(x) = sum_{k != i} [ 1/(x_i - x_k) * prod_{j != i,k} (x - x_j)/(x_i - x_j) ]
    // NOTE: this form is used rather than the more compact L_i(x) * sum_k 1/(x - x_k)
    // because the latter divides by (x - x_k), which blows up whenever the interpolation
    // point coincides with a stencil node (i.e. sits exactly at a cell center).
    for (int i = 0; i < NSTENCIL; ++i) {
      Real d1 = 0., d2 = 0., d3 = 0.;
      for (int k = 0; k < NSTENCIL; ++k) {
        if (k == i) continue;
        Real t1 = 1. / (xn1[i] - xn1[k]);
        Real t2 = 1. / (xn2[i] - xn2[k]);
        Real t3 = 1. / (xn3[i] - xn3[k]);
        for (int j = 0; j < NSTENCIL; ++j) {
          if (j != i && j != k) {
            t1 *= (rcoords[0] - xn1[j]) / (xn1[i] - xn1[j]);
            t2 *= (rcoords[1] - xn2[j]) / (xn2[i] - xn2[j]);
            t3 *= (rcoords[2] - xn3[j]) / (xn3[i] - xn3[j]);
          }
        }
        d1 += t1;
        d2 += t2;
        d3 += t3;
      }
      result.dwght1[i] = d1;
      result.dwght2[i] = d2;
      result.dwght3[i] = d3;
    }
  }

  return result;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real InterpolateLagrange(
  const DvceArray5D<Real> &val,
  int var,
  const RegionIndcs &indcs,
  IndAndWghts<NGHOST> indcs_wghts
) {
  Real ival = 0.;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;

  if (indcs_wghts.ii0 == -1) { // point not on this rank
    ival = 0.0;
  } else {
    for (int i = 0; i < 2 * NGHOST; i++) {
      for (int j = 0; j < 2 * NGHOST; j++) {
        for (int k = 0; k < 2 * NGHOST; k++) {
          Real iwght = indcs_wghts.wght1[i] * indcs_wghts.wght2[j] * indcs_wghts.wght3[k];
          ival += iwght * val(indcs_wghts.ii0, var,
                              indcs_wghts.ii3 - (NGHOST - k - ks) + 1,
                              indcs_wghts.ii2 - (NGHOST - j - js) + 1,
                              indcs_wghts.ii1 - (NGHOST - i - is) + 1);
        }
      }
    }
  }

  return ival;
}

//----------------------------------------------------------------------------------------
//! \fn Real InterpolateLagrangeDeriv
//! \brief Partial derivative of the Lagrange interpolant with respect to dir (0,1,2).
//!
//! The interpolant is a tensor product of 1D Lagrange polynomials, so its exact partial
//! derivative is obtained by swapping the basis functions in the differentiated direction
//! for their derivatives. This is preferable to differencing the sampled field on the
//! mesh and then interpolating the result: it only ever reads the field itself, which is
//! valid in the ghost zones, whereas a precomputed derivative array would have to be
//! filled there separately (by a boundary exchange) before it could be interpolated near
//! a MeshBlock face.
//!
//! The result is one order less accurate than the interpolant, i.e. order 2*NGHOST-1.
template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real InterpolateLagrangeDeriv(
  const DvceArray5D<Real> &val,
  int var,
  const RegionIndcs &indcs,
  IndAndWghts<NGHOST> indcs_wghts,
  int dir
) {
  Real ival = 0.;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;

  if (indcs_wghts.ii0 == -1) { // point not on this rank
    return 0.0;
  }

  for (int i = 0; i < 2 * NGHOST; i++) {
    Real w1 = (dir == 0) ? indcs_wghts.dwght1[i] : indcs_wghts.wght1[i];
    for (int j = 0; j < 2 * NGHOST; j++) {
      Real w2 = (dir == 1) ? indcs_wghts.dwght2[j] : indcs_wghts.wght2[j];
      for (int k = 0; k < 2 * NGHOST; k++) {
        Real w3 = (dir == 2) ? indcs_wghts.dwght3[k] : indcs_wghts.wght3[k];
        ival += w1 * w2 * w3 * val(indcs_wghts.ii0, var,
                                   indcs_wghts.ii3 - (NGHOST - k - ks) + 1,
                                   indcs_wghts.ii2 - (NGHOST - j - js) + 1,
                                   indcs_wghts.ii1 - (NGHOST - i - is) + 1);
      }
    }
  }

  return ival;
}

#endif  // UTILS_INLINE_INTERPOLATOR_HPP_
