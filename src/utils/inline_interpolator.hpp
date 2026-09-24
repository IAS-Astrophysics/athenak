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
  // d/dx of the 1D Lagrange cardinal functions, evaluated at the target point
  Real dwght1[2 * NGHOST];
  Real dwght2[2 * NGHOST];
  Real dwght3[2 * NGHOST];
  bool point_exist;
};

//----------------------------------------------------------------------------------------
//! \fn void LagrangeWeights1D
//! \brief 1D Lagrange cardinal functions and their derivatives at a target point
//!
//! For nodes x_i, the cardinal functions are L_i(x) = prod_{j!=i} (x-x_j)/(x_i-x_j).
//! The derivative is evaluated as
//!   L_i'(x) = sum_{j!=i} 1/(x_i-x_j) * prod_{k!=i,j} (x-x_k)/(x_i-x_k),
//! rather than the shorter L_i(x) * sum_{j!=i} 1/(x-x_j), because the latter is
//! singular whenever the target point falls exactly on a stencil node.

template <int NPOINT>
KOKKOS_INLINE_FUNCTION
void LagrangeWeights1D(const Real x, const Real xn[NPOINT],
                       Real wght[NPOINT], Real dwght[NPOINT]) {
  for (int i = 0; i < NPOINT; ++i) {
    Real w = 1.;
    Real dw = 0.;
    for (int j = 0; j < NPOINT; ++j) {
      if (j == i) continue;
      w *= (x - xn[j]) / (xn[i] - xn[j]);
      Real term = 1. / (xn[i] - xn[j]);
      for (int k = 0; k < NPOINT; ++k) {
        if (k == i || k == j) continue;
        term *= (x - xn[k]) / (xn[i] - xn[k]);
      }
      dw += term;
    }
    wght[i] = w;
    dwght[i] = dw;
  }
}

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
  if (result.ii0 == -1) {
    for (int i = 0; i < 2 * NGHOST; ++i) {
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

    // stencil node positions along each direction
    Real x1n[2 * NGHOST], x2n[2 * NGHOST], x3n[2 * NGHOST];
    for (int i = 0; i < 2 * NGHOST; ++i) {
      x1n[i] = CellCenterX(result.ii1 - NGHOST + i + 1,
                           indcs.nx1, mb0.x1min, mb0.x1max);
      x2n[i] = CellCenterX(result.ii2 - NGHOST + i + 1,
                           indcs.nx2, mb0.x2min, mb0.x2max);
      x3n[i] = CellCenterX(result.ii3 - NGHOST + i + 1,
                           indcs.nx3, mb0.x3min, mb0.x3max);
    }

    // set interpolation weights and their derivatives
    LagrangeWeights1D<2 * NGHOST>(rcoords[0], x1n, result.wght1, result.dwght1);
    LagrangeWeights1D<2 * NGHOST>(rcoords[1], x2n, result.wght2, result.dwght2);
    LagrangeWeights1D<2 * NGHOST>(rcoords[2], x3n, result.wght3, result.dwght3);
  }

  return result;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real InterpolateLagrange(
  const DvceArray5D<Real> &val,
  int var,
  const RegionIndcs &indcs,
  const IndAndWghts<NGHOST> &indcs_wghts
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
//! \brief partial derivative of the Lagrange interpolant along direction `dir`
//!
//! Returns d/dx_dir of the same polynomial that InterpolateLagrange() evaluates, so the
//! derivative is consistent with the interpolated field by construction. It reads only
//! `val` itself, which means no separately differenced (and separately ghost-filled)
//! array is required.

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real InterpolateLagrangeDeriv(
  const DvceArray5D<Real> &val,
  int var,
  const RegionIndcs &indcs,
  const IndAndWghts<NGHOST> &indcs_wghts,
  int dir
) {
  Real ival = 0.;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;

  if (indcs_wghts.ii0 == -1) { // point not on this rank
    return 0.0;
  }

  // differentiate along `dir` by swapping in the derivative of that direction's weights
  const Real *w1 = (dir == 0) ? indcs_wghts.dwght1 : indcs_wghts.wght1;
  const Real *w2 = (dir == 1) ? indcs_wghts.dwght2 : indcs_wghts.wght2;
  const Real *w3 = (dir == 2) ? indcs_wghts.dwght3 : indcs_wghts.wght3;

  for (int i = 0; i < 2 * NGHOST; i++) {
    for (int j = 0; j < 2 * NGHOST; j++) {
      for (int k = 0; k < 2 * NGHOST; k++) {
        Real iwght = w1[i] * w2[j] * w3[k];
        ival += iwght * val(indcs_wghts.ii0, var,
                            indcs_wghts.ii3 - (NGHOST - k - ks) + 1,
                            indcs_wghts.ii2 - (NGHOST - j - js) + 1,
                            indcs_wghts.ii1 - (NGHOST - i - is) + 1);
      }
    }
  }

  return ival;
}

#endif  // UTILS_INLINE_INTERPOLATOR_HPP_
