//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrange_interpolator.hpp

#include <cmath>
#include <iostream>
#include <list>

#include "lagrange_interpolator.hpp"
#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"

// calculate indices of the mesh block and xyz coordinate indices for the given
// point whose coordinate is given by rcoord; results returned in
// interp_indcs[4]

LagrangeInterpolator::LagrangeInterpolator(
  MeshBlockPack *pmy_pack,
  Real rcoords[3]):
              pmy_pack(pmy_pack),
              rcoord("rccord", 1), interp_indcs("interp_indcs", 1),
              interp_wghts("interp_wghts", 1, 1), point_exist(false) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  // int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int &ng = indcs.ng;
  Kokkos::realloc(rcoord, 3);
  Kokkos::realloc(interp_wghts, 2 * ng, 3);
  Kokkos::realloc(interp_indcs, 4);

  for (int i = 0; i < 3; ++i) {
    rcoord(i) = rcoords[i];
  }
  SetInterpolationIndices();
  CalculateWeight();
}

Real LagrangeInterpolator::Interpolate(DvceArray5D<Real> &val, int nvars) {
  Real ivals    = 0.;
  int &ng       = pmy_pack->pmesh->mb_indcs.ng;
  int &ii0      = interp_indcs(0);
  int &ii1      = interp_indcs(1);
  int &ii2      = interp_indcs(2);
  int &ii3      = interp_indcs(3);
  auto &weights = interp_wghts;
  auto &indcs   = pmy_pack->pmesh->mb_indcs;

  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;

  if (interp_indcs(0) == -1) { // point not on this rank
    ivals = 0.0;
  } else {
    for (int i = 0; i < 2 * ng; i++) {
      for (int j = 0; j < 2 * ng; j++) {
        for (int k = 0; k < 2 * ng; k++) {
          Real iwght  = weights(i, 0) * weights(j, 1) * weights(k, 2);
          auto buffer = Kokkos::subview(
            val,
            ii0,
            nvars,
            ii3 - (ng - k - ks) + 1,
            ii2 - (ng - j - js) + 1,
            ii1 - (ng - i - is) + 1);
          Real scalar;
          // Deep Copy Scalar View into a scalar
          Kokkos::deep_copy(scalar, buffer);
          ivals += iwght * scalar;
        }
      }
    }
  }
  return ivals;
}

Real LagrangeInterpolator::ResetPointAndInterpolate(
  DvceArray5D<Real> &val,
  int nvars,
  Real rcoords2[3]) {
  for (int i = 0; i < 3; ++i) {
    rcoord(i) = rcoords2[i];
  }
  SetInterpolationIndices();
  CalculateWeight();
  Real ivals = Interpolate(val, nvars);
  return ivals;
}

void LagrangeInterpolator::SetInterpolationIndices() {
  auto &size = pmy_pack->pmb->mb_size;
  int nmb1   = pmy_pack->nmb_thispack - 1;

  // indices default to -1 if the point is outside this MeshBlockPack
  for (int i = 0; i < 4; ++i) {
    interp_indcs(i) = -1;
  }

  for (int m = 0; m <= nmb1; ++m) {
    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    // extract MeshBlock grid cell spacings
    Real &dx1 = size.h_view(m).dx1;
    Real &dx2 = size.h_view(m).dx2;
    Real &dx3 = size.h_view(m).dx3;

    // save MeshBlock and zone indicies for nearest position to spherical patch
    // center if this angle position resides in this MeshBlock
    if (
      (rcoord(0) >= x1min && rcoord(0) < x1max)
      && (rcoord(1) >= x2min && rcoord(1) < x2max)
      && (rcoord(2) >= x3min && rcoord(2) < x3max)) {
      point_exist     = true;
      interp_indcs(0) = m;
      interp_indcs(1) =
        static_cast<int>(std::floor((rcoord(0) - (x1min + dx1 / 2.0)) / dx1));
      interp_indcs(2) =
        static_cast<int>(std::floor((rcoord(1) - (x2min + dx2 / 2.0)) / dx2));
      interp_indcs(3) =
        static_cast<int>(std::floor((rcoord(2) - (x3min + dx3 / 2.0)) / dx3));

      break;
    }
  }
}

void LagrangeInterpolator::CalculateWeight() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int &ng     = indcs.ng;
  int &ii0    = interp_indcs(0);
  int &ii1    = interp_indcs(1);
  int &ii2    = interp_indcs(2);
  int &ii3    = interp_indcs(3);

  // auto &weights = interp_wghts;
  if (ii0 == -1) { // angle not on this rank
    for (int i = 0; i < 2 * ng; ++i) {
      interp_wghts(i, 0) = 0.;
      interp_wghts(i, 1) = 0.;
      interp_wghts(i, 2) = 0.;
    }
  } else {
    // extract spherical grid positions
    Real &x0 = rcoord(0);
    Real &y0 = rcoord(1);
    Real &z0 = rcoord(2);

    // extract MeshBlock bounds
    Real &x1min = size.h_view(ii0).x1min;
    Real &x1max = size.h_view(ii0).x1max;
    Real &x2min = size.h_view(ii0).x2min;
    Real &x2max = size.h_view(ii0).x2max;
    Real &x3min = size.h_view(ii0).x3min;
    Real &x3max = size.h_view(ii0).x3max;
    // set interpolation weights
    for (int i = 0; i < 2 * ng; ++i) {
      interp_wghts(i, 0) = 1.;
      interp_wghts(i, 1) = 1.;
      interp_wghts(i, 2) = 1.;
      for (int j = 0; j < 2 * ng; ++j) {
        if (j != i) {
          Real x1vpi1 = CellCenterX(ii1 - ng + i + 1, indcs.nx1, x1min, x1max);
          Real x1vpj1 = CellCenterX(ii1 - ng + j + 1, indcs.nx1, x1min, x1max);
          interp_wghts(i, 0) *= (x0 - x1vpj1) / (x1vpi1 - x1vpj1);
          Real x2vpi1 = CellCenterX(ii2 - ng + i + 1, indcs.nx2, x2min, x2max);
          Real x2vpj1 = CellCenterX(ii2 - ng + j + 1, indcs.nx2, x2min, x2max);
          interp_wghts(i, 1) *= (y0 - x2vpj1) / (x2vpi1 - x2vpj1);
          Real x3vpi1 = CellCenterX(ii3 - ng + i + 1, indcs.nx3, x3min, x3max);
          Real x3vpj1 = CellCenterX(ii3 - ng + j + 1, indcs.nx3, x3min, x3max);
          interp_wghts(i, 2) *= (z0 - x3vpj1) / (x3vpi1 - x3vpj1);
        }
      }
    }
  }
}
