//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cart_grid.cpp
//  \brief Initializes a Cartesian grid to interpolate data onto

// C/C++ headers
#include <cmath>
#include <iostream>
#include <list>

// AthenaK headers
#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "utils/arbitrary_grid_interpolator.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ArbitraryGrid::ArbitraryGrid(MeshBlockPack *pmy_pack, std::vector<std::array<Real,3>>& cart_coord_, int rpow_):
    pmy_pack(pmy_pack),
    interp_indcs("interp_indcs",1,1),
    interp_wghts("interp_wghts",1,1,1),
    interp_vals("interp_vals",1),
    interp_cart_coord("interp_cart_coord", 1,1) {

  // setup grid coordinate
  cart_coord = cart_coord_;
  npts       = static_cast<int>(cart_coord.size());
  rpow       = rpow_;

  // allocate memory for interpolation coordinates, indices, and weights
  int &ng = pmy_pack->pmesh->mb_indcs.ng;
  Kokkos::realloc(interp_indcs,npts,4);
  Kokkos::realloc(interp_wghts,npts,2*ng,3);
  Kokkos::realloc(interp_cart_coord,npts,3);

  for (int npt = 0; npt < npts; npt++) {
    interp_cart_coord.h_view(npt,0) = cart_coord[npt][0];
    interp_cart_coord.h_view(npt,1) = cart_coord[npt][1];
    interp_cart_coord.h_view(npt,2) = cart_coord[npt][2];
  }
  interp_cart_coord.template modify<HostMemSpace>();
  interp_cart_coord.template sync<DevExeSpace>();

  // Call functions to prepare ArbitraryGrid object for interpolation
  // SetInterpolationCoordinates();
  SetInterpolationIndices();
  SetInterpolationWeights();

  return;
}

void ArbitraryGrid::ResetGrid(std::vector<std::array<Real,3>>& cart_coord_) {
  cart_coord = cart_coord_;
  SetInterpolationIndices();
  SetInterpolationWeights();

  for (int npt = 0; npt < npts; npt++) {
    interp_cart_coord.h_view(npt,0) = cart_coord[npt][0];
    interp_cart_coord.h_view(npt,1) = cart_coord[npt][1];
    interp_cart_coord.h_view(npt,2) = cart_coord[npt][2];
  }
  interp_cart_coord.template modify<HostMemSpace>();
  interp_cart_coord.template sync<DevExeSpace>();
}

void ArbitraryGrid::ResetCenter(Real center_x1_, Real center_x2_, Real cetner_x3_) {
  // grid center
  center_x1 = center_x1_;
  center_x2 = center_x2_;
  center_x3 = cetner_x3_;
}

void ArbitraryGrid::SetInterpolationIndices() {
  auto &size = pmy_pack->pmb->mb_size;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &iindcs = interp_indcs;
  for (int npt=0; npt<npts; ++npt) {
    // calculate x, y, z coordinate for each point
    Real& x1 = cart_coord[npt][0];
    Real& x2 = cart_coord[npt][1];
    Real& x3 = cart_coord[npt][2];

    // indices default to -1 if point does not reside in this MeshBlockPack
    iindcs.h_view(npt,0) = -1;
    iindcs.h_view(npt,1) = -1;
    iindcs.h_view(npt,2) = -1;
    iindcs.h_view(npt,3) = -1;
    for (int m=0; m<=nmb1; ++m) {
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

      // save MeshBlock and zone indicies for nearest
      // position to spherical patch center
      // if this angle position resides in this MeshBlock
      if ((x1 >= x1min && x1 < x1max) &&
          (x2 >= x2min && x2 < x2max) &&
          (x3 >= x3min && x3 < x3max)) {
          iindcs.h_view(npt,0) = m;
          iindcs.h_view(npt,1) =
              static_cast<int>(std::floor((x1-(x1min+dx1/2.0))/dx1));
          iindcs.h_view(npt,2) =
              static_cast<int>(std::floor((x2-(x2min+dx2/2.0))/dx2));
          iindcs.h_view(npt,3) =
              static_cast<int>(std::floor((x3-(x3min+dx3/2.0))/dx3));
      }
    }
  }

  // sync dual arrays
  interp_indcs.template modify<HostMemSpace>();
  interp_indcs.template sync<DevExeSpace>();

  return;
}

void ArbitraryGrid::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  for (int npt=0; npt<npts; ++npt) {
    // extract indices
    int &ii0 = iindcs.h_view(npt,0);
    int &ii1 = iindcs.h_view(npt,1);
    int &ii2 = iindcs.h_view(npt,2);
    int &ii3 = iindcs.h_view(npt,3);

    if (ii0==-1) {  // angle not on this rank
      for (int i=0; i<2*ng; ++i) {
        iwghts.h_view(npt,i,0) = 0.0;
        iwghts.h_view(npt,i,1) = 0.0;
        iwghts.h_view(npt,i,2) = 0.0;
      }
    } else {
      // extract cartesian grid positions
      Real& x0 = cart_coord[npt][0];
      Real& y0 = cart_coord[npt][1];
      Real& z0 = cart_coord[npt][2];

      // extract MeshBlock bounds
      Real &x1min = size.h_view(ii0).x1min;
      Real &x1max = size.h_view(ii0).x1max;
      Real &x2min = size.h_view(ii0).x2min;
      Real &x2max = size.h_view(ii0).x2max;
      Real &x3min = size.h_view(ii0).x3min;
      Real &x3max = size.h_view(ii0).x3max;

      // set interpolation weights
      for (int i=0; i<2*ng; ++i) {
        iwghts.h_view(npt,i,0) = 1.;
        iwghts.h_view(npt,i,1) = 1.;
        iwghts.h_view(npt,i,2) = 1.;
        for (int j=0; j<2*ng; ++j) {
          if (j != i) {
            Real x1vpi1 = CellCenterX(ii1-ng+i+1, indcs.nx1, x1min, x1max);
            Real x1vpj1 = CellCenterX(ii1-ng+j+1, indcs.nx1, x1min, x1max);
            iwghts.h_view(npt,i,0) *= (x0-x1vpj1)/(x1vpi1-x1vpj1);
            Real x2vpi1 = CellCenterX(ii2-ng+i+1, indcs.nx2, x2min, x2max);
            Real x2vpj1 = CellCenterX(ii2-ng+j+1, indcs.nx2, x2min, x2max);
            iwghts.h_view(npt,i,1) *= (y0-x2vpj1)/(x2vpi1-x2vpj1);
            Real x3vpi1 = CellCenterX(ii3-ng+i+1, indcs.nx3, x3min, x3max);
            Real x3vpj1 = CellCenterX(ii3-ng+j+1, indcs.nx3, x3min, x3max);
            iwghts.h_view(npt,i,2) *= (z0-x3vpj1)/(x3vpi1-x3vpj1);
          }
        }
      }
    }
  }

  // sync dual arrays
  interp_wghts.template modify<HostMemSpace>();
  interp_wghts.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ArbitraryGrid::InterpolateToGrid
//! \brief interpolate Cartesian data to cart_grid for output

void ArbitraryGrid::InterpolateToGrid(int ind, DvceArray5D<Real> &val) {
  // reinitialize interpolation indices and weights if AMR
  //if (pmy_pack->pmesh->adaptive) {
  //  SetInterpolationIndices();
  //  SetInterpolationWeights();
  //}

  // capturing variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int &ng = indcs.ng;
  int index = ind;
  int & npts_ = npts;

  Real cx1 = center_x1;
  Real cx2 = center_x2;
  Real cx3 = center_x3;

  // reallocate container
  Kokkos::realloc(interp_vals,npts);

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = interp_vals;
  auto &cart = interp_cart_coord;
  par_for("int2cart",DevExeSpace(),0,npts_-1,
  KOKKOS_LAMBDA(int npt) {
    int ii0 = iindcs.d_view(npt,0);
    int ii1 = iindcs.d_view(npt,1);
    int ii2 = iindcs.d_view(npt,2);
    int ii3 = iindcs.d_view(npt,3);
    if (ii0==-1) {  // point not on this rank
      ivals.d_view(npt) = 0.0;
    } else {
      Real int_value = 0.0;
      for (int i=0; i<2*ng; i++) {
        for (int j=0; j<2*ng; j++) {
          for (int k=0; k<2*ng; k++) {
            Real iwght = iwghts.d_view(npt,i,0)*
                  iwghts.d_view(npt,j,1)*iwghts.d_view(npt,k,2);

            // Extract MB bounds
            Real &x1min = size.d_view(ii0).x1min;
            Real &x1max = size.d_view(ii0).x1max;
            Real &x2min = size.d_view(ii0).x2min;
            Real &x2max = size.d_view(ii0).x2max;
            Real &x3min = size.d_view(ii0).x3min;
            Real &x3max = size.d_view(ii0).x3max;

            Real x1v = CellCenterX(ii1-(ng-i)+1, indcs.nx1, x1min, x1max);
            Real x2v = CellCenterX(ii2-(ng-j)+1, indcs.nx2, x2min, x2max);
            Real x3v = CellCenterX(ii3-(ng-k)+1, indcs.nx3, x3min, x3max);

            x1v -= cx1;
            x2v -= cx2;
            x3v -= cx3;
            Real r = sqrt(pow(x1v,2) + pow(x2v, 2) + pow(x3v, 2));
            int_value += iwght*val(ii0,index,ii3-(ng-k-ks)+1,
                                  ii2-(ng-j-js)+1,ii1-(ng-i-is)+1) * pow(r,rpow);
          }
        }
      }
      // extract cartesian grid positions relative to grid center
      Real x0 = cart.d_view(npt,0)  - cx1;
      Real y0 = cart.d_view(npt,1) - cx2;
      Real z0 = cart.d_view(npt,2) - cx3;

      Real r0 = sqrt(pow(x0,2) + pow(y0, 2) + pow(z0, 2));

      ivals.d_view(npt) = int_value / pow(r0,rpow);
    }
  });

  // sync dual arrays
  interp_vals.template modify<DevExeSpace>();
  interp_vals.template sync<HostMemSpace>();

  return;
}

