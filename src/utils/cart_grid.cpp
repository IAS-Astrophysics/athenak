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
#include "cart_grid.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

CartesianGrid::CartesianGrid(MeshBlockPack *pmy_pack, Real center[3],
                                    Real extent[3], int numpoints[3], bool is_cheb):
    pmy_pack(pmy_pack),
    interp_indcs("interp_indcs",1,1,1,1),
    interp_wghts("interp_wghts",1,1,1,1,1),
    interp_vals("interp_vals",1,1,1) {
  // initialize parameters for the grid
  // uniform grid or spectral grid
  is_cheby = is_cheb;

  // grid center
  center_x1 = center[0];
  center_x2 = center[1];
  center_x3 = center[2];

  // grid center
  extent_x1 = extent[0];
  extent_x2 = extent[1];
  extent_x3 = extent[2];

  // lower bound
  min_x1 = center_x1 - extent_x1;
  min_x2 = center_x2 - extent_x2;
  min_x3 = center_x3 - extent_x3;

  // upper bound
  max_x1 = center_x1 + extent_x1;
  max_x2 = center_x2 + extent_x2;
  max_x3 = center_x3 + extent_x3;

  // number of points
  nx1 = numpoints[0];
  nx2 = numpoints[1];
  nx3 = numpoints[2];

  // resolution
  d_x1 = (max_x1-min_x1)/(nx1-1);
  d_x2 = (max_x2-min_x2)/(nx2-1);
  d_x3 = (max_x3-min_x3)/(nx3-1);

  // allocate memory for interpolation coordinates, indices, and weights
  int &ng = pmy_pack->pmesh->mb_indcs.ng;
  Kokkos::realloc(interp_indcs,nx1,nx2,nx3,4);
  Kokkos::realloc(interp_wghts,nx1,nx2,nx3,2*ng,3);

  // Call functions to prepare CartesianGrid object for interpolation
  // SetInterpolationCoordinates();
  SetInterpolationIndices();
  SetInterpolationWeights();

  return;
}

void CartesianGrid::ResetCenter(Real center[3]) {
  // grid center
  center_x1 = center[0];
  center_x2 = center[1];
  center_x3 = center[2];

  // lower bound
  min_x1 = center_x1 - extent_x1;
  min_x2 = center_x2 - extent_x2;
  min_x3 = center_x3 - extent_x3;

  // upper bound
  max_x1 = center_x1 + extent_x1;
  max_x2 = center_x2 + extent_x2;
  max_x3 = center_x3 + extent_x3;

  SetInterpolationIndices();
  SetInterpolationWeights();
}

void CartesianGrid::ResetCenterAndExtent(Real center[3], Real extent[3]) {
  // grid center
  center_x1 = center[0];
  center_x2 = center[1];
  center_x3 = center[2];

  // grid extent
  extent_x1 = extent[0];
  extent_x2 = extent[1];
  extent_x3 = extent[2];

  // lower bound
  min_x1 = center_x1 - extent_x1;
  min_x2 = center_x2 - extent_x2;
  min_x3 = center_x3 - extent_x3;

  // upper bound
  max_x1 = center_x1 + extent_x1;
  max_x2 = center_x2 + extent_x2;
  max_x3 = center_x3 + extent_x3;

  SetInterpolationIndices();
  SetInterpolationWeights();
}

void CartesianGrid::SetInterpolationIndices() {
  auto &size = pmy_pack->pmb->mb_size;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &iindcs = interp_indcs;
  for (int nx=0; nx<nx1; ++nx) {
    for (int ny=0; ny<nx2; ++ny) {
      for (int nz=0; nz<nx3; ++nz) {
        // calculate x, y, z coordinate for each point
        Real x1 = min_x1 + nx * d_x1;
        Real x2 = min_x2 + ny * d_x2;
        Real x3 = min_x3 + nz * d_x3;
        if (is_cheby) {
          x1 = center_x1 + extent_x1*std::cos(nx*M_PI/(nx1-1));
          x2 = center_x2 + extent_x2*std::cos(ny*M_PI/(nx2-1));
          x3 = center_x3 + extent_x3*std::cos(nz*M_PI/(nx3-1));
        }
        // indices default to -1 if point does not reside in this MeshBlockPack
        iindcs.h_view(nx,ny,nz,0) = -1;
        iindcs.h_view(nx,ny,nz,1) = -1;
        iindcs.h_view(nx,ny,nz,2) = -1;
        iindcs.h_view(nx,ny,nz,3) = -1;
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
          if ((x1 >= x1min && x1 <= x1max) &&
              (x2 >= x2min && x2 <= x2max) &&
              (x3 >= x3min && x3 <= x3max)) {
              iindcs.h_view(nx,ny,nz,0) = m;
              iindcs.h_view(nx,ny,nz,1) =
                  static_cast<int>(std::floor((x1-(x1min+dx1/2.0))/dx1));
              iindcs.h_view(nx,ny,nz,2) =
                  static_cast<int>(std::floor((x2-(x2min+dx2/2.0))/dx2));
              iindcs.h_view(nx,ny,nz,3) =
                  static_cast<int>(std::floor((x3-(x3min+dx3/2.0))/dx3));
          }
        }
      }
    }
  }

  // sync dual arrays
  interp_indcs.template modify<HostMemSpace>();
  interp_indcs.template sync<DevExeSpace>();

  return;
}

void CartesianGrid::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  for (int nx=0; nx<nx1; ++nx) {
  for (int ny=0; ny<nx2; ++ny) {
  for (int nz=0; nz<nx3; ++nz) {
    // extract indices
    int &ii0 = iindcs.h_view(nx,ny,nz,0);
    int &ii1 = iindcs.h_view(nx,ny,nz,1);
    int &ii2 = iindcs.h_view(nx,ny,nz,2);
    int &ii3 = iindcs.h_view(nx,ny,nz,3);

    if (ii0==-1) {  // angle not on this rank
      for (int i=0; i<2*ng; ++i) {
        iwghts.h_view(nx,ny,nz,i,0) = 0.0;
        iwghts.h_view(nx,ny,nz,i,1) = 0.0;
        iwghts.h_view(nx,ny,nz,i,2) = 0.0;
      }
    } else {
      // extract cartesian grid positions
      Real x0 = min_x1 + nx * d_x1;
      Real y0 = min_x2 + ny * d_x2;
      Real z0 = min_x3 + nz * d_x3;
      if (is_cheby) {
        x0 = center_x1 + extent_x1*std::cos(nx*M_PI/(nx1-1));
        y0 = center_x2 + extent_x2*std::cos(ny*M_PI/(nx2-1));
        z0 = center_x3 + extent_x3*std::cos(nz*M_PI/(nx3-1));
      }
      // extract MeshBlock bounds
      Real &x1min = size.h_view(ii0).x1min;
      Real &x1max = size.h_view(ii0).x1max;
      Real &x2min = size.h_view(ii0).x2min;
      Real &x2max = size.h_view(ii0).x2max;
      Real &x3min = size.h_view(ii0).x3min;
      Real &x3max = size.h_view(ii0).x3max;

      // set interpolation weights
      for (int i=0; i<2*ng; ++i) {
        iwghts.h_view(nx,ny,nz,i,0) = 1.;
        iwghts.h_view(nx,ny,nz,i,1) = 1.;
        iwghts.h_view(nx,ny,nz,i,2) = 1.;
        for (int j=0; j<2*ng; ++j) {
          if (j != i) {
            Real x1vpi1 = CellCenterX(ii1-ng+i+1, indcs.nx1, x1min, x1max);
            Real x1vpj1 = CellCenterX(ii1-ng+j+1, indcs.nx1, x1min, x1max);
            iwghts.h_view(nx,ny,nz,i,0) *= (x0-x1vpj1)/(x1vpi1-x1vpj1);
            Real x2vpi1 = CellCenterX(ii2-ng+i+1, indcs.nx2, x2min, x2max);
            Real x2vpj1 = CellCenterX(ii2-ng+j+1, indcs.nx2, x2min, x2max);
            iwghts.h_view(nx,ny,nz,i,1) *= (y0-x2vpj1)/(x2vpi1-x2vpj1);
            Real x3vpi1 = CellCenterX(ii3-ng+i+1, indcs.nx3, x3min, x3max);
            Real x3vpj1 = CellCenterX(ii3-ng+j+1, indcs.nx3, x3min, x3max);
            iwghts.h_view(nx,ny,nz,i,2) *= (z0-x3vpj1)/(x3vpi1-x3vpj1);
          }
        }
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
//! \fn void CartesianGrid::InterpolateToGrid
//! \brief interpolate Cartesian data to cart_grid for output

void CartesianGrid::InterpolateToGrid(int ind, DvceArray5D<Real> &val) {
  // reinitialize interpolation indices and weights if AMR
  //if (pmy_pack->pmesh->adaptive) {
  //  SetInterpolationIndices();
  //  SetInterpolationWeights();
  //}

  // capturing variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int &ng = indcs.ng;
  int n_x1 = nx1 - 1;
  int n_x2 = nx2 - 1;
  int n_x3 = nx3 - 1;
  int index = ind;

  // reallocate container
  Kokkos::realloc(interp_vals,nx1,nx2,nx3);

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = interp_vals;
  par_for("int2cart",DevExeSpace(),0,n_x1,0,n_x2,0,n_x3,
  KOKKOS_LAMBDA(int nx, int ny, int nz) {
    int ii0 = iindcs.d_view(nx,ny,nz,0);
    int ii1 = iindcs.d_view(nx,ny,nz,1);
    int ii2 = iindcs.d_view(nx,ny,nz,2);
    int ii3 = iindcs.d_view(nx,ny,nz,3);
    if (ii0==-1) {  // point not on this rank
      ivals.d_view(nx,ny,nz) = 0.0;
    } else {
      Real int_value = 0.0;
      for (int i=0; i<2*ng; i++) {
        for (int j=0; j<2*ng; j++) {
          for (int k=0; k<2*ng; k++) {
            Real iwght = iwghts.d_view(nx,ny,nz,i,0)*
                  iwghts.d_view(nx,ny,nz,j,1)*iwghts.d_view(nx,ny,nz,k,2);
            int_value += iwght*val(ii0,index,ii3-(ng-k-ks)+1,
                                  ii2-(ng-j-js)+1,ii1-(ng-i-is)+1);
          }
        }
      }
      ivals.d_view(nx,ny,nz) = int_value;
    }
  });

  // sync dual arrays
  interp_vals.template modify<DevExeSpace>();
  interp_vals.template sync<HostMemSpace>();

  return;
}

