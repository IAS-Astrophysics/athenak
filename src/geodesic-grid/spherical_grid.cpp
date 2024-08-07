//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.cpp
//  \brief Initializes a spherical grid to interpolate data onto

// C/C++ headers
#include <cmath>
#include <iostream>
#include <list>

// AthenaK headers
#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "spherical_grid.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

SphericalGrid::SphericalGrid(MeshBlockPack *ppack, int nlev, Real rad):
    GeodesicGrid(nlev,true,false),
    pmy_pack(ppack),
    radius(rad),
    interp_coord("interp_coord",1,1),
    interp_indcs("interp_indcs",1,1),
    interp_wghts("interp_wghts",1,1,1),
    interp_vals("interp_vals",1,1) {
  // reallocate and set interpolation coordinates, indices, and weights
  int &ng = pmy_pack->pmesh->mb_indcs.ng;
  Kokkos::realloc(interp_coord,nangles,3);
  Kokkos::realloc(interp_indcs,nangles,4);
  Kokkos::realloc(interp_wghts,nangles,2*ng,3);

  // Call functions to prepare SphericalGrid object for interpolation
  SetInterpolationCoordinates();
  SetInterpolationIndices();
  SetInterpolationWeights();

  return;
}

//----------------------------------------------------------------------------------------
//! \brief SphericalGrid destructor

SphericalGrid::~SphericalGrid() {
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalGrid::SetInterpolationCoordinates
//! \brief set Cartesian coordinates corresponding to radius at spherical surface

void SphericalGrid::SetInterpolationCoordinates() {
  // NOTE(@pdmullen): if constructing a SphericalGrid to interface with Cartesian Kerr-
  // Schild data, the SphericalGrid radius is assumed to correspond to a spherical Kerr-
  // Schild radius, meaning that when setting the x1, x2, and x3 interpolation coordinates
  // we must translate between the two coordinate systems.
  if (pmy_pack->pcoord->is_general_relativistic ||
      pmy_pack->pcoord->is_dynamical_relativistic) {
    for (int n=0; n<nangles; ++n) {
      Real &spin = pmy_pack->pcoord->coord_data.bh_spin;
      Real &theta = polar_pos.h_view(n,0);
      Real &phi = polar_pos.h_view(n,1);
      interp_coord.h_view(n,0) = (radius*cos(phi)-spin*sin(phi))*sin(theta);
      interp_coord.h_view(n,1) = (radius*sin(phi)+spin*cos(phi))*sin(theta);
      interp_coord.h_view(n,2) = radius*cos(theta);
    }
  } else {
    for (int n=0; n<nangles; ++n) {
      Real &theta = polar_pos.h_view(n,0);
      Real &phi = polar_pos.h_view(n,1);
      interp_coord.h_view(n,0) = radius*cos(phi)*sin(theta);
      interp_coord.h_view(n,1) = radius*sin(phi)*sin(theta);
      interp_coord.h_view(n,2) = radius*cos(theta);
    }
  }

  // sync dual arrays
  interp_coord.template modify<HostMemSpace>();
  interp_coord.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalGrid::SetInterpolationIndices
//! \brief determine which MeshBlocks and MeshBlock zones therein will be used in
//         interpolation onto the sphere

void SphericalGrid::SetInterpolationIndices() {
  auto &size = pmy_pack->pmb->mb_size;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = nangles - 1;

  auto &rcoord = interp_coord;
  auto &iindcs = interp_indcs;
  for (int n=0; n<=nang1; ++n) {
    // indices default to -1 if angle does not reside in this MeshBlockPack
    iindcs.h_view(n,0) = -1;
    iindcs.h_view(n,1) = -1;
    iindcs.h_view(n,2) = -1;
    iindcs.h_view(n,3) = -1;
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

      // save MeshBlock and zone indicies for nearest position to spherical patch center
      // if this angle position resides in this MeshBlock
      if ((rcoord.h_view(n,0) >= x1min && rcoord.h_view(n,0) <= x1max) &&
          (rcoord.h_view(n,1) >= x2min && rcoord.h_view(n,1) <= x2max) &&
          (rcoord.h_view(n,2) >= x3min && rcoord.h_view(n,2) <= x3max)) {
        iindcs.h_view(n,0) = m;
        iindcs.h_view(n,1) = static_cast<int>(std::floor((rcoord.h_view(n,0)-
                                                          (x1min+dx1/2.0))/dx1));
        iindcs.h_view(n,2) = static_cast<int>(std::floor((rcoord.h_view(n,1)-
                                                          (x2min+dx2/2.0))/dx2));
        iindcs.h_view(n,3) = static_cast<int>(std::floor((rcoord.h_view(n,2)-
                                                          (x3min+dx3/2.0))/dx3));
      }
    }
  }

  // sync dual arrays
  interp_indcs.template modify<HostMemSpace>();
  interp_indcs.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalGrid::SetInterpolationWeights
//! \brief set weights used by Lagrangian interpolation

void SphericalGrid::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  for (int n=0; n<nangles; ++n) {
    // extract indices
    int &ii0 = iindcs.h_view(n,0);
    int &ii1 = iindcs.h_view(n,1);
    int &ii2 = iindcs.h_view(n,2);
    int &ii3 = iindcs.h_view(n,3);

    if (ii0==-1) {  // angle not on this rank
      for (int i=0; i<2*ng; ++i) {
        iwghts.h_view(n,i,0) = 0.0;
        iwghts.h_view(n,i,1) = 0.0;
        iwghts.h_view(n,i,2) = 0.0;
      }
    } else {
      // extract spherical grid positions
      Real &x0 = interp_coord.h_view(n,0);
      Real &y0 = interp_coord.h_view(n,1);
      Real &z0 = interp_coord.h_view(n,2);

      // extract MeshBlock bounds
      Real &x1min = size.h_view(ii0).x1min;
      Real &x1max = size.h_view(ii0).x1max;
      Real &x2min = size.h_view(ii0).x2min;
      Real &x2max = size.h_view(ii0).x2max;
      Real &x3min = size.h_view(ii0).x3min;
      Real &x3max = size.h_view(ii0).x3max;

      // set interpolation weights
      for (int i=0; i<2*ng; ++i) {
        iwghts.h_view(n,i,0) = 1.;
        iwghts.h_view(n,i,1) = 1.;
        iwghts.h_view(n,i,2) = 1.;
        for (int j=0; j<2*ng; ++j) {
          if (j != i) {
            Real x1vpi1 = CellCenterX(ii1-ng+i+1, indcs.nx1, x1min, x1max);
            Real x1vpj1 = CellCenterX(ii1-ng+j+1, indcs.nx1, x1min, x1max);
            iwghts.h_view(n,i,0) *= (x0-x1vpj1)/(x1vpi1-x1vpj1);
            Real x2vpi1 = CellCenterX(ii2-ng+i+1, indcs.nx2, x2min, x2max);
            Real x2vpj1 = CellCenterX(ii2-ng+j+1, indcs.nx2, x2min, x2max);
            iwghts.h_view(n,i,1) *= (y0-x2vpj1)/(x2vpi1-x2vpj1);
            Real x3vpi1 = CellCenterX(ii3-ng+i+1, indcs.nx3, x3min, x3max);
            Real x3vpj1 = CellCenterX(ii3-ng+j+1, indcs.nx3, x3min, x3max);
            iwghts.h_view(n,i,2) *= (z0-x3vpj1)/(x3vpi1-x3vpj1);
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
//! \fn void SphericalGrid::InterpolateToSphere
//! \brief interpolate Cartesian data to surface of sphere

void SphericalGrid::InterpolateToSphere(int nvars, DvceArray5D<Real> &val) {
  // reinitialize interpolation indices and weights if AMR
  if (pmy_pack->pmesh->adaptive) {
    SetInterpolationIndices();
    SetInterpolationWeights();
  }

  // capturing variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int &ng = indcs.ng;
  int nang1 = nangles - 1;
  int nvar1 = nvars - 1;

  // reallocate container
  Kokkos::realloc(interp_vals,nangles,nvars);

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = interp_vals;
  par_for("int2sph",DevExeSpace(),0,nang1,0,nvar1,
  KOKKOS_LAMBDA(int n, int v) {
    int ii0 = iindcs.d_view(n,0);
    int ii1 = iindcs.d_view(n,1);
    int ii2 = iindcs.d_view(n,2);
    int ii3 = iindcs.d_view(n,3);

    if (ii0==-1) {  // angle not on this rank
      ivals.d_view(n,v) = 0.0;
    } else {
      Real int_value = 0.0;
      for (int i=0; i<2*ng; i++) {
        for (int j=0; j<2*ng; j++) {
          for (int k=0; k<2*ng; k++) {
            Real iwght = iwghts.d_view(n,i,0)*iwghts.d_view(n,j,1)*iwghts.d_view(n,k,2);
            int_value += iwght*val(ii0,v,ii3-(ng-k-ks)+1,ii2-(ng-j-js)+1,ii1-(ng-i-is)+1);
          }
        }
      }
      ivals.d_view(n,v) = int_value;
    }
  });

  // sync dual arrays
  interp_vals.template modify<DevExeSpace>();
  interp_vals.template sync<HostMemSpace>();

  return;
}
