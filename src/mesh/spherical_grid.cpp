//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.cpp
//  \brief Initializes angular mesh and orthonormal tetrad

#include <cmath>
#include <list>

//#include "coordinates/coordinates.hpp"
#include "spherical_grid.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"

SphericalGrid::SphericalGrid(MeshBlockPack *pmbp, int *nlev, int *nang, bool *rotate_g, Real rad_): 
  cartcoord("cartcoord",1,1),
  interp_indices("interp_indices",1,1),
  GeodesicGrid(pmbp,nlev,nang,rotate_g) {
  int nangles = *nang;
  Kokkos::realloc(cartcoord,nangles,3);
  Kokkos::realloc(interp_indices,nangles,4);

  // set radius
  rad = rad_;

  for (int n=0; n<nangles; ++n) {
    // set cartesian coord
    cartcoord.h_view(n,0) = rad*sin(polarcoord.h_view(n,0))*cos(polarcoord.h_view(n,1));
    cartcoord.h_view(n,1) = rad*sin(polarcoord.h_view(n,0))*sin(polarcoord.h_view(n,1));
    cartcoord.h_view(n,2) = rad*cos(polarcoord.h_view(n,0));

    // overwrite solid angle (area/weight)
    solid_angle.h_view(n) *= rad*rad;
    
    // overwrite arc length
    //for (int m=0; m<6; ++m){
    //    arc_lengths.h_view(n,m) *= rad;
    //}
  }

  // sync data
  cartcoord.template modify<HostMemSpace>();
  cartcoord.template sync<DevExeSpace>();
  solid_angle.template modify<HostMemSpace>();
  solid_angle.template sync<DevExeSpace>();

  // set index for meshblocks and the cells that a gridpoint is in
  auto &size = pmbp->pmb->mb_size;

  for (int m=0; m<(pmbp->nmb_thispack);++m) {
    Real origin[3];
    Real delta[3];
    int sizes[3];

    auto &indcs = pmbp->pmesh->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    int &nghost = indcs.ng;
    int &nx1 = indcs.nx1;
    int &nx2 = indcs.nx2;
    int &nx3 = indcs.nx3;

    auto &x1min = size.h_view(m).x1min;
    auto &x1max = size.h_view(m).x1max;
    auto &x2min = size.h_view(m).x2min;
    auto &x2max = size.h_view(m).x2max;
    auto &x3min = size.h_view(m).x3min;
    auto &x3max = size.h_view(m).x3max;

    origin[0] = CellCenterX(is, nx1, x1min, x1max);
    origin[1] = CellCenterX(js, nx2, x2min, x2max);
    origin[2] = CellCenterX(ks, nx3, x3min, x3max);

    sizes[0] = nx1 + 2*(nghost);
    sizes[1] = nx2 + 2*(nghost);
    sizes[2] = nx3 + 2*(nghost);

    delta[0] = size.h_view(m).dx1;
    delta[1] = size.h_view(m).dx2;
    delta[2] = size.h_view(m).dx3;
    
    // Loop over all points to find those belonging to this spherical patch
    for (int n=0; n<nangles; ++n){
      if (cartcoord.h_view(n,0) >= x1min && cartcoord.h_view(n,0) <= x1max 
        && cartcoord.h_view(n,1) >= x2min && cartcoord.h_view(n,1) <= x2max 
        && cartcoord.h_view(n,2) >= x3min && cartcoord.h_view(n,3) <= x3max){
        // save which meshblock the nth point on the geodesic grid belongs to
        interp_indices.h_view(n,0) = m;
        // save the index of the closest point in the meshblock (closer on the origin)
        interp_indices.h_view(n,1) = (int) std::floor((cartcoord.h_view(n,0)-x1min)/delta[0]);
        interp_indices.h_view(n,2) = (int) std::floor((cartcoord.h_view(n,1)-x2min)/delta[1]);
        interp_indices.h_view(n,3) = (int) std::floor((cartcoord.h_view(n,2)-x3min)/delta[2]);
      }
    }
  }
}