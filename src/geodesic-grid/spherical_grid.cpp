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
#include "utils/lagrange_interp.hpp"
#include "hydro/hydro.hpp"
SphericalGrid::SphericalGrid(MeshBlockPack *pmbp, int nlev, Real ctr_[3], bool rotate_g, bool fluxes): 
  cartcoord("cartcoord",1,1),
  interp_indices("interp_indices",1,1),
  area("area",1),
  radius("radius",1),
  intensity("intensity",1),
  pmbp(pmbp),
  polarcoord("polarcoord",1,1),
  GeodesicGrid(nlev,rotate_g,fluxes) {
  Kokkos::realloc(area,nangles);
  Kokkos::realloc(cartcoord,nangles,3);
  Kokkos::realloc(interp_indices,nangles,4);
  Kokkos::realloc(polarcoord,nangles,2);
  auto &w0 = pmbp->phydro->w0;

  ctr[0] = ctr_[0];
  ctr[1] = ctr_[1];
  ctr[2] = ctr_[2];

  for (int n=0; n<nangles; ++n) {
    polarcoord.h_view(n,0) = acos(cart_pos.h_view(n,2));
    polarcoord.h_view(n,1) = atan2(cart_pos.h_view(n,1),cart_pos.h_view(n,0));
  }

  // sync to device
  polarcoord.template modify<HostMemSpace>();
  polarcoord.template sync<DevExeSpace>();
}

// set constant radius
void SphericalGrid::SetRadius(Real rad_) {
  Real rad = rad_;
  for (int n=0; n<nangles; ++n) {
    // set cartesian coord
    cartcoord.h_view(n,0) = rad*sin(polarcoord.h_view(n,0))*cos(polarcoord.h_view(n,1)) + ctr[0];
    cartcoord.h_view(n,1) = rad*sin(polarcoord.h_view(n,0))*sin(polarcoord.h_view(n,1)) + ctr[1];
    cartcoord.h_view(n,2) = rad*cos(polarcoord.h_view(n,0)) + ctr[2];
    // calculate area
    area.h_view(n) = solid_angles.h_view(n)*rad*rad;
  }

  // sync data
  cartcoord.template modify<HostMemSpace>();
  cartcoord.template sync<DevExeSpace>();
  area.template modify<HostMemSpace>();
  area.template sync<DevExeSpace>();
}

// set radius for deformed sphere
void SphericalGrid::SetRadius(DualArray1D<Real> radius_) {
  Kokkos::realloc(radius,nangles);
  // Kokkos::deep_copy(DevExeSpace(), radius, radius_);
  for (int n=0; n<nangles; ++n) {
    radius.h_view(n) = radius_.h_view(n);
    // set cartesian coord
    cartcoord.h_view(n,0) = radius.h_view(n)*sin(polarcoord.h_view(n,0))*cos(polarcoord.h_view(n,1)) + ctr[0];
    cartcoord.h_view(n,1) = radius.h_view(n)*sin(polarcoord.h_view(n,0))*sin(polarcoord.h_view(n,1)) + ctr[1];
    cartcoord.h_view(n,2) = radius.h_view(n)*cos(polarcoord.h_view(n,0)) + ctr[2];
    // calculate area (area/weight)
    area.h_view(n) = solid_angles.h_view(n)*radius.h_view(n)*radius.h_view(n);
  }

  // sync data
  cartcoord.template modify<HostMemSpace>();
  cartcoord.template sync<DevExeSpace>();
  area.template modify<HostMemSpace>();
  area.template sync<DevExeSpace>();
  radius.template modify<HostMemSpace>();
  radius.template sync<DevExeSpace>();
}


void SphericalGrid::CalculateIndex() {
  // set index for meshblocks and the cells that a gridpoint is in
  auto &size = pmbp->pmb->mb_size;
  size_t scr_size = ScrArray1D<Real>::shmem_size(0);

  int scr_level = 0;

  par_for_outer("spherical-grid-index", DevExeSpace(), scr_size, scr_level, 0,(pmbp->nmb_thispack-1),
  KOKKOS_LAMBDA(TeamMember_t member, const int m) {
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

    auto &x1min = size.d_view(m).x1min;
    auto &x1max = size.d_view(m).x1max;
    auto &x2min = size.d_view(m).x2min;
    auto &x2max = size.d_view(m).x2max;
    auto &x3min = size.d_view(m).x3min;
    auto &x3max = size.d_view(m).x3max;

    delta[0] = size.d_view(m).dx1;
    delta[1] = size.d_view(m).dx2;
    delta[2] = size.d_view(m).dx3;
    
    // Loop over all points to find those belonging to this spherical patch
    par_for_inner(member, 0, nangles-1, [&](const int n) {
      // Default meshblock indices to -1
      if (m==0){
        interp_indices.d_view(n,0) = -1;
      }
      if (cartcoord.d_view(n,0) >= x1min && cartcoord.d_view(n,0) <= x1max 
        && cartcoord.d_view(n,1) >= x2min && cartcoord.d_view(n,1) <= x2max 
        && cartcoord.d_view(n,2) >= x3min && cartcoord.d_view(n,2) <= x3max) {
        // save which meshblock the nth point on the geodesic grid belongs to
        interp_indices.d_view(n,0) = m;
        // save the index of the closest point in the meshblock (closer on the origin)
        interp_indices.d_view(n,1) = (int) std::floor((cartcoord.d_view(n,0)-x1min-delta[0]/2)/delta[0]);
        interp_indices.d_view(n,2) = (int) std::floor((cartcoord.d_view(n,1)-x2min-delta[1]/2)/delta[1]);
        interp_indices.d_view(n,3) = (int) std::floor((cartcoord.d_view(n,2)-x3min-delta[2]/2)/delta[2]);
      }
    });
  });

  interp_indices.template modify<DevExeSpace>();
  interp_indices.template sync<HostMemSpace>();
}


// For now the interpolator is tuned towards hydro/mhd
// Input is a 5-dim array for conserved/primitive variable

// In this function, we need to set the stencils for each gridpoint
void SphericalGrid::InterpToSphere(DvceArray5D<Real> &value) {
  Kokkos::realloc(intensity,nangles);
  
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nghost = indcs.ng;

  for (int n=0;n<nangles;++n) {
    int coordinate_ind[3];
    coordinate_ind[0] = interp_indices.h_view(n,1);
    coordinate_ind[1] = interp_indices.h_view(n,2);
    coordinate_ind[2] = interp_indices.h_view(n,3);

    int axis[3];
    axis[0] = 1;
    axis[1] = 2;
    axis[2] = 3;
    std::cout << interp_indices.h_view(n,0) << "  " << interp_indices.h_view(n,3) << std::endl;
    DualArray3D<Real> value_interp;
    Kokkos::realloc(value_interp,2*(nghost+1),2*(nghost+1),2*(nghost+1));
    for (int i=0; i<2*nghost+2; i++) {
      for (int j=0; j<2*nghost+2; j++) {
        for (int k=0; k<2*nghost+2; k++) {
          std::cout << coordinate_ind[0]-nghost+i+is << " " << coordinate_ind[1]-nghost+j+js << " " << coordinate_ind[2]-nghost+k+ks << std::endl;
          value_interp.h_view(i,j,k) = value(interp_indices.h_view(n,0),IDN,coordinate_ind[2]-nghost+k+ks,coordinate_ind[1]-nghost+j+js,coordinate_ind[0]-nghost+i+is);
        }
      }
    }
    Real x_interp[3];
    x_interp[0] = cartcoord.h_view(n,0);
    x_interp[1] = cartcoord.h_view(n,1);
    x_interp[2] = cartcoord.h_view(n,2);

    LagrangeInterp3D A = LagrangeInterp3D(pmbp, &interp_indices.h_view(n,0), coordinate_ind, x_interp, axis);
    intensity.h_view(n) = A.Evaluate(pmbp,value_interp);
  }
}
