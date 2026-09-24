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

ArbitraryGrid::ArbitraryGrid(MeshBlockPack *pmy_pack, std::vector<std::array<Real,3>>& cart_coord_, int rpow_,
                             int ns_):
    interp_vals("interp_vals",1,1),
    interp_indcs("interp_indcs",1,1),
    pmy_pack(pmy_pack),
    interp_wghts("interp_wghts",1,1,1),
    interp_cart_coord("interp_cart_coord", 1,1) {

  int ng = pmy_pack->pmesh->mb_indcs.ng;
  ns = (ns_ > 0) ? ns_ : ng;
  if (ns > ng || ns > kMaxHalfWidth) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "ArbitraryGrid stencil half width " << ns << " exceeds min(ng, "
              << kMaxHalfWidth << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // setup grid coordinate
  rpow       = rpow_;
  npts       = 0;
  capacity   = 0;

  // allocate memory and prepare ArbitraryGrid object for interpolation
  ResetGrid(cart_coord_);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ArbitraryGrid::ResetGrid
//! \brief set new interpolation points. Memory is only reallocated if the number of
//! points exceeds the largest number used so far, so the number of points can change
//! between calls without repeated reallocation.

void ArbitraryGrid::ResetGrid(std::vector<std::array<Real,3>>& cart_coord_) {
  cart_coord = cart_coord_;
  npts       = static_cast<int>(cart_coord.size());

  if (npts > capacity) {
    capacity = npts;
    Kokkos::realloc(interp_indcs,capacity,4);
    Kokkos::realloc(interp_wghts,capacity,2*ns,3);
    Kokkos::realloc(interp_cart_coord,capacity,3);
    Kokkos::realloc(interp_vals,capacity,interp_vals.extent_int(1));
  }

  // coordinates have to be on the device before indices and weights are computed
  for (int npt = 0; npt < npts; npt++) {
    interp_cart_coord.h_view(npt,0) = cart_coord[npt][0];
    interp_cart_coord.h_view(npt,1) = cart_coord[npt][1];
    interp_cart_coord.h_view(npt,2) = cart_coord[npt][2];
  }
  interp_cart_coord.template modify<HostMemSpace>();
  interp_cart_coord.template sync<DevExeSpace>();

  SetInterpolationIndices();
  SetInterpolationWeights();
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
  auto &cart_coord_ = interp_cart_coord;
  auto &iindcs = interp_indcs;

  par_for("parfor_arbitrary_intp_indices", DevExeSpace(), 0, npts-1,
  KOKKOS_LAMBDA(int npt) {
    // calculate x, y, z coordinate for each point
    Real& x1 = cart_coord_.d_view(npt,0);
    Real& x2 = cart_coord_.d_view(npt,1);
    Real& x3 = cart_coord_.d_view(npt,2);

    // indices default to -1 if point does not reside in this MeshBlockPack
    iindcs.d_view(npt,0) = -1;
    iindcs.d_view(npt,1) = -1;
    iindcs.d_view(npt,2) = -1;
    iindcs.d_view(npt,3) = -1;
    for (int m = 0; m <= nmb1; ++m) {
      // extract MeshBlock bounds
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      // extract MeshBlock grid cell spacings
      Real &dx1 = size.d_view(m).dx1;
      Real &dx2 = size.d_view(m).dx2;
      Real &dx3 = size.d_view(m).dx3;

      // save MeshBlock and zone indicies for nearest
      // position to spherical patch center
      // if this angle position resides in this MeshBlock
      if ((x1 >= x1min && x1 < x1max) &&
          (x2 >= x2min && x2 < x2max) &&
          (x3 >= x3min && x3 < x3max)) {
          iindcs.d_view(npt,0) = m;
          iindcs.d_view(npt,1) =
              static_cast<int>(Kokkos::floor((x1-(x1min+dx1/2.0))/dx1));
          iindcs.d_view(npt,2) =
              static_cast<int>(Kokkos::floor((x2-(x2min+dx2/2.0))/dx2));
          iindcs.d_view(npt,3) =
              static_cast<int>(Kokkos::floor((x3-(x3min+dx3/2.0))/dx3));
      }
    }
  });

  // sync dual arrays
  interp_indcs.template modify<DevExeSpace>();
  interp_indcs.template sync<HostMemSpace>();

  return;
}

void ArbitraryGrid::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int ns_ = ns;
  auto &cart_coord_ = interp_cart_coord;
  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  par_for("parfor_arbitrary_intp_weights", DevExeSpace(), 0, npts-1,
  KOKKOS_LAMBDA(int npt) {
    int m = iindcs.d_view(npt,0);
    if (m == -1) {
      for (int i=0; i<2*ns_; ++i) {
        iwghts.d_view(npt,i,0) = 0.0;
        iwghts.d_view(npt,i,1) = 0.0;
        iwghts.d_view(npt,i,2) = 0.0;
      }
      return;
    }
    int ii1 = iindcs.d_view(npt,1);
    int ii2 = iindcs.d_view(npt,2);
    int ii3 = iindcs.d_view(npt,3);
    Real x0 = cart_coord_.d_view(npt,0);
    Real y0 = cart_coord_.d_view(npt,1);
    Real z0 = cart_coord_.d_view(npt,2);
    Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;

    Real xs[2*kMaxHalfWidth], ys[2*kMaxHalfWidth], zs[2*kMaxHalfWidth];
    for (int s=0; s<2*ns_; ++s) {
      xs[s] = CellCenterX(ii1-ns_+s+1, nx1, x1min, x1max);
      ys[s] = CellCenterX(ii2-ns_+s+1, nx2, x2min, x2max);
      zs[s] = CellCenterX(ii3-ns_+s+1, nx3, x3min, x3max);
    }
    for (int i=0; i<2*ns_; ++i) {
      Real wx = 1.0, wy = 1.0, wz = 1.0;
      for (int j=0; j<2*ns_; ++j) {
        if (j != i) {
          wx *= (x0-xs[j])/(xs[i]-xs[j]);
          wy *= (y0-ys[j])/(ys[i]-ys[j]);
          wz *= (z0-zs[j])/(zs[i]-zs[j]);
        }
      }
      iwghts.d_view(npt,i,0) = wx;
      iwghts.d_view(npt,i,1) = wy;
      iwghts.d_view(npt,i,2) = wz;
    }
  });

  // sync dual arrays
  interp_wghts.template modify<DevExeSpace>();
  interp_wghts.template sync<HostMemSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ArbitraryGrid::InterpolateToGrid
//! \brief interpolate Cartesian data to cart_grid for output

void ArbitraryGrid::InterpolateToGrid(int ind0, int nvar, DvceArray5D<Real> &val) {
  if (nvar > kMaxVar) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "ArbitraryGrid can interpolate at most " << kMaxVar
              << " variables at once" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (interp_vals.extent_int(0) < capacity || interp_vals.extent_int(1) < nvar) {
    Kokkos::realloc(interp_vals, capacity, nvar);
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int ns_ = ns;
  int rp_ = rpow;
  Real cx1 = center_x1, cx2 = center_x2, cx3 = center_x3;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = interp_vals;
  auto &cart = interp_cart_coord;
  par_for("arb_interp", DevExeSpace(), 0, npts-1,
  KOKKOS_LAMBDA(int n) {
    Real acc[kMaxVar];
    for (int v=0; v<nvar; ++v) acc[v] = 0.0;
    int m = iindcs.d_view(n,0);
    if (m >= 0) {
      int ii1 = iindcs.d_view(n,1);
      int ii2 = iindcs.d_view(n,2);
      int ii3 = iindcs.d_view(n,3);
      Real xs2[2*kMaxHalfWidth], ys2[2*kMaxHalfWidth], zs2[2*kMaxHalfWidth];
      if (rp_ != 0) {
        Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
        Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
        Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
        for (int s=0; s<2*ns_; ++s) {
          Real x = CellCenterX(ii1-ns_+s+1, nx1, x1min, x1max) - cx1;
          Real y = CellCenterX(ii2-ns_+s+1, nx2, x2min, x2max) - cx2;
          Real z = CellCenterX(ii3-ns_+s+1, nx3, x3min, x3max) - cx3;
          xs2[s] = x*x;
          ys2[s] = y*y;
          zs2[s] = z*z;
        }
      }
      int i0 = ii1-ns_+1+is, j0 = ii2-ns_+1+js, k0 = ii3-ns_+1+ks;
      for (int k=0; k<2*ns_; ++k) {
        Real wk = iwghts.d_view(n,k,2);
        for (int j=0; j<2*ns_; ++j) {
          Real wjk = wk*iwghts.d_view(n,j,1);
          for (int i=0; i<2*ns_; ++i) {
            Real w = wjk*iwghts.d_view(n,i,0);
            if (rp_ != 0) {
              Real r2 = xs2[i] + ys2[j] + zs2[k];
              Real rp = (rp_ % 2 != 0) ? Kokkos::sqrt(r2) : 1.0;
              for (int p=0; p<rp_/2; ++p) rp *= r2;
              w *= rp;
            }
            for (int v=0; v<nvar; ++v) {
              acc[v] += w*val(m, ind0+v, k0+k, j0+j, i0+i);
            }
          }
        }
      }
      if (rp_ != 0) {
        Real x0 = cart.d_view(n,0) - cx1;
        Real y0 = cart.d_view(n,1) - cx2;
        Real z0 = cart.d_view(n,2) - cx3;
        Real r2 = x0*x0 + y0*y0 + z0*z0;
        Real rp = (rp_ % 2 != 0) ? Kokkos::sqrt(r2) : 1.0;
        for (int p=0; p<rp_/2; ++p) rp *= r2;
        Real inv = 1.0/rp;
        for (int v=0; v<nvar; ++v) acc[v] *= inv;
      }
    }
    for (int v=0; v<nvar; ++v) ivals.d_view(n,v) = acc[v];
  });

  // sync dual arrays
  interp_vals.template modify<DevExeSpace>();
  interp_vals.template sync<HostMemSpace>();

  return;
}
