//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_surface.cpp
//! \brief Implements SphericalSurface for interpolating data onto one or more spheres

#include "spherical_surface.hpp"

#include <cmath>
#include <cstdlib>  // exit
#include <iostream>
#include <list>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

//----------------------------------------------------------------------------------------
// constructors

SphericalSurface::SphericalSurface(MeshBlockPack *pmy_pack, int ntheta,
                                   Real rad, Real xc, Real yc, Real zc,
                                   int nphi_in, bool uniform_theta, int nintp)
    : SphericalSurface(pmy_pack, ntheta, std::vector<Real>{rad}, xc, yc, zc,
                       nphi_in, uniform_theta, nintp) {}

SphericalSurface::SphericalSurface(MeshBlockPack *pmy_pack, int ntheta,
                                   const std::vector<Real> &rad, Real xc, Real yc,
                                   Real zc, int nphi_in, bool uniform_theta,
                                   int nintp)
    : ntheta(ntheta),
      nphi(nphi_in > 0 ? nphi_in : 2 * ntheta),
      nradii(static_cast<int>(rad.size())),
      radii("radii", 1),
      xc(xc),
      yc(yc),
      zc(zc),
      int_weights("int_weights", 1),
      cart_pos("cart_pos", 1, 1),
      polar_pos("polar_pos", 1, 1),
      interp_vals("interp_vals", 1),
      interp_indcs("interp_indcs", 1, 1),
      interp_wghts("interp_wghts", 1, 1, 1),
      pmy_pack(pmy_pack),
      uniform_theta_(uniform_theta) {
  // ninterp semantics follow SphericalGrid: number of points per axis
  ninterp = (nintp <= 0) ? pmy_pack->pmesh->mb_indcs.ng * 2 : nintp;
  if (ninterp > pmy_pack->pmesh->mb_indcs.ng * 2 + 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "ninterp = " << ninterp << " exceeds maximum allowed value of "
              << pmy_pack->pmesh->mb_indcs.ng * 2 + 1 << std::endl;
    std::exit(EXIT_FAILURE);
  }

  nangles = nphi * ntheta;
  npoints = nradii * nangles;

  Kokkos::realloc(radii, nradii);
  for (int r = 0; r < nradii; ++r) {
    radii.h_view(r) = rad[r];
  }
  radii.template modify<HostMemSpace>();
  radii.template sync<DevExeSpace>();

  Kokkos::realloc(int_weights, nangles);
  Kokkos::realloc(polar_pos, nangles, 2);
  Kokkos::realloc(cart_pos, npoints, 3);
  Kokkos::realloc(interp_vals, npoints);
  Kokkos::realloc(interp_indcs, npoints, 4);
  Kokkos::realloc(interp_wghts, npoints, ninterp, 3);

  InitializeAngleAndWeights();
  InitializeRadius();
  SetInterpolationIndices();
  SetInterpolationWeights();
  return;
}

//----------------------------------------------------------------------------------------
//! \brief SphericalSurface destructor

SphericalSurface::~SphericalSurface() {}

void SphericalSurface::InitializeAngleAndWeights() {
  int n = 0;
  if (uniform_theta_) {
    // Uniform spacing in θ: θ_j = j * π / (ntheta - 1)
    Real dphi = 2.0 * M_PI / nphi;
    Real dtheta = M_PI / (ntheta - 1);
    for (int i = 0; i < nphi; ++i) {
      Real phi = dphi * i;
      for (int j = 0; j < ntheta; ++j) {
        Real theta = dtheta * j;
        int_weights.h_view(n) = dphi * std::sin(theta) * dtheta;
        polar_pos.h_view(n, 0) = theta;
        polar_pos.h_view(n, 1) = phi;
        n++;
      }
    }
  } else {
    // Uniform spacing in cos(θ) (default)
    for (int i = 0; i < nphi; ++i) {
      Real phi = 2.0 * M_PI / nphi * i;
      for (int j = 0; j < ntheta; ++j) {
        Real mu = -1.0 + 2.0 / (ntheta - 1) * j;
        int_weights.h_view(n) = (2.0 * M_PI / nphi) * (2.0 / ntheta);
        polar_pos.h_view(n, 0) = acos(mu);
        polar_pos.h_view(n, 1) = phi;
        n++;
      }
    }
  }

  polar_pos.template modify<HostMemSpace>();
  polar_pos.template sync<DevExeSpace>();

  int_weights.template modify<HostMemSpace>();
  int_weights.template sync<DevExeSpace>();
}

void SphericalSurface::InitializeRadius() {
  for (int r = 0; r < nradii; ++r) {
    Real &rad = radii.h_view(r);
    for (int n = 0; n < nangles; ++n) {
      Real &theta = polar_pos.h_view(n, 0);
      Real &phi = polar_pos.h_view(n, 1);
      int p = r * nangles + n;
      cart_pos.h_view(p, 0) = rad * cos(phi) * sin(theta) + xc;
      cart_pos.h_view(p, 1) = rad * sin(phi) * sin(theta) + yc;
      cart_pos.h_view(p, 2) = rad * cos(theta) + zc;
    }
  }
  cart_pos.template modify<HostMemSpace>();
  cart_pos.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalSurface::SetInterpolationIndices

void SphericalSurface::SetInterpolationIndices() {
  auto &size = pmy_pack->pmb->mb_size;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = npoints - 1;
  auto &rcoord = cart_pos;
  auto &iindcs = interp_indcs;
  // Even stencils anchor on last cell whose center is at/left of the point;
  // odd stencils anchor on the containing cell.
  Real offset = (ninterp % 2 == 0) ? -0.5 : 0.0;
  for (int n = 0; n <= nang1; ++n) {
    iindcs.h_view(n, 0) = -1;
    iindcs.h_view(n, 1) = -1;
    iindcs.h_view(n, 2) = -1;
    iindcs.h_view(n, 3) = -1;
    for (int m = 0; m <= nmb1; ++m) {
      Real &x1min = size.h_view(m).x1min;
      Real &x1max = size.h_view(m).x1max;
      Real &x2min = size.h_view(m).x2min;
      Real &x2max = size.h_view(m).x2max;
      Real &x3min = size.h_view(m).x3min;
      Real &x3max = size.h_view(m).x3max;

      Real &dx1 = size.h_view(m).dx1;
      Real &dx2 = size.h_view(m).dx2;
      Real &dx3 = size.h_view(m).dx3;

      // Half-open [xmin, xmax) claims boundary points for exactly one block.
      if ((rcoord.h_view(n, 0) >= x1min && rcoord.h_view(n, 0) < x1max) &&
          (rcoord.h_view(n, 1) >= x2min && rcoord.h_view(n, 1) < x2max) &&
          (rcoord.h_view(n, 2) >= x3min && rcoord.h_view(n, 2) < x3max)) {
        iindcs.h_view(n, 0) = m;
        iindcs.h_view(n, 1) = static_cast<int>(
            std::floor((rcoord.h_view(n, 0) - (x1min + offset * dx1)) / dx1));
        iindcs.h_view(n, 2) = static_cast<int>(
            std::floor((rcoord.h_view(n, 1) - (x2min + offset * dx2)) / dx2));
        iindcs.h_view(n, 3) = static_cast<int>(
            std::floor((rcoord.h_view(n, 2) - (x3min + offset * dx3)) / dx3));
        break;
      }
    }
  }

  interp_indcs.template modify<HostMemSpace>();
  interp_indcs.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalSurface::SetInterpolationWeights

void SphericalSurface::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  for (int n = 0; n < npoints; ++n) {
    int &ii0 = iindcs.h_view(n, 0);
    int &ii1 = iindcs.h_view(n, 1);
    int &ii2 = iindcs.h_view(n, 2);
    int &ii3 = iindcs.h_view(n, 3);

    if (ii0 == -1) {
      for (int i = 0; i < ninterp; ++i) {
        iwghts.h_view(n, i, 0) = 0.0;
        iwghts.h_view(n, i, 1) = 0.0;
        iwghts.h_view(n, i, 2) = 0.0;
      }
    } else {
      Real &x0 = cart_pos.h_view(n, 0);
      Real &y0 = cart_pos.h_view(n, 1);
      Real &z0 = cart_pos.h_view(n, 2);

      Real &x1min = size.h_view(ii0).x1min;
      Real &x1max = size.h_view(ii0).x1max;
      Real &x2min = size.h_view(ii0).x2min;
      Real &x2max = size.h_view(ii0).x2max;
      Real &x3min = size.h_view(ii0).x3min;
      Real &x3max = size.h_view(ii0).x3max;

      int nleft = ninterp / 2;
      for (int i = 0; i < ninterp; ++i) {
        iwghts.h_view(n, i, 0) = 1.;
        iwghts.h_view(n, i, 1) = 1.;
        iwghts.h_view(n, i, 2) = 1.;
        Real x1vpi1 = CellCenterX(ii1 - nleft + i, indcs.nx1, x1min, x1max);
        Real x2vpi1 = CellCenterX(ii2 - nleft + i, indcs.nx2, x2min, x2max);
        Real x3vpi1 = CellCenterX(ii3 - nleft + i, indcs.nx3, x3min, x3max);
        for (int j = 0; j < ninterp; ++j) {
          if (j != i) {
            Real x1vpj1 = CellCenterX(ii1 - nleft + j, indcs.nx1, x1min, x1max);
            iwghts.h_view(n, i, 0) *= (x0 - x1vpj1) / (x1vpi1 - x1vpj1);
            Real x2vpj1 = CellCenterX(ii2 - nleft + j, indcs.nx2, x2min, x2max);
            iwghts.h_view(n, i, 1) *= (y0 - x2vpj1) / (x2vpi1 - x2vpj1);
            Real x3vpj1 = CellCenterX(ii3 - nleft + j, indcs.nx3, x3min, x3max);
            iwghts.h_view(n, i, 2) *= (z0 - x3vpj1) / (x3vpi1 - x3vpj1);
          }
        }
      }
    }
  }

  interp_wghts.template modify<HostMemSpace>();
  interp_wghts.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphericalSurface::InterpolateToSphere

void SphericalSurface::InterpolateToSphere(int var_ind,
                                           DvceArray5D<Real> &val) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int nang1 = npoints - 1;
  int v = var_ind;
  int nintp = ninterp;
  int nleft = nintp / 2;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = interp_vals;
  par_for(
      "int2sph", DevExeSpace(), 0, nang1, KOKKOS_LAMBDA(int n) {
        int &ii0 = iindcs.d_view(n, 0);
        int &ii1 = iindcs.d_view(n, 1);
        int &ii2 = iindcs.d_view(n, 2);
        int &ii3 = iindcs.d_view(n, 3);

        if (ii0 == -1) {
          ivals.d_view(n) = 0.0;
        } else {
          Real int_value = 0.0;
          for (int i = 0; i < nintp; i++) {
            for (int j = 0; j < nintp; j++) {
              for (int k = 0; k < nintp; k++) {
                Real iwght = iwghts.d_view(n, i, 0) * iwghts.d_view(n, j, 1) *
                             iwghts.d_view(n, k, 2);
                int_value += iwght * val(ii0, v, ii3 + k + ks - nleft,
                                         ii2 + j + js - nleft,
                                         ii1 + i + is - nleft);
              }
            }
          }
          ivals.d_view(n) = int_value;
        }
      });

  interp_vals.template modify<DevExeSpace>();
  interp_vals.template sync<HostMemSpace>();

  return;
}
