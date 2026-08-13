#ifndef UTILS_SPHERICAL_SURFACE_HPP_
#define UTILS_SPHERICAL_SURFACE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geodesic_grid.hpp
//  \brief definitions for GaussLegendreGrid class

#include <vector>

#include "athena.hpp"
#include "athena_tensor.hpp"

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class SphericalSurface
//! \brief One or more spherical surfaces sharing a common grid. The surface
//! contains npoints = nradii*nangles. Point ir*nangles+n is angle n on shell ir.
class SphericalSurface {
 public:
  // Single-radius surface
  SphericalSurface(MeshBlockPack *pmy_pack, int ntheta, Real rad, Real xc = 0.0,
                   Real yc = 0.0, Real zc = 0.0);
  // Multi-radii surface
  SphericalSurface(MeshBlockPack *pmy_pack, int ntheta, const std::vector<Real> &rad,
                   Real xc = 0.0, Real yc = 0.0, Real zc = 0.0);
  ~SphericalSurface();
  int nangles;  // total number of gridpoints per surface
  int ntheta;   // number of gridpoints along theta direction, nphi = 2ntheta
  int nradii;   // number of surfaces
  int npoints;  // total number of grid points, nradii*nangles
  DualArray1D<Real> radii;        // radii of the surfaces
  Real xc, yc, zc;                // sphere center
  DualArray1D<Real> int_weights;  // weights for quadrature integration (per angle)
  DualArray2D<Real> cart_pos;     // coord position (cartesian) at gridpoints
  DualArray2D<Real> polar_pos;    // (theta,phi) at gridpoints (per angle)
  DualArray1D<Real> interp_vals;  // container for data interpolated to sphere

  // functions
  void InitializeAngleAndWeights();
  void InitializeRadius();

  // interpolate scalar field to sphere
  void InterpolateToSphere(int nvars, DvceArray5D<Real> &val);
  DualArray2D<int>
      interp_indcs;  // indices of MeshBlock and zones therein for interp
  DualArray3D<Real> interp_wghts;  // weights for interpolation

  void SetInterpolationCoordinates();  // set indexing for interpolation
  void SetInterpolationIndices();      // set indexing for interpolation
  void SetInterpolationWeights();      // set weights for interpolation

 private:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};
#endif  // UTILS_SPHERICAL_SURFACE_HPP_
