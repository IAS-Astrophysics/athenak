#ifndef UTILS_SPHERICAL_SURFACE_HPP_
#define UTILS_SPHERICAL_SURFACE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_surface.hpp
//  \brief definitions for the SphericalSurface class

#include "athena.hpp"
#include "athena_tensor.hpp"

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class SphericalSurface

class SphericalSurface {
 public:
  // ninterp: interpolation points per axis (SphericalGrid convention);
  // <=0 selects the default full stencil of 2*ng points (max 2*ng+1).
  SphericalSurface(MeshBlockPack *pmy_pack, int ntheta, Real rad, Real xc = 0.0,
                   Real yc = 0.0, Real zc = 0.0, int nphi = -1,
                   bool uniform_theta = false, int ninterp = -1);
  ~SphericalSurface();
  int nangles;  // total number of gridpoints (= nphi * ntheta)
  int ntheta;   // number of gridpoints along theta direction
  int nphi;     // number of gridpoints along phi direction
  int ninterp;  // number of interpolation points along each dimension
  Real radius;  // radius to initialize the sphere
  Real xc, yc, zc;                // sphere center
  DualArray1D<Real> int_weights;  // weights for quadrature integration
  DualArray2D<Real> cart_pos;     // coord position (cartesian) at gridpoints
  DualArray2D<Real> polar_pos;
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
  bool uniform_theta_;      // true → uniform θ spacing; false → uniform cos(θ)
};
#endif  // UTILS_SPHERICAL_SURFACE_HPP_
