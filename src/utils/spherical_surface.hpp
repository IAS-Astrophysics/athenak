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
//! \brief layout of the polar angle grid of a SphericalSurface
//! uniform_theta:    points equally spaced in theta, i.e. uniform angular resolution
//! uniform_costheta: points equally spaced in cos(theta), i.e. uniform in solid angle
//!                   but with poor angular resolution near the poles
enum class SphThetaSpacing {uniform_theta, uniform_costheta};

//----------------------------------------------------------------------------------------
//! \brief placement of the polar angle grid points relative to their cells
//! cell: points sit at cell centers, the first one half a cell away from the pole, so
//!       that no point lies on the z-axis (and the grid can be mirrored across it)
//! node: points sit at cell edges, so the first and last point lie on the z-axis
enum class SphThetaCentering {cell, node};

//----------------------------------------------------------------------------------------
//! \brief layout of the angular grid of a SphericalSurface. Defaults to the node centered
//! grid uniform in cos(theta) with nphi = 2*ntheta that the class has always used; the
//! cell centered grid uniform in theta is opt-in.
struct SphericalSurfaceAngles {
  SphThetaSpacing theta_spacing = SphThetaSpacing::uniform_costheta;
  SphThetaCentering theta_centering = SphThetaCentering::node;
  int nphi = -1;  // number of gridpoints along phi, any value < 0 selects 2*ntheta
};

//----------------------------------------------------------------------------------------
//! \class SphericalSurface
//! \brief One or more spherical surfaces sharing a common grid. The surface
//! contains npoints = nradii*nangles. Point ir*nangles+n is angle n on shell ir.
class SphericalSurface {
 public:
  using AngleOptions = SphericalSurfaceAngles;

  // Single-radius surface
  SphericalSurface(MeshBlockPack *pmy_pack, int ntheta, Real rad, Real xc = 0.0,
                   Real yc = 0.0, Real zc = 0.0,
                   const AngleOptions &aopt = AngleOptions());
  // Multi-radii surface
  SphericalSurface(MeshBlockPack *pmy_pack, int ntheta, const std::vector<Real> &rad,
                   Real xc = 0.0, Real yc = 0.0, Real zc = 0.0,
                   const AngleOptions &aopt = AngleOptions());
  ~SphericalSurface();
  int nangles;  // total number of gridpoints per surface, ntheta*nphi
  int ntheta;   // number of gridpoints along theta direction
  int nphi;     // number of gridpoints along phi direction
  int nradii;   // number of surfaces
  int npoints;  // total number of grid points, nradii*nangles
  SphThetaSpacing theta_spacing;      // layout of the theta grid
  SphThetaCentering theta_centering;  // placement of the theta gridpoints
  DualArray1D<Real> radii;        // radii of the surfaces
  Real xc, yc, zc;                // sphere center
  DualArray1D<Real> int_weights;  // solid angle element dOmega (per angle), sums to 4pi
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
  void UpdateInterpolationOnMeshChange();  // redo both if AMR moved the mesh

 private:
  int amr_nmb_created;      // mesh stamp the interpolation indices were built against
  int amr_nmb_deleted;
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};
#endif  // UTILS_SPHERICAL_SURFACE_HPP_
