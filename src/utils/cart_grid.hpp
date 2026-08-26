#ifndef UTILS_CART_GRID_HPP_
#define UTILS_CART_GRID_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cart_grid.hpp
//  \brief definitions for SphericalGrid class

#include "athena.hpp"

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class CartesianGrid

class CartesianGrid {
 public:
  // Creates a geodesic grid with refinement level nlev and radius rad.
  // Set bitant_mirror to look points with z<0 up via their z-reflected counterpart on a
  // bitant mesh (see bitant() below). It is opt-in because the value returned is the raw
  // value at the mirrored point: only a caller that applies the reflection parity of the
  // quantity it is interpolating gets a correct answer. A caller that does not should
  // leave this false, so that points outside the domain stay zero rather than silently
  // acquiring the wrong sign.
  CartesianGrid(MeshBlockPack *pmy_pack, Real center[3],
                Real extend[3], int numpoints[3], bool is_cheb = false,
                bool bitant_mirror = false);

  // parameters for the grid
  Real center_x1, center_x2, center_x3;   // grid centers
  Real min_x1, min_x2, min_x3;            // min for xyz
  Real max_x1, max_x2, max_x3;            // max value for xyz
  Real d_x1, d_x2, d_x3;                     // resolution
  int nx1, nx2, nx3;                      // number of points
  Real extent_x1, extent_x2, extent_x3;

  // dump on chebyshev or uniform grid, default is uniform
  bool is_cheby;

  // For simplicity, unravel all points into a 1d array
  DualArray3D<Real> interp_vals;   // container for data interpolated to sphere
  void InterpolateToGrid(int nvars, DvceArray5D<Real> &val);  // interpolate to sphere
  void ResetCenter(Real center[3]);  // set indexing for interpolation
  void SetInterpolationIndices();      // set indexing for interpolation
  void SetInterpolationWeights();      // set weights for interpolation
  void ResetCenterAndExtent(Real center[3], Real extent[3]);

  // true if bitant_mirror was requested AND the mesh really is bitant (reflect at
  // x3min=0). Points with z<0 are then looked up via their z-reflected counterpart, which
  // physically lies inside the domain. The value returned by InterpolateToGrid is the raw
  // mirrored-point value, so the caller must apply the reflection parity appropriate to
  // the quantity being interpolated.
  bool bitant() const { return bitant_; }

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  bool bitant_;                    // whether the mesh is bitant about x3=0
  DualArray4D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
  DualArray5D<Real> interp_wghts;  // weights for interpolation
};

#endif // UTILS_CART_GRID_HPP_
