#ifndef UTILS_ARBITRARY_GRID_INTERPOLTOR_HPP_
#define UTILS_ARBITRARY_GRID_INTERPOLTOR_HPP_

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
//! \class ArbitraryGrid

class ArbitraryGrid {
 public:
  // Creates a geodesic grid with refinement level nlev and radius rad
  ArbitraryGrid(MeshBlockPack *pmy_pack, std::vector<std::array<Real,3>>& cart_coord);

  // parameters for the grid
  int npts;                               // number of points
  std::vector<std::array<Real,3>> cart_coord; // cartesian coordinate for 

  // For simplicity, unravell all points into a 1d array
  DualArray1D<Real> interp_vals;   // container for data interpolated to sphere
  void InterpolateToGrid(int nvars, DvceArray5D<Real> &val);  // interpolate to sphere
  void ResetGrid(std::vector<std::array<Real,3>>& cart_coord);  // set indexing for interpolation
  void SetInterpolationIndices();      // set indexing for interpolation
  void SetInterpolationWeights();      // set weights for interpolation

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  DualArray2D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
  DualArray3D<Real> interp_wghts;  // weights for interpolation
};

#endif // UTILS_ARBITRARY_GRID_INTERPOLTOR_HPP_
