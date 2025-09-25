#ifndef GEODESIC_GRID_SPHERICAL_GRID_HPP_
#define GEODESIC_GRID_SPHERICAL_GRID_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.hpp
//  \brief definitions for SphericalGrid class

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class SphericalGrid

class SphericalGrid: public GeodesicGrid {
 public:
    // Creates a geodesic grid with refinement level nlev and radius rad
    SphericalGrid(MeshBlockPack *pmy_pack, int nlev, Real rad);
    ~SphericalGrid();

    Real radius;  // const radius for SphericalGrid
    DualArray2D<Real> interp_coord;  // Cartesian coordinates for grid points
    DualArray2D<Real> interp_vals;   // container for data interpolated to sphere
    void InterpolateToSphere(int nvars, DvceArray5D<Real> &val);  // interpolate to sphere

 private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
    DualArray2D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
    DualArray3D<Real> interp_wghts;  // weights for interpolation
    void SetInterpolationCoordinates();  // set indexing for interpolation
    void SetInterpolationIndices();      // set indexing for interpolation
    void SetInterpolationWeights();      // set weights for interpolation
};

#endif // GEODESIC_GRID_SPHERICAL_GRID_HPP_
