#ifndef SPHERICAL_GRID_HPP_
#define SPHERICAL_GRID_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.hpp
//  \brief Initializes angular mesh and orthonormal tetrad

#include <vector>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/geodesic_grid.hpp"

//! \class SphericalGrid
//! \brief A class representing a grid on a topological sphere (wrapping around GeodesicGrid)
class SphericalGrid: public GeodesicGrid {
  public:
    // Creates a geodetic grid with nlev levels and radius rad
    SphericalGrid(MeshBlockPack *ppack, int *nlev, bool *rotate_g, Real rad_ = 1.0, Real ctr_[3] = 0);
    Real rad;                           // radius of a sphere
    DualArray2D<Real> cartcoord;        // cartesian coordinate for grid points
    DualArray2D<int> interp_indices;    // indices of meshblock and the cell with in, for interpolation
};
#endif