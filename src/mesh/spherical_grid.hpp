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
    SphericalGrid(MeshBlockPack *ppack, int *nlev, bool *rotate_g, Real ctr_[3] = 0);
    Real ctr[3];                       // center of the sphere
    DualArray1D<Real> radius;           // radius of the sphere
    DualArray2D<Real> cartcoord;        // cartesian coordinate for grid points
    DualArray2D<int> interp_indices;    // indices of meshblock and the cell with in, for interpolation
    DualArray1D<Real> area;             // area for each face, for integration
    void SetRadius(DualArray1D<Real>);  // set radius of the star-shaped region
    void SetRadius(Real rad_);          // set constant radius for sphere

    void CalculateIndex();              // calculate the interpolation index
  private:
    MeshBlockPack* pmbp;  // ptr to MeshBlockPack containing this Hydro
};
#endif