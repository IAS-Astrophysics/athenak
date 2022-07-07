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
#include "geodesic-grid/geodesic_grid.hpp"

//! \class SphericalGrid
//! \brief A class representing a grid on a topological sphere (wrapping around GeodesicGrid)
class SphericalGrid: public GeodesicGrid {
  public:
    // Creates a geodetic grid with nlev levels and radius rad
    SphericalGrid(MeshBlockPack *pmbp, int nlev, Real ctr_[3], bool rotate_g, bool fluxes);
    Real ctr[3];                       // center of the sphere
    DualArray1D<Real> radius;           // radius of the sphere
    DualArray2D<Real> polarcoord;
    DualArray2D<Real> cartcoord;        // cartesian coordinate for grid points
    DualArray2D<int> interp_indices;    // indices of meshblock and the cell with in, for interpolation
    DualArray1D<Real> area;             // area for each face, for integration
    DualArray1D<Real> intensity;        // values to be stored on a geodesic mesh
    void SetRadius(DualArray1D<Real>);  // set radius of the star-shaped region
    void SetRadius(Real rad_);          // set constant radius for sphere

    void CalculateIndex();              // calculate the interpolation index
    void InterpToSphere(DvceArray5D<Real> &value);   // interpolating a scalar field onto SpericalGrid
  private:
    MeshBlockPack* pmbp;  // ptr to MeshBlockPack containing this Hydro
};
#endif