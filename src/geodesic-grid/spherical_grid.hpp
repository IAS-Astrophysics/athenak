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
    // Creates a geodesic grid with refinement level nlev and radius rad.
    // ng_interp controls the Lagrange stencil half-width per axis (full stencil = 2×ng_interp points):
    //   ng_interp < 0 : default — use full mesh ghost-zone depth (original behaviour, e.g. ng=4 → 8-point, 7th-order)
    //   ng_interp = 0 : nearest-cell (1 point, fastest, strictly monotone)
    //   ng_interp = 1 : trilinear (2 points per axis, monotone)
    //   ng_interp = 2 : cubic Lagrange (4 points per axis)
    //   ng_interp = 4 : 7th-order Lagrange (8 points per axis, same as original ng=4 default)
    SphericalGrid(MeshBlockPack *pmy_pack, int nlev, Real rad, int ng_interp = -1);
    ~SphericalGrid();

    Real radius;  // const radius for SphericalGrid
    DualArray2D<Real> interp_coord;  // Cartesian coordinates for grid points
    DualArray2D<Real> interp_vals;   // container for data interpolated to sphere
    void InterpolateToSphere(int nvars, DvceArray5D<Real>& val);  // interpolate to sphere
    // interpolate a range of variables to a sphere
    void InterpolateToSphere(int vs, int ve, DvceArray5D<Real>& val);
    // Public methods for updating interpolation when mesh changes (e.g., AMR)
    void SetInterpolationIndices();
    // Set Lagrange weights for a given stencil half-width (ng_interp > 0) or
    // nearest-cell mode (ng_interp = 0). Default -1 uses the stored ng_interp_.
    void SetInterpolationWeights(int ng_interp = -1);

 private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
    DualArray2D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
    DualArray3D<Real> interp_wghts;  // weights for interpolation
    int ng_interp_;                  // effective stencil half-width (0 = nearest-cell)
    void SetInterpolationCoordinates();  // set interpolation coordinates
};

#endif // GEODESIC_GRID_SPHERICAL_GRID_HPP_
