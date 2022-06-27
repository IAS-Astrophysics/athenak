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
//#include "utils/lagrange_interp.hpp"
#include "mesh/mesh.hpp"
#include "mesh/geodesic_grid.hpp"

//! \class SphericalGrid
//! \brief A class representing a grid on a topological sphere (wrapping around GeodesicGrid)
class SphericalGrid: public GeodesicGrid {
  public:
    // Creates a geodetic grid with nlev levels and radius rad
    SphericalGrid(MeshBlockPack *ppack, int *nlev, bool *rotate_g, Real rad_ = 1.0);
    Real rad;                           // radius of a sphere
    DualArray2D<Real> cartcoord;        // cartesian coordinate for grid points
    DualArray2D<int> interp_indices;    // indices of meshblock and the cell with in, for interpolation
};
#endif

//! \class SphericalPatch
//! \brief This class represents the intersection between a spherical grid and a mesh block
//!  Note: this class assumes that the MeshBlock is Cartesian uniformly spaced
/*
class SphericalPatch {
  public:
    enum collocation_t {cell, vertex};
  public:
    SphericalPatch(SphericalGrid const * psphere, MeshBlockPack const *ppack, collocation_t coll);
    ~SphericalPatch();
    // Interpolate a group of arrays defined on the MeshBlock to the SphericalPatch
    //  The destination array should be allocated
    void InterpToSpherical(DvceArray4D<Real> const & src,    //! [in] data defined on the MeshBlock
                           DvceArray4D<Real> * dst) const;   //! [out] data defined on the SphericalGrid
    //! Merge data into a global array defined on the whole sphere
    void MergeData(DvceArray4D<Real> const & src,            //! [in] data defined on the SphericalPatch
                   DvceArray4D<Real> * dst) const;           //! [out] data defined on the SphericalGrid
    //! Number of points on the spherical patch
    inline int NumPoints() const {
      return n;
    }
    //! Map patch degrees of freedom to the corresponding index in the full SphericalGrid
    inline int idxMap(int idx) const {
      return map[idx];
    }
  private:
    //! Interpolate an array defined on the MeshBlock to the SphericalPatch
    //  The destination array should be allocated
    void interpToSpherical(Real const * src,    //! [in] 1D data defined on the MeshBlock
                           Real * dst) const;   //! [out] 1D data defined on the SphericalGrid
    //! Merge data arrays into a global arrays defined on the whole sphere
    void mergeData(Real const * src,            //! [in] 1D data defined on the SphericalPatch
                   Real * dst) const;           //! [out] 1D data defined on the SphericalGrid
  public:
    //! Type of collocation
    collocation_t const coll;
    //! Parent spherical grid
    SphericalGrid const * psphere;
    // Parent mesh block
    MeshBlockPack const * ppack;
  private:
    // Number of points in the spherical patch
    int n;
    // Maps local indices to global indices on the SphericalGrid
    std::vector<int> map;
    // Interpolating polynomials
    LagrangeInterpND<2*NGHOST-1, 3> ** pinterp;
};
*/
