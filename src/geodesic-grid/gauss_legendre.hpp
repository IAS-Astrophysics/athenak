#ifndef GEODESIC_GRID_GAUSS_LEGENDRE_HPP_
#define GEODESIC_GRID_GAUSS_LEGENDRE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geodesic_grid.hpp
//  \brief definitions for GaussLegendreGrid class

#include "athena.hpp"
#include "athena_tensor.hpp"

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class GaussLegendreGrid

class GaussLegendreGrid {
 public:
  GaussLegendreGrid(MeshBlockPack *pmy_pack, int ntheta, Real rad);
  ~GaussLegendreGrid();
    int nangles;  // total number of gridpoints
    int ntheta;  // number of gridpoints along theta direction, nphi = 2ntheta
    Real radius; // radius to initialize the sphere
    DualArray1D<Real> int_weights;         // weights for quadrature integration
    DualArray2D<Real> cart_pos;             // coord position (cartesian) at gridpoints
    DualArray2D<Real> polar_pos;
    DualArray1D<Real> interp_vals;   // container for data interpolated to sphere

    // functions
    void InitializeAngleAndWeights();
    void InitializeRadius();

    // interpolate scalar field to sphere
    void InterpolateToSphere(int nvars, DvceArray5D<Real> &val);
    DualArray2D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
    DualArray3D<Real> interp_wghts;  // weights for interpolation

    void SetInterpolationCoordinates();  // set indexing for interpolation
    void SetInterpolationIndices();      // set indexing for interpolation
    void SetInterpolationWeights();      // set weights for interpolation

 private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};
#endif // GEODESIC_GRID_GAUSS_LEGENDRE_HPP_
