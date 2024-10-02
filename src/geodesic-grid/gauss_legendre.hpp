#ifndef GEODESIC_GRID_GAUSSLEGENDRA_GRID_HPP_
#define GEODESIC_GRID_GAUSSLEGENDRA_GRID_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geodesic_grid.hpp
//  \brief definitions for GaussLegendreGrid class

#include "athena.hpp"
#include "athena_tensor.hpp"

//----------------------------------------------------------------------------------------
//! \class GaussLegendreGrid

class GaussLegendreGrid {
 public:
  GaussLegendreGrid(MeshBlockPack *pmy_pack, int ntheta, Real rad, int nfilt);
  ~GaussLegendreGrid();
    int nangles;  // total number of gridpoints
    int ntheta;  // number of gridpoints along theta direction, nphi = 2ntheta
    int nfilt;   // number of filtering
    Real radius; // radius to initialize the sphere
    DualArray1D<Real> int_weights;         // weights for quadrature integration
    DualArray2D<Real> cart_pos;             // coord position (cartesian) at gridpoints
    DualArray1D<Real> theta;            // theta coordinates at gridpoints
    DualArray1D<Real> phi;            // phi coordinates at gridpoints
    DualArray2D<Real> polar_pos;
    DualArray1D<Real> pointwise_radius;
    DualArray3D<Real> basis_functions;
    DualArray3D<Real> surface_jacobian;
    DualArray4D<Real> d_surface_jacobian;
    DualArray2D<Real> interp_vals;   // container for data interpolated to sphere

    // functions
    void InitializeAngleAndWeights();
    void InitializeRadius();

    void SetPointwiseRadius(DualArray1D<Real> rad_tmp, Real ctr[3]);  // set indexing for interpolation
    void EvaluateSphericalHarm();
    void EvaluateTangentVectors();
    void EvaluateNormalOneForms();
    void EvaluateSurfaceJacobian();
    void EvaluateSurfaceJacobianDerivative();
    Real Integrate(DualArray1D<Real> integrand);
    Real Integrate(AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> integrand);

    DualArray1D<Real> ThetaDerivative(DualArray1D<Real> scalar_function);
    DualArray1D<Real> PhiDerivative(DualArray1D<Real> scalar_function);
    DualArray1D<Real> SpatialToSpectral(DualArray1D<Real> scalar_function);
    DualArray1D<Real> SpatialToSpectral(AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> scalar_function);
    DualArray1D<Real> SpectralToSpatial(DualArray1D<Real> scalar_spectrum);
    void InterpolateToSphere(int nvars, DvceArray5D<Real> &val);  // interpolate scalar field to sphere
    // Interpolate rank 2 tensors to sphere
    AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,2> InterpolateToSphere(AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd);
    // Interpolate rank 3 tensors to sphere
    AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,3> InterpolateToSphere(AthenaTensor<Real, TensorSymm::SYM2, 3, 3> &d_g_ddd);
    DualArray2D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
    DualArray3D<Real> interp_wghts;  // weights for interpolation

    void SetInterpolationCoordinates();  // set indexing for interpolation
    void SetInterpolationIndices();      // set indexing for interpolation
    void SetInterpolationWeights();      // set weights for interpolation

 private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

#endif // GEODESIC_GRID_GAUSSLEGENDRA_GRID_HPP_