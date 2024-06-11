//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_geodesic_grid.hpp
//  \brief FEM/FPN grid and matrices

#ifndef ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP
#define ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP

#include "athena.hpp"

namespace radiationfemn
{
    // quadrature and geodesic grid functions
    void LoadQuadrature(std::string& scheme_name, int& scheme_num_points, HostArray1D<Real>& scheme_weights, HostArray2D<Real>& scheme_points);
    void GeodesicGridBaseGenerate(int& geogrid_level, int& geogrid_num_points, int& geogrid_num_edges, int& geogrid_num_triangles,
                                  HostArray1D<Real>& x, HostArray1D<Real>& y, HostArray1D<Real>& z, HostArray1D<Real>& r, HostArray1D<Real>& theta,
                                  HostArray1D<Real>& phi, HostArray2D<int>& edges, HostArray2D<int>& triangles);
    void GeodesicGridRefine(int& geogrid_level, int& geogrid_num_points, int& geogrid_num_edges, int& geogrid_num_triangles,
                            HostArray1D<Real>& x, HostArray1D<Real>& y, HostArray1D<Real>& z, HostArray1D<Real>& r, HostArray1D<Real>& theta,
                            HostArray1D<Real>& phi, HostArray2D<int>& edges, HostArray2D<int>& triangles);
    int FindEdgesIndex(int e1, int e2, HostArray2D<int>& edges); // Given two edge indices, find
    void FindTriangles(int a, int b, const HostArray2D<int>& triangles, HostArray2D<int>& edge_triangles, bool& is_edge);

    // basis functions and derivatives
    Real fem_overtent_index1(Real xi1, Real xi2, Real xi3);
    Real fem_overtent_index2(Real xi1, Real xi2, Real xi3);
    Real fem_overtent_index3(Real xi1, Real xi2, Real xi3);

    Real fem_smalltent_index1(Real xi1, Real xi2, Real xi3);
    Real fem_smalltent_index2(Real xi1, Real xi2, Real xi3);
    Real fem_smalltent_index3(Real xi1, Real xi2, Real xi3);

    Real fem_overhoney_index1(Real xi1, Real xi2, Real xi3);
    Real fem_overhoney_index2(Real xi1, Real xi2, Real xi3);
    Real fem_overhoney_index3(Real xi1, Real xi2, Real xi3);

    Real fem_smallhoney_index1(Real xi1, Real xi2, Real xi3);
    Real fem_smallhoney_index2(Real xi1, Real xi2, Real xi3);
    Real fem_smallhoney_index3(Real xi1, Real xi2, Real xi3);

    Real fem_basis(Real xi1, Real xi2, Real xi3, int basis_index, int basis_choice);
    Real fem_basis_ab(int a, int b, int t1, int t2, int t3, Real xi1, Real xi2, Real xi3, int basis_choice);
    Real fem_basis_a(int a, int t1, int t2, int t3, Real xi1, Real xi2, Real xi3, int basis_choice);
    Real dfem_dpihat(int ihat, int a, int t1, int t2, int t3, Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3,
                     Real xi1, Real xi2, Real xi3, int basis_choice);

    Real fpn_basis_lm(int l, int m, Real phi, Real theta);
    Real fpn_basis_lm_alt(int l, int m, Real phi, Real theta);
    Real dfpn_dtheta(int l, int m, Real phi, Real theta);
    Real cosec_dfpn_dphi(int l, int m, Real phi, Real theta);
    Real dfpn_dOmega(int l, int m, Real phi, Real theta, int var_index);
    Real inv_jac_itilde_ihat(Real phi, Real theta, int tilde_index, int hat_index);

    Real legendre(int l, int m, Real x);
    Real legendre_factor(int l, int m);
    Real recurrence_legendre(int l, int m, Real x);
    Real recurrence_legendre_alt(int l, int m, Real x);
    Real recurrence_derivative_legendre(Real l, Real m, Real x);

    // helper functions and other functions over geodesic grid
    void BarycentricToCartesian(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3,
                                Real& xval, Real& yval, Real& zval);
    Real cos_phi_sin_theta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
    Real sin_phi_sin_theta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
    Real cos_theta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
    Real mom(int mu, Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
    Real mom(int mu, Real phi, Real theta);

    // functions for integration over geodesic grid
    Real CalculateDeterminantJacobian(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3);
    Real IntegrateMatrixSphericalTriangle(int a, int b, int basis, int t1, int t2, int t3, const HostArray1D<Real>& x, const HostArray1D<Real>& y, const HostArray1D<Real>& z,
                                          const HostArray1D<Real>& scheme_weights, const HostArray2D<Real>& scheme_points, int matrixnumber, int nu = -42, int mu = -42,
                                          int ihat = -42);
    Real IntegrateMatrixFEMN(int a, int b, int basis, const HostArray1D<Real>& x, const HostArray1D<Real>& y, const HostArray1D<Real>& z, const HostArray1D<Real>& scheme_weights,
                             const HostArray2D<Real>& scheme_points, const HostArray2D<int>& triangles, int matrixchoice, int nu, int mu, int ihat);
    Real IntegrateMatrixFPN(int la, int ma, int lb, int mb, const HostArray1D<Real>& scheme_weights, const HostArray2D<Real>& scheme_points,
                            int matrixchoice, int nu, int mu, int ihat);
} // namespace radiationfemn

#endif //ATHENA_RADIATION_FEMN_GEODESIC_GRID_MATRICES_HPP
