//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_matrices.cpp
//  \brief implementation of the radiation FEM_N matrices for the GR radiation code
//         Functions in this file:
//         (a) IntegrateMatrixSphericalTriangle: integrate a given function over a spherical triangle
//         (b) IntegrateMatrixFEMN: Find integrals of a given function over unit sphere (FEM)
//         (c) IntegrateMatrixFPN: Find integrals of a given function over unit sphere (FP_N)
//         (d) CalculateDeterminantJacobian: Calculate determinant of Jacobian needed for mapping to/from angles to barycentric

#include <complex>
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

/* Integrate functions over a finite element spherical triangle element
 * Note: Modify this function to add new functions to integrate for FEM
 *
 * Choice of matrix is to be provided in matrixchoice:
 * [0] mass matrix: \int psi_a psi_b dOmega
 * [1] stiffness matrix x: \int cos(phi) sin(theta) psi_a psi_b dOmega
 * [2] stiffness matrix y: \int sin(phi) sin(theta) psi_a psi_b dOmega
 * [3] stiffness matrix z: \int cos(theta) psi_a psi_b dOmega
 * [4] G^nu^mu_ihat: \int p(1)^nu p(1)^mu \psi_a dpsi_b/dp^ihat dOmega
 * [5] F^nu^mu_ihat: \int p(1)^nu p(1)^mu \psi_a \psi_b p(1)_ihat dOmega
 * [6] \int psi_a dOmega
 * [7] \int cos(phi) sin(theta) psi_a dOmega
 * [8] \int sin(phi) sin(theta) psi_a dOmega
 * [9] \int cos(theta) psi_a dOmega
 *
 * Inputs:
 * ------
 * a: basis vector index (psi_a) [row index]
 * b: basis vector index (psi_b) [column index]
 * t1: index of triangle vertex [x1,y1,z1]
 * t2: index of triangle vertex [x2,y2,z2]
 * t3: index of triangle vertex [x3,y3,z3]
 * x: array of cartesian coordinate inside spherical triangle (x)
 * y: array of cartesian coordinate inside spherical triangle (y)
 * z: array of cartesian coordinate inside spherical triangle (z)
 * scheme_weights: array of quadrature weights
 * scheme_points: array of quadrature points
 * matrixchoice: choice of matrix
 * nu: optional index for p(1)^nu
 * mu: optional index for p(1)^mu
 * ihat: optional index for p(1)_ihat
 */
inline Real IntegrateMatrixSphericalTriangle(int a, int b, int basis, int t1, int t2, int t3,
                                             const HostArray1D<Real> &x, const HostArray1D<Real> &y, const HostArray1D<Real> &z,
                                             const HostArray1D<Real> &scheme_weights, const HostArray2D<Real> &scheme_points,
                                             int matrixnumber, int nu, int mu, int ihat) {

  Real x1 = x(t1);
  Real y1 = y(t1);
  Real z1 = z(t1);

  Real x2 = x(t2);
  Real y2 = y(t2);
  Real z2 = z(t2);

  Real x3 = x(t3);
  Real y3 = y(t3);
  Real z3 = z(t3);

  Real result = 0.;

  switch (matrixnumber) {

    case 0:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * fem_basis_ab(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    case 1:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * cos_phi_sin_theta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * fem_basis_ab(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    case 2:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * sin_phi_sin_theta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * fem_basis_ab(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    case 3:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * cos_theta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * fem_basis_ab(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    case 4:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * mom(nu, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * mom(mu, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * fem_basis_a(a, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis)
                * dfem_dpihat(ihat, b, t1, t2, t3, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    case 5:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * mom(nu, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * mom(mu, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * fem_basis_ab(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis)
                * mom(ihat, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2));
      }
      break;
    case 6:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * fem_basis_a(a, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    case 7:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * cos_phi_sin_theta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * fem_basis_a(a, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    case 8:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * sin_phi_sin_theta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * fem_basis_a(a, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    case 9:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result +=
            sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) * scheme_weights(i)
                * cos_theta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
                * fem_basis_a(a, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis);
      }
      break;
    default:result = -42.;
  }

  result = 0.5 * result;

  return result;
}

/* Integrate a function over the surface of a unit sphere (FP_N)
 * Note: Modify this function to integrate for FP_N
 *
 * Choice of matrix is to be provided in matrixchoice:
 * [0] mass matrix: \int psi_a psi_b dOmega
 * [1] stiffness matrix x: \int cos(phi) sin(theta) psi_a psi_b dOmega
 * [2] stiffness matrix y: \int sin(phi) sin(theta) psi_a psi_b dOmega
 * [3] stiffness matrix z: \int cos(theta) psi_a psi_b dOmega
 * [4] G^nu^mu_ihat: \int p(1)^nu p(1)^mu \psi_a dpsi_b/dp^ihat dOmega
 * [5] F^nu^mu_ihat: \int p(1)^nu p(1)^mu \psi_a \psi_b p(1)_ihat dOmega
 *
 * Inputs:
 * ------
 * la: l corresponding to psi_a [row index a = (la,ma)]
 * ma: m corresponding to psi_a [row index a = (la,ma)]
 * lb: l corresponding to psi_b [column index b = (lb,mb)]
 * mb: m corresponding to psi_b [column index b = (lb,mb)]
 * scheme_weights: quadrature weights
 * scheme_points: quadrature points
 * matrixchoice: choice of matrix
 * nu: optional index for p(1)^nu
 * mu: optional index for p(1)^mu
 * ihat: optional index for p(1)_ihat
 */
Real IntegrateMatrixFPN(int la, int ma, int lb, int mb, const HostArray1D<Real> &scheme_weights, const HostArray2D<Real> &scheme_points,
                        int matrixchoice, int nu, int mu, int ihat) {

  Real result = 0.;

  switch (matrixchoice) {

    case 0:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result += 4. * M_PI * scheme_weights(i)
            * fpn_basis_lm(la, ma, scheme_points(i, 0), scheme_points(i, 1))
            * fpn_basis_lm(lb, mb, scheme_points(i, 0), scheme_points(i, 1));
      }
      break;
    case 1:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result += 4. * M_PI * scheme_weights(i)
            * cos(scheme_points(i, 0)) * sin(scheme_points(i, 1))
            * fpn_basis_lm(la, ma, scheme_points(i, 0), scheme_points(i, 1))
            * fpn_basis_lm(lb, mb, scheme_points(i, 0), scheme_points(i, 1));
      }
      break;
    case 2:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result += 4. * M_PI * scheme_weights(i)
            * sin(scheme_points(i, 0)) * sin(scheme_points(i, 1))
            * fpn_basis_lm(la, ma, scheme_points(i, 0), scheme_points(i, 1))
            * fpn_basis_lm(lb, mb, scheme_points(i, 0), scheme_points(i, 1));
      }
      break;
    case 3:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result += 4. * M_PI * scheme_weights(i)
            * cos(scheme_points(i, 1))
            * fpn_basis_lm(la, ma, scheme_points(i, 0), scheme_points(i, 1))
            * fpn_basis_lm(lb, mb, scheme_points(i, 0), scheme_points(i, 1));
      }
      break;
    case 4:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        if (!(fabs(scheme_points(i, 0) - 0.) < 1e-14 || fabs(scheme_points(i, 0) - M_PI) < 1e-14)) { // basis derivatives vanish at 0 and pi
          result += 4. * M_PI * scheme_weights(i)
              * mom(nu, scheme_points(i, 0), scheme_points(i, 1))
              * mom(mu, scheme_points(i, 0), scheme_points(i, 1))
              * fpn_basis_lm(la, ma, scheme_points(i, 0), scheme_points(i, 1))
              * (inv_jac_itilde_ihat(scheme_points(i, 0), scheme_points(i, 1), 1, ihat) * dfpn_dOmega(lb, mb, scheme_points(i, 0), scheme_points(i, 1), 1)
                  + inv_jac_itilde_ihat(scheme_points(i, 0), scheme_points(i, 1), 2, ihat) * dfpn_dOmega(lb, mb, scheme_points(i, 0), scheme_points(i, 1), 2));
        }
      }
      break;
    case 5:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result += 4. * M_PI * scheme_weights(i)
            * mom(nu, scheme_points(i, 0), scheme_points(i, 1))
            * mom(mu, scheme_points(i, 0), scheme_points(i, 1))
            * fpn_basis_lm(la, ma, scheme_points(i, 0), scheme_points(i, 1))
            * fpn_basis_lm(lb, mb, scheme_points(i, 0), scheme_points(i, 1))
            * mom(ihat, scheme_points(i, 0), scheme_points(i, 1));
      }
      break;
    case 6:
      for (size_t i = 0; i < scheme_weights.size(); i++) {
        result += 4. * M_PI * scheme_weights(i)
            * fpn_basis_lm(la, ma, scheme_points(i, 0), scheme_points(i, 1));
      }
      break;
    default:result = -42.;

  }

  return result;
}

// Calculate determinant of Jacobian term for FEM matrix computation
inline Real CalculateDeterminantJacobian(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {

  Real result =
      pow(x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3, 2) /
          pow(x1 * x1 * xi1 * xi1 + 2. * x1 * x2 * xi1 * xi2 + x2 * x2 * xi2 * xi2 +
              x3 * x3 * (-1. + xi1 + xi2) * (-1. + xi1 + xi2) -
              2. * x3 * (-1. + xi1 + xi2) * (x1 * xi1 + x2 * xi2) + xi1 * xi1 * y1 * y1 +
              2. * xi1 * xi2 * y1 * y2 + xi2 * xi2 * y2 * y2 + 2. * xi1 * y1 * y3 - 2. * xi1 * xi1 * y1 * y3 -
              2. * xi1 * xi2 * y1 * y3 + 2. * xi2 * y2 * y3 - 2. * xi1 * xi2 * y2 * y3 -
              2. * xi2 * xi2 * y2 * y3 + y3 * y3 - 2. * xi1 * y3 * y3 + xi1 * xi1 * y3 * y3 - 2. * xi2 * y3 * y3 +
              2. * xi1 * xi2 * y3 * y3 + xi2 * xi2 * y3 * y3 + xi1 * xi1 * z1 * z1 + 2. * xi1 * xi2 * z1 * z2 +
              xi2 * xi2 * z2 * z2 - 2. * (-1. + xi1 + xi2) * (xi1 * z1 + xi2 * z2) * z3 +
              (-1. + xi1 + xi2) * (-1. + xi1 + xi2) * z3 * z3, 3);

  return result;
}

// ---------------------------------------------------------------
// Find mass/stiffness and other associated matrices for FEM basis
/* Compute matrices for FEM
 *
 * Inputs:
 * ------
 * a: basis vector index (psi_a) [row index]
 * b: basis vector index (psi_b) [column index]
 * basis: choice of basis function
 * x: array of cartesian coordinate inside spherical triangle (x)
 * y: array of cartesian coordinate inside spherical triangle (y)
 * z: array of cartesian coordinate inside spherical triangle (z)
 * scheme_weights: array of quadrature weights
 * scheme_points: array of quadrature points
 * triangles: triangle information
 * matrixchoice: choice of matrix
 * nu: optional index for p(1)^nu
 * mu: optional index for p(1)^mu
 * ihat: optional index for p(1)_ihat
 */
Real IntegrateMatrixFEMN(int a, int b, int basis, const HostArray1D<Real> &x, const HostArray1D<Real> &y, const HostArray1D<Real> &z,
                         const HostArray1D<Real> &scheme_weights, const HostArray2D<Real> &scheme_points, const HostArray2D<int> &triangles,
                         int matrixchoice, int nu, int mu, int ihat) {

  Real result = 0.;

  bool is_edge{false};
  HostArray2D<int> edge_triangles;
  FindTriangles(a, b, triangles, edge_triangles, is_edge);

  if (is_edge) {
    for (size_t i = 0; i < 6; i++) {
      if (edge_triangles(i, 0) >= 0) {
        int triangle_index_1 = edge_triangles(i, 0);
        int triangle_index_2 = edge_triangles(i, 1);
        int triangle_index_3 = edge_triangles(i, 2);

        Real integrated_result =
            IntegrateMatrixSphericalTriangle(a, b, basis, triangle_index_1, triangle_index_2, triangle_index_3, x, y, z,
                                             scheme_weights, scheme_points, matrixchoice, nu, mu, ihat);
        result += integrated_result;
      }
    }
  }
  return result;
}

} // namespace radiationfemn