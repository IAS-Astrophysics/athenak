//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_matrices.cpp
//  \brief implementation of the radiation FEM_N matrices for the GR radiation code
//         Functions in this file:
//         (a) IntegrateMatrixSphericalTriangle: integrate a given function over a spherical triangle
//         (b) CalculateDeterminantJacobian: Calculate determinant of Jacobian needed for mapping to/from angles to barycentric
//         (c) IntegrateMatrixFEMN: Find the integrals over all angles for a given function

#include <cassert>
#include <complex>
#include "athena.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

/* Top level function to integrate functions over a finite element spherical triangle element
 *
 * Choice of matrix is to be provided in matrixchoice:
 * [0] Psi_A Psi_B: mass matrix
 * [1] Cos Phi Sin Theta Psi_A Psi_B: stiffness matrix x
 * [2] Sin Phi Sin Theta Psi_A Psi_B: stiffness matrix y
 * [3] Cos Theta Psi_A Psi_B: stffness matrix z
 * [4] G^nu^mu_ihat
 * [5] F^nu^mu_ihat
 *
 * Inputs:
 * a,b: basis vector indices
 * t1, t2, t3: cartesian coordinates of the triangle vertices
 * x, y, z: cartesian coordinates of all vertices of the geodesic grid
 * scheme_weights: quadrature weights
 * scheme_points: quadrature points
 * matrixchoice: choice of matrix
 * nu, mu, ihat: optional for some matrices
 */
KOKKOS_INLINE_FUNCTION
Real IntegrateMatrixSphericalTriangle(int a, int b, int basis, int t1, int t2, int t3, const HostArray1D<Real> &x, const HostArray1D<Real> &y, const HostArray1D<Real> &z,
                                      const HostArray1D<Real> &scheme_weights, const HostArray2D<Real> &scheme_points, int matrixnumber, int nu, int mu, int ihat) {

  Real x1 = x(t1);
  Real y1 = y(t1);
  Real z1 = z(t1);

  Real x2 = x(t2);
  Real y2 = y(t2);
  Real z2 = z(t2);

  Real x3 = x(t3);
  Real y3 = y(t3);
  Real z3 = z(t3);

  Real result{0.};

  if (matrixnumber == 0) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2)))
          * FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis) * scheme_weights(i);
    }
  } else if (matrixnumber == 1) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += CosPhiSinTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
          * sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) *
          FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis) * scheme_weights(i);
    }
  } else if (matrixnumber == 2) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += SinPhiSinTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
          * sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) *
          FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis) * scheme_weights(i);
    }
  } else if (matrixnumber == 3) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += CosTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
          * sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))) *
          FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis) * scheme_weights(i);
    }
  } else if (matrixnumber == 4) {
    for (size_t i = 0; i < scheme_points.size(); i++) {
      result += MomentumUnitEnergy(nu, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
          * MomentumUnitEnergy(mu, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
          * sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2)))
          * FEMBasisA(a, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis)
          * PartialFEMBasiswithoute(ihat, b, t1, t2, t3, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis)
          * scheme_weights(i);
    }
  } else if (matrixnumber == 5) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += MomentumUnitEnergy(nu, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
          * MomentumUnitEnergy(mu, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2))
          * sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2)))
          * FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis) * scheme_weights(i) *
          MomentumUnitEnergy(ihat, x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2)) * scheme_weights(i);
    }
  }

  result = 0.5 * result;

  return result;
}

// --------------------------------------------
// Calculate determinant of Jacobian term
KOKKOS_INLINE_FUNCTION
Real
CalculateDeterminantJacobian(Real x1,
                             Real y1,
                             Real z1,
                             Real x2,
                             Real y2,
                             Real z2,
                             Real x3,
                             Real y3,
                             Real z3,
                             Real xi1,
                             Real xi2,
                             Real xi3) {

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
//KOKKOS_INLINE_FUNCTION
Real
IntegrateMatrixFEMN(int a,                                    // matrix row (this is an angle pair index)
                    int b,                                    // matrix column (this is an angle pairindex)
                    int basis,                                // the choice of basis (1: 'overlapping tent FEM basis')
                    const HostArray1D<Real> &x,               // cartesian x-coordinate of geodesic grid (can be anything when spherical harmonics is chosen)
                    const HostArray1D<Real> &y,               // cartesian y-coordinate of geodesic grid (can be anything when spherical harmonics is chosen)
                    const HostArray1D<Real> &z,               // cartesian z-coordinate of geodesic grid (can be anything when spherical harmonics is chosen)
                    const HostArray1D<Real> &scheme_weights,  // quadrature weights
                    const HostArray2D<Real> &scheme_points,   // quadrature points
                    const HostArray2D<int> &triangles,        // triangle information
                    int matrixchoice,                         // choice of matrix
                    int nu,
                    int mu,
                    int ihat) {
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

/* Top level function to integrate a matrix from a list for FP_N
 *
 * Choice of matrix is to be provided in matrixchoice:
 * [0] Psi_A Psi_B: mass matrix
 * [1] Cos Phi Sin Theta Psi_A Psi_B: stiffness matrix x
 * [2] Sin Phi Sin Theta Psi_A Psi_B: stiffness matrix y
 * [3] Cos Theta Psi_A Psi_B: stffness matrix z
 * [4] G^nu^mu_ihat
 * [5] F^nu^mu_ihat
 *
 * Inputs:
 * (la, ma): (l,m) corresponding to index A
 * (lb, mb): (l,m) corresponding to index B
 * scheme_weights: quadrature weights
 * scheme_points: quadrature points
 * matrixchoice: choice of matrix
 * nu, mu, ihat: optional for some matrices
 */
Real IntegrateMatrixFPN(int la, int ma, int lb, int mb, const HostArray1D<Real> &scheme_weights, const HostArray2D<Real> &scheme_points,
                        int matrixchoice, int nu, int mu, int ihat) {

  Real result = 0.;

  if (matrixchoice == 0) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * FPNBasis(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * FPNBasis(lb, mb, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  } else if (matrixchoice == 1) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * cos(scheme_points(i, 0)) * sin(scheme_points(i, 1))
          * FPNBasis(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * FPNBasis(lb, mb, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  } else if (matrixchoice == 2) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * sin(scheme_points(i, 0)) * sin(scheme_points(i, 1))
          * FPNBasis(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * FPNBasis(lb, mb, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  } else if (matrixchoice == 3) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * cos(scheme_points(i, 1))
          * FPNBasis(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * FPNBasis(lb, mb, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  } else if (matrixchoice == 4) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * MomentumUnitEnergy(nu, scheme_points(i, 0), scheme_points(i, 1)) *
          MomentumUnitEnergy(mu, scheme_points(i, 0), scheme_points(i, 1))
          * FPNBasis(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * (PtildehatJac(scheme_points(i, 0), scheme_points(i, 1), 1, ihat) * dFPNBasisdOmega(lb, mb, scheme_points(i, 0), scheme_points(i, 1), 1)
              + PtildehatJac(scheme_points(i, 0), scheme_points(i, 1), 2, ihat) * dFPNBasisdOmega(lb, mb, scheme_points(i, 0), scheme_points(i, 1), 2));
    }
  } else if (matrixchoice == 5) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * MomentumUnitEnergy(nu, scheme_points(i, 0), scheme_points(i, 1)) *
          MomentumUnitEnergy(mu, scheme_points(i, 0), scheme_points(i, 1))
          * FPNBasis(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * FPNBasis(lb, mb, scheme_points(i, 0), scheme_points(i, 1))
          * MomentumUnitEnergy(ihat, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  }
  return result;
}
} // namespace radiationfemn