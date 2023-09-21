//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_basis.cpp
//  \brief implementation of the radiation FEM_N basis functions and helper functions

#include <iostream>
#include <gsl/gsl_sf_legendre.h>
#include "athena.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

/* Convert Barycentric coordinates to Cartesian coordinates given vertices of triangle
 *
 * Inputs:
 * (x1,y1,z1), (x2,y2,z2), (x3,y3,z3):  the three triangle vertices in cartesian coordinates
 * (xi1, xi2, xi3):the barycentric coordinates of a point inside the triangle
 *
 * Output:
 * (xval, yval, zval): the cartesian coordinates of the point
 */
KOKKOS_INLINE_FUNCTION void BarycentricToCartesian(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3,
                                                   Real xi1, Real xi2, Real xi3, Real &xval, Real &yval, Real &zval) {

  xval = xi1 * x1 + xi2 * x2 + xi3 * x3;
  yval = xi1 * y1 + xi2 * y2 + xi3 * y3;
  zval = xi1 * z1 + xi2 * z2 + xi3 * z3;

}

/* Given index numbers of two vertices, finds if they share an edge and if so, return triangle info
 *
 * If a = b, this return all triangles which share the vertex
 *
 * Inputs:
 * a, b: index number of vertices
 * triangles: triangle information of the geodesic grid
 *
 * Outputs:
 * edge_triangles: the vertex information of shared edge(s)
 * is_edge: bool for if the vertices share an edge or not
 */
void FindTriangles(int a, int b, const HostArray2D<int> &triangles, HostArray2D<int> &edge_triangles, bool &is_edge) {

  is_edge = false;
  Kokkos::realloc(edge_triangles, 6, 3);
  Kokkos::deep_copy(edge_triangles, -42.);

  if (a == b) {
    size_t index{0};
    for (size_t i = 0; i < triangles.size() / 3; i++) {
      {
        if (triangles(i, 0) == a || triangles(i, 1) == a || triangles(i, 2) == a) {
          is_edge = true;
          edge_triangles(index, 0) = triangles(i, 0);
          edge_triangles(index, 1) = triangles(i, 1);
          edge_triangles(index, 2) = triangles(i, 2);
          index++;
        }
      }
    }
  } else if (a != b) {
    size_t index{0};
    for (size_t i = 0; i < triangles.size() / 3; i++) {
      if ((triangles(i, 0) == a && triangles(i, 1) == b) || (triangles(i, 0) == a && triangles(i, 2) == b) || (triangles(i, 0) == b && triangles(i, 1) == a)
          || (triangles(i, 0) == b && triangles(i, 2) == a) || (triangles(i, 1) == a && triangles(i, 2) == b) || (triangles(i, 1) == b && triangles(i, 2) == a)) {
        is_edge = true;
        edge_triangles(index, 0) = triangles(i, 0);
        edge_triangles(index, 1) = triangles(i, 1);
        edge_triangles(index, 2) = triangles(i, 2);
        index++;
      }
    }
  }
}

/* FEM basis functions: 'overlapping tent'
 *
 * Basis is given in barycentric coordinates
 */

// Overlapping tent basis 1
KOKKOS_INLINE_FUNCTION Real FEMBasis1Type1(Real xi1, Real xi2, Real xi3) {
  return 2. * xi1 + xi2 + xi3 - 1.;
}

// Overlapping tent basis 2
KOKKOS_INLINE_FUNCTION Real FEMBasis2Type1(Real xi1, Real xi2, Real xi3) {
  return xi1 + 2. * xi2 + xi3 - 1.;
}

// Overlapping tent basis 1
KOKKOS_INLINE_FUNCTION Real FEMBasis3Type1(Real xi1, Real xi2, Real xi3) {
  return xi1 + xi2 + 2. * xi3 - 1.;
}

/* FEM basis functions: 'small tent'
 *
 * Basis is given in barycentric coordinates
 */

// Small tent basis 1
KOKKOS_INLINE_FUNCTION Real FEMBasis1Type2(Real xi1, Real xi2, Real xi3) {
  return (xi1 >= 0.5) * (xi1 - xi2 - xi3);
}

// Small tent basis 2
KOKKOS_INLINE_FUNCTION Real FEMBasis2Type2(Real xi1, Real xi2, Real xi3) {
  return (xi2 >= 0.5) * (xi2 - xi3 - xi1);
}

// Small tent basis 3
KOKKOS_INLINE_FUNCTION Real FEMBasis3Type2(Real xi1, Real xi2, Real xi3) {
  return (xi3 >= 0.5) * (xi3 - xi1 - xi2);
}

/* FEM basis functions: 'overlapping honeycomb'
 *
 * Basis is given in barycentric coordinates
 */

// Overlapping honeycomb basis 1
KOKKOS_INLINE_FUNCTION Real FEMBasis1Type3(Real xi1, Real xi2, Real xi3) {
  return 1.;
}

// Overlapping honeycomb basis 2
KOKKOS_INLINE_FUNCTION Real FEMBasis2Type3(Real xi1, Real xi2, Real xi3) {
  return 1.;
}

// Overlapping honeycomb basis 3
KOKKOS_INLINE_FUNCTION Real FEMBasis3Type3(Real xi1, Real xi2, Real xi3) {
  return 1.;
}

/* FEM basis functions: 'non-overlapping honeycomb'
 *
 * Basis is given in barycentric coordinates
 */

// Non-overlapping honeycomb basis 1
KOKKOS_INLINE_FUNCTION Real FEMBasis1Type4(Real xi1, Real xi2, Real xi3) {
  return (xi1 >= xi2) * (xi1 > xi3) * 1.;
}

// Non-overlapping honeycomb basis 2
KOKKOS_INLINE_FUNCTION Real FEMBasis2Type4(Real xi1, Real xi2, Real xi3) {
  return (xi2 >= xi3) * (xi2 > xi1) * 1.;
}

// Non-overlapping honeycomb basis 3
KOKKOS_INLINE_FUNCTION Real FEMBasis3Type4(Real xi1, Real xi2, Real xi3) {
  return (xi3 >= xi1) * (xi3 > xi2) * 1.;
}

/* Main FEM basis function in barycentric coordinates: allows choice of basis number and basis type
 *
 * choice: [1] overlapping tent [2] small tent [3] overlapping honeycomb [4] small honeycomb
 * basis_index: [1]: basis peaked at xi1 = 1 [2], basis peaked at xi2 = 1 [3] basis peaked at xi3 = 1
 */
KOKKOS_INLINE_FUNCTION Real FEMBasis(Real xi1, Real xi2, Real xi3, int basis_index, int basis_choice) {
  if (basis_index == 1 && basis_choice == 1) {
    return FEMBasis1Type1(xi1, xi2, xi3);
  } else if (basis_index == 1 && basis_choice == 2) {
    return FEMBasis1Type2(xi1, xi2, xi3);
  } else if (basis_index == 1 && basis_choice == 3) {
    return FEMBasis1Type3(xi1, xi2, xi3);
  } else if (basis_index == 1 && basis_choice == 4) {
    return FEMBasis1Type4(xi1, xi2, xi3);
  } else if (basis_index == 2 && basis_choice == 1) {
    return FEMBasis2Type1(xi1, xi2, xi3);
  } else if (basis_index == 2 && basis_choice == 2) {
    return FEMBasis2Type2(xi1, xi2, xi3);
  } else if (basis_index == 2 && basis_choice == 3) {
    return FEMBasis2Type3(xi1, xi2, xi3);
  } else if (basis_index == 2 && basis_choice == 4) {
    return FEMBasis2Type4(xi1, xi2, xi3);
  } else if (basis_index == 3 && basis_choice == 1) {
    return FEMBasis3Type1(xi1, xi2, xi3);
  } else if (basis_index == 3 && basis_choice == 2) {
    return FEMBasis3Type2(xi1, xi2, xi3);
  } else if (basis_index == 3 && basis_choice == 3) {
    return FEMBasis3Type3(xi1, xi2, xi3);
  } else if (basis_index == 3 && basis_choice == 4) {
    return FEMBasis3Type4(xi1, xi2, xi3);
  } else {
    std::cout << "Incorrect basis_choice of basis function in radiation-femn block!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

// ---------------------------------------------------------------------------------------
// Partial derivatives of 'overlapping tent' basis with respect to Barycentric coordinates
KOKKOS_INLINE_FUNCTION Real dFEMBasis1Type1dxi1(Real xi1, Real xi2, Real xi3) {
  return 2.;
}

KOKKOS_INLINE_FUNCTION Real dFEMBasis2Type1dxi1(Real xi1, Real xi2, Real xi3) {
  return 1.;
}

KOKKOS_INLINE_FUNCTION Real dFEMBasis3Type1dxi1(Real xi1, Real xi2, Real xi3) {
  return 1.;
}

KOKKOS_INLINE_FUNCTION Real dFEMBasis1Type1dxi2(Real xi1, Real xi2, Real xi3) {
  return 1.;
}

KOKKOS_INLINE_FUNCTION Real dFEMBasis2Type1dxi2(Real xi1, Real xi2, Real xi3) {
  return 2.;
}

KOKKOS_INLINE_FUNCTION Real dFEMBasis3Type1dxi2(Real xi1, Real xi2, Real xi3) {
  return 1.;
}

/* Derivative of 'overlapping tent' basis functions with respect to barycentric coordinates
 *
 * Note: basis_choice is set to 1. Do not use any other number
 *
 * (xi1, xi2, xi3) is the point at which the derivative is taken
 */
KOKKOS_INLINE_FUNCTION Real dFEMBasisdxi(Real xi1, Real xi2, Real xi3, int basis_index, int xi_index) {
  if (basis_index == 1 && xi_index == 1) {
    return dFEMBasis1Type1dxi1(xi1, xi2, xi3);
  } else if (basis_index == 2 && xi_index == 1) {
    return dFEMBasis2Type1dxi1(xi1, xi2, xi3);
  } else if (basis_index == 3 && xi_index == 1) {
    return dFEMBasis3Type1dxi1(xi1, xi2, xi3);
  }
  if (basis_index == 1 && xi_index == 2) {
    return dFEMBasis1Type1dxi2(xi1, xi2, xi3);
  } else if (basis_index == 2 && xi_index == 2) {
    return dFEMBasis2Type1dxi2(xi1, xi2, xi3);
  } else if (basis_index == 3 && xi_index == 2) {
    return dFEMBasis3Type1dxi2(xi1, xi2, xi3);
  } else {
    std::cout << "Incorrect basis_choice of basis function in radiation-femn block!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

// ------------------------------------------------------------
// Product of two FEM basis given their index and triangle info
//KOKKOS_INLINE_FUNCTION
Real FEMBasisABasisB(int a, int b, int t1, int t2, int t3, Real xi1, Real xi2, Real xi3, int basis_choice) {

  int basis_index_a = (a == t1) * 1 + (a == t2) * 2 + (a == t3) * 3;
  int basis_index_b = (b == t1) * 1 + (b == t2) * 2 + (b == t3) * 3;

  auto FEMBasisA = FEMBasis(xi1, xi2, xi3, basis_index_a, basis_choice);
  auto FEMBasisB = FEMBasis(xi1, xi2, xi3, basis_index_b, basis_choice);

  return FEMBasisA * FEMBasisB;
}

/* FPN basis function: real spherical harmonics
 *
 * Calculated for (l,m) at point (phi, theta) on the sphere
 */
Real FPNBasis(int l, int m, Real phi, Real theta) {
  Real result = 0.;
  if (m > 0) {
    result = sqrt(2.) * cos(m * phi) * gsl_sf_legendre_sphPlm(l, m, cos(theta));
  } else if (m == 0) {
    result = gsl_sf_legendre_sphPlm(l, 0, cos(theta));
  } else {
    result = sqrt(2.) * sin(abs(m) * phi) * gsl_sf_legendre_sphPlm(l, abs(m), cos(theta));
  }

  return result;
}

KOKKOS_INLINE_FUNCTION Real dFPNBasisdphi(int l, int m, Real phi, Real theta) {
  return m * FPNBasis(l, -m, phi, theta);
}

KOKKOS_INLINE_FUNCTION Real dFPNBasisdtheta(int l, int m, Real phi, Real theta) {
  Real result = 0.;

  Real der_legendre =
      -sin(theta) * ((m - l - 1.) * gsl_sf_legendre_sphPlm(l + 1, abs(m), cos(theta)) - (l + 1) * cos(theta) * gsl_sf_legendre_sphPlm(l, abs(m), cos(theta)))
          / (1. - cos(theta) * cos(theta));

  if (m > 0) {
    result = sqrt(2.) * cos(m * phi) * der_legendre;
  } else if (m == 0) {
    result = der_legendre;
  } else {
    result = sqrt(2.) * sin(abs(m) * phi) * der_legendre;
  }

  return result;
}

Real dFPNBasisdOmega(int l, int m, Real phi, Real theta, int var_index) {
  if (var_index == 1) {
    return dFPNBasisdphi(l, m, phi, theta);
  } else if (var_index == 2) {
    return dFPNBasisdtheta(l, m, phi, theta);
  } else {
    std::cout << "Incorrect choice of variable index in radiation-femn block!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

Real PtildehatJac(Real phi, Real theta, int tilde_index, int hat_index) {
  if (tilde_index == 1 && hat_index == 1) {
    return -sin(phi) / sin(theta);
  } else if (tilde_index == 1 && hat_index == 2) {
    return cos(phi) / sin(theta);
  } else if (tilde_index == 1 && hat_index == 3) {
    return 0.;
  } else if (tilde_index == 2 && hat_index == 1) {
    return cos(phi) * cos(theta);
  } else if (tilde_index == 2 && hat_index == 2) {
    return sin(phi) * cos(theta);
  } else if (tilde_index == 2 && hat_index == 3) {
    return -sin(theta);
  } else {
    std::cout << "Incorrect choice of index in radiation-femn block!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

// -------------------------------------------------------------------------
// Cos Phi Sin Theta
//KOKKOS_INLINE_FUNCTION
Real CosPhiSinTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  Real rval = sqrt(xval * xval + yval * yval + zval * zval);
  Real thetaval = acos(zval / rval);
  Real phival = atan2(yval, xval);

  return cos(phival) * sin(thetaval);
}

// ------------------------------------------------------------------------
// Sin Phi Sin Theta
//KOKKOS_INLINE_FUNCTION
Real SinPhiSinTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  Real rval = sqrt(xval * xval + yval * yval + zval * zval);
  Real thetaval = acos(zval / rval);
  Real phival = atan2(yval, xval);

  return sin(phival) * sin(thetaval);
}

// ------------------------------------------------------------------------
// Cos Theta
//KOKKOS_INLINE_FUNCTION
Real CosTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  Real rval = sqrt(xval * xval + yval * yval + zval * zval);
  Real thetaval = acos(zval / rval);

  return cos(thetaval);
}

// -------------------------------------------------------------------------
// sin Phi Cosec Theta
//KOKKOS_INLINE_FUNCTION
Real SinPhiCosecTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  Real rval = sqrt(xval * xval + yval * yval + zval * zval);
  Real thetaval = acos(zval / rval);
  Real phival = atan2(yval, xval);

  return sin(phival) / sin(thetaval);
}

// -------------------------------------------------------------------------
// Cos Phi Cos Theta
//KOKKOS_INLINE_FUNCTION
Real CosPhiCosTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  Real rval = sqrt(xval * xval + yval * yval + zval * zval);
  Real thetaval = acos(zval / rval);
  Real phival = atan2(yval, xval);

  return cos(phival) * cos(thetaval);
}

// ------------------------------------------------------------------------
// Cos Phi Cosec Theta
//KOKKOS_INLINE_FUNCTION
Real CosPhiCosecTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  Real rval = sqrt(xval * xval + yval * yval + zval * zval);
  Real thetaval = acos(zval / rval);
  Real phival = atan2(yval, xval);

  return cos(phival) / sin(thetaval);
}

// ------------------------------------------------------------------------
// Sin Phi Cos Theta
//KOKKOS_INLINE_FUNCTION
Real SinPhiCosTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  Real rval = sqrt(xval * xval + yval * yval + zval * zval);
  Real thetaval = acos(zval / rval);
  Real phival = atan2(yval, xval);

  return sin(phival) * cos(thetaval);
}

// ------------------------------------------------------------------------
// Sin Theta
//KOKKOS_INLINE_FUNCTION
Real SinTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  Real rval = sqrt(xval * xval + yval * yval + zval * zval);
  Real thetaval = acos(zval / rval);

  return sin(thetaval);
}

// ------------------------------------------------------------
// Momentum contra-vector divided by energy (in comoving frame)
Real mom_by_energy(int mu, Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  Real result = 0.;
  if (mu == 0) {
    result = 1.;
  } else if (mu == 1) {
    result = CosPhiSinTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3);
  } else if (mu == 2) {
    result = SinPhiSinTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3);
  } else if (mu == 3) {
    result = CosTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3);
  } else {
    std::cout << "Incorrect choice of index for p^mu/e!" << std::endl;
    exit(EXIT_FAILURE);
  }

  return result;
}

Real mom_by_energy(int mu, Real phi, Real theta) {
  Real result = 0.;
  if (mu == 0) {
    result = 1.;
  } else if (mu == 1) {
    result = cos(phi) * sin(theta);
  } else if (mu == 2) {
    result = sin(phi) * sin(theta);
  } else if (mu == 3) {
    result = cos(theta);
  } else {
    std::cout << "Incorrect choice of index for p^mu/e!" << std::endl;
    exit(EXIT_FAILURE);
  }

  return result;
}
/*
// ------------------------------------------------------------------------
// partial xi1 / partial phi
Real pXi1pPhi(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  return (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2))
      / (x3 * (y1 - xi2 * y1 + xi2 * y2) + x2 * xi2 * (y1 - y3) - x1 * (xi2 * y2 + y3 - xi2 * y3));
}

// ------------------------------------------------------------------------
// partial xi2 / partial phi
Real pXi2pPhi(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  return (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2))
      / (x3 * (xi1 * y1 + y2 - xi1 * y2) + x1 * xi1 * (y2 - y3) - x2 * (xi1 * (y1 - y3) + y3));
}

// ------------------------------------------------------------------------
// partial xi1 / partial theta
Real pXi1pTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  return (-2 * pow(
      pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2) + pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2),
      1.5) * sqrt(1 - pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)
      / (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2)
          + pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)))) / (-2 * (xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3)
      * ((x1 - x3) * (x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2)) + (y1 - y3) * (xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3)
          + (z1 - z3) * (xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3)) + 2 * (z1 - z3)
      * (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2)
          + pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)));
}

// ------------------------------------------------------------------------
// partial xi2 / partial theta
Real pXi2pTheta(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3, Real xi1, Real xi2, Real xi3) {
  return (-2 * pow(
      pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2) + pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2),
      1.5) * sqrt(1 - pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)
      / (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2)
          + pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)))) / (-2 * (xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3)
      * ((x2 - x3) * (x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2)) + (y2 - y3) * (xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3)
          + (z2 - z3) * (xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3)) + 2 * (z2 - z3)
      * (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2)
          + pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)));
}

Real PartialFEMBasisAwithoute(int ihat, int a, int t1, int t2, int t3, Real x1, Real y1, Real z1, Real x2, Real y2, Real z2, Real x3, Real y3, Real z3,
                              Real xi1, Real xi2, Real xi3, int basis_choice) {

  int basis_index_a = (a == t1) * 1 + (a == t2) * 2 + (a == t3) * 3;

  Real dFEMBasisdphi = dFEMBasisdxi(xi1, xi2, xi3, basis_index_a, basis_choice, 1) * pXi1pPhi(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3)
      + dFEMBasisdxi(xi1, xi2, xi3, basis_index_a, basis_choice, 2) * pXi2pPhi(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3);
  Real dFEMBasisdtheta = dFEMBasisdxi(xi1, xi2, xi3, basis_index_a, basis_choice, 1) * pXi1pTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3)
      + dFEMBasisdxi(xi1, xi2, xi3, basis_index_a, basis_choice, 2) * pXi2pTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3);

  if (ihat == 1) {
    return -SinPhiCosecTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdphi
        + CosPhiCosTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdtheta;
  } else if (ihat == 2) {
    return CosPhiCosecTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdphi
        + SinPhiCosTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdtheta;
  } else if (ihat == 3) {
    return -SinTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdtheta;
  } else {
    std::cout << "Incorrect choice of index ihat!" << std::endl;
    exit(EXIT_FAILURE);
  }
}*/
} // namespace radiationfemn