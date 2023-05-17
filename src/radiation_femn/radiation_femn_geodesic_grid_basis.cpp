//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_basis.cpp
//  \brief implementation of the radiation FEM_N basis functions and helper functions

#include <iostream>
#include "athena.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

// -----------------------------------------------------------------------------------
// Convert Barycentric coordinates to Cartesian coordinates given vertices of triangle
KOKKOS_INLINE_FUNCTION
void BarycentricToCartesian(double x1,
                            double y1,
                            double z1,
                            double x2,
                            double y2,
                            double z2,
                            double x3,
                            double y3,
                            double z3,
                            double xi1,
                            double xi2,
                            double xi3,
                            double &xval,
                            double &yval,
                            double &zval) {

  xval = xi1 * x1 + xi2 * x2 + xi3 * x3;
  yval = xi1 * y1 + xi2 * y2 + xi3 * y3;
  zval = xi1 * z1 + xi2 * z2 + xi3 * z3;

}

// ------------------------------------------------------------------------------------------------
// Given index numbers of two vertices, finds if they share an edge and if so, return triangle info
// If a = b, this return all triangles which share the vertex
void FindTriangles(int a, int b, const HostArray2D<int> &triangles, HostArray2D<int> edge_triangles, bool &is_edge) {

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
      if ((triangles(i, 0) == a && triangles(i, 1) == b) || (triangles(i, 0) == a && triangles(i, 2) == b) ||
          (triangles(i, 0) == b && triangles(i, 1) == a) || (triangles(i, 0) == b && triangles(i, 2) == a) ||
          (triangles(i, 1) == a && triangles(i, 2) == b) || (triangles(i, 1) == b && triangles(i, 2) == a)) {
        is_edge = true;
        edge_triangles(index, 0) = triangles(i, 0);
        edge_triangles(index, 1) = triangles(i, 1);
        edge_triangles(index, 2) = triangles(i, 2);
        index++;
      }
    }
  }
}

// --------------------------------------------------------------------
// Basis 1: 'overlapping tent
KOKKOS_INLINE_FUNCTION
double FEMBasis1Type1(double xi1, double xi2, double xi3) {
  return 2. * xi1 + xi2 + xi3 - 1.;
}

KOKKOS_INLINE_FUNCTION
double FEMBasis2Type1(double xi1, double xi2, double xi3) {
  return xi1 + 2. * xi2 + xi3 - 1.;
}

KOKKOS_INLINE_FUNCTION
double FEMBasis3Type1(double xi1, double xi2, double xi3) {
  return xi1 + xi2 + 2. * xi3 - 1.;
}

// -------------------------------------------------------------------
// Basis 2: 'small tent'
KOKKOS_INLINE_FUNCTION
double FEMBasis1Type2(double xi1, double xi2, double xi3) {
  return (xi1 >= 0.5) * (xi1 - xi2 - xi3);
}

KOKKOS_INLINE_FUNCTION
double FEMBasis2Type2(double xi1, double xi2, double xi3) {
  return (xi2 >= 0.5) * (xi2 - xi3 - xi1);
}

KOKKOS_INLINE_FUNCTION
double FEMBasis3Type2(double xi1, double xi2, double xi3) {
  return (xi3 >= 0.5) * (xi3 - xi1 - xi2);
}

// --------------------------------------------------------------------
// Basis 3: 'overlapping honeycomb'
KOKKOS_INLINE_FUNCTION
double FEMBasis1Type3(double xi1, double xi2, double xi3) {
  return 1.;
}

KOKKOS_INLINE_FUNCTION
double FEMBasis2Type3(double xi1, double xi2, double xi3) {
  return 1.;
}

KOKKOS_INLINE_FUNCTION
double FEMBasis3Type3(double xi1, double xi2, double xi3) {
  return 1.;
}

// -------------------------------------------------------------------
// Basis 4: 'non-overlapping honeycomb'
KOKKOS_INLINE_FUNCTION
double FEMBasis1Type4(double xi1, double xi2, double xi3) {
  return (xi1 >= xi2) * (xi1 > xi3) * 1.;
}

KOKKOS_INLINE_FUNCTION
double FEMBasis2Type4(double xi1, double xi2, double xi3) {
  return (xi2 >= xi3) * (xi2 > xi1) * 1.;
}

KOKKOS_INLINE_FUNCTION
double FEMBasis3Type4(double xi1, double xi2, double xi3) {
  return (xi3 >= xi1) * (xi3 > xi2) * 1.;
}

// ---------------------------------------------------------------------
// FEM basis in barycentric coordinates
KOKKOS_INLINE_FUNCTION
double FEMBasis(double xi1, double xi2, double xi3, int basis_index, int basis_choice) {
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
KOKKOS_INLINE_FUNCTION
double dFEMBasis1Type1dxi1(double xi1, double xi2, double xi3) {
  return 2.;
}

KOKKOS_INLINE_FUNCTION
double dFEMBasis2Type1dxi1(double xi1, double xi2, double xi3) {
  return 1.;
}

KOKKOS_INLINE_FUNCTION
double dFEMBasis3Type1dxi1(double xi1, double xi2, double xi3) {
  return 1.;
}

KOKKOS_INLINE_FUNCTION
double dFEMBasis1Type1dxi2(double xi1, double xi2, double xi3) {
  return 1.;
}

KOKKOS_INLINE_FUNCTION
double dFEMBasis2Type1dxi2(double xi1, double xi2, double xi3) {
  return 2.;
}

KOKKOS_INLINE_FUNCTION
double dFEMBasis3Type1dxi2(double xi1, double xi2, double xi3) {
  return 1.;
}

// ------------------------------------------------------------------
// Derivative of 'overlapping tent' basis wrt barycentric coordinates
KOKKOS_INLINE_FUNCTION
double dFEMBasisdxi(double xi1, double xi2, double xi3, int basis_index, int basis_choice, int xi_index) {
  if (basis_index == 1 && basis_choice == 1 && xi_index == 1) {
    return dFEMBasis1Type1dxi1(xi1, xi2, xi3);
  } else if (basis_index == 2 && basis_choice == 1 && xi_index == 1) {
    return dFEMBasis2Type1dxi1(xi1, xi2, xi3);
  } else if (basis_index == 3 && basis_choice == 1 && xi_index == 1) {
    return dFEMBasis3Type1dxi1(xi1, xi2, xi3);
  }
  if (basis_index == 1 && basis_choice == 1 && xi_index == 2) {
    return dFEMBasis1Type1dxi2(xi1, xi2, xi3);
  } else if (basis_index == 2 && basis_choice == 1 && xi_index == 2) {
    return dFEMBasis2Type1dxi2(xi1, xi2, xi3);
  } else if (basis_index == 3 && basis_choice == 1 && xi_index == 2) {
    return dFEMBasis3Type1dxi2(xi1, xi2, xi3);
  } else {
    std::cout << "Incorrect basis_choice of basis function in radiation-femn block!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

// ------------------------------------------------------------
// Product of two FEM basis given their index and triangle info
KOKKOS_INLINE_FUNCTION
double FEMBasisABasisB(int a, int b, int t1, int t2, int t3, double xi1, double xi2, double xi3, int basis_choice) {

  int basis_index_a = (a == t1) * 1 + (a == t2) * 2 + (a == t3) * 3;
  int basis_index_b = (b == t1) * 1 + (b == t2) * 2 + (b == t3) * 3;

  auto FEMBasisA = FEMBasis(xi1, xi2, xi3, basis_index_a, basis_choice);
  auto FEMBasisB = FEMBasis(xi1, xi2, xi3, basis_index_b, basis_choice);

  return FEMBasisA * FEMBasisB;
}

// -----------------------------------------------------------------------
// FEM basis given its index and triangle info
KOKKOS_INLINE_FUNCTION
double FEMBasisA(int a, int t1, int t2, int t3, double xi1, double xi2, double xi3, int basis_choice) {

  int basis_index_a = (a == t1) * 1 + (a == t2) * 2 + (a == t3) * 3;

  auto FEMBasisA = FEMBasis(xi1, xi2, xi3, basis_index_a, basis_choice);

  return FEMBasisA;
}

// -------------------------------------------------------------------------
// Cos Phi Sin Theta
KOKKOS_INLINE_FUNCTION
double CosPhiSinTheta(double x1,
                      double y1,
                      double z1,
                      double x2,
                      double y2,
                      double z2,
                      double x3,
                      double y3,
                      double z3,
                      double xi1,
                      double xi2,
                      double xi3) {
  double xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  double rval = sqrt(xval * xval + yval * yval + zval * zval);
  double thetaval = acos(zval / rval);
  double phival = atan2(yval, xval);

  return cos(phival) * sin(thetaval);
}

// ------------------------------------------------------------------------
// Sin Phi Sin Theta
KOKKOS_INLINE_FUNCTION
double SinPhiSinTheta(double x1,
                      double y1,
                      double z1,
                      double x2,
                      double y2,
                      double z2,
                      double x3,
                      double y3,
                      double z3,
                      double xi1,
                      double xi2,
                      double xi3) {
  double xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  double rval = sqrt(xval * xval + yval * yval + zval * zval);
  double thetaval = acos(zval / rval);
  double phival = atan2(yval, xval);

  return sin(phival) * sin(thetaval);
}

// ------------------------------------------------------------------------
// Cos Theta
KOKKOS_INLINE_FUNCTION
double CosTheta(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double x3,
                double y3,
                double z3,
                double xi1,
                double xi2,
                double xi3) {
  double xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  double rval = sqrt(xval * xval + yval * yval + zval * zval);
  double thetaval = acos(zval / rval);

  return cos(thetaval);
}

// -------------------------------------------------------------------------
// sin Phi Cosec Theta
KOKKOS_INLINE_FUNCTION
double SinPhiCosecTheta(double x1,
                        double y1,
                        double z1,
                        double x2,
                        double y2,
                        double z2,
                        double x3,
                        double y3,
                        double z3,
                        double xi1,
                        double xi2,
                        double xi3) {
  double xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  double rval = sqrt(xval * xval + yval * yval + zval * zval);
  double thetaval = acos(zval / rval);
  double phival = atan2(yval, xval);

  return sin(phival) / sin(thetaval);
}

// -------------------------------------------------------------------------
// Cos Phi Cos Theta
KOKKOS_INLINE_FUNCTION
double CosPhiCosTheta(double x1,
                      double y1,
                      double z1,
                      double x2,
                      double y2,
                      double z2,
                      double x3,
                      double y3,
                      double z3,
                      double xi1,
                      double xi2,
                      double xi3) {
  double xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  double rval = sqrt(xval * xval + yval * yval + zval * zval);
  double thetaval = acos(zval / rval);
  double phival = atan2(yval, xval);

  return cos(phival) * cos(thetaval);
}

// ------------------------------------------------------------------------
// Cos Phi Cosec Theta
KOKKOS_INLINE_FUNCTION
double CosPhiCosecTheta(double x1,
                        double y1,
                        double z1,
                        double x2,
                        double y2,
                        double z2,
                        double x3,
                        double y3,
                        double z3,
                        double xi1,
                        double xi2,
                        double xi3) {
  double xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  double rval = sqrt(xval * xval + yval * yval + zval * zval);
  double thetaval = acos(zval / rval);
  double phival = atan2(yval, xval);

  return cos(phival) / sin(thetaval);
}

// ------------------------------------------------------------------------
// Sin Phi Cos Theta
KOKKOS_INLINE_FUNCTION
double SinPhiCosTheta(double x1,
                      double y1,
                      double z1,
                      double x2,
                      double y2,
                      double z2,
                      double x3,
                      double y3,
                      double z3,
                      double xi1,
                      double xi2,
                      double xi3) {
  double xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  double rval = sqrt(xval * xval + yval * yval + zval * zval);
  double thetaval = acos(zval / rval);
  double phival = atan2(yval, xval);

  return sin(phival) * cos(thetaval);
}

// ------------------------------------------------------------------------
// Sin Theta
KOKKOS_INLINE_FUNCTION
double SinTheta(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double x3,
                double y3,
                double z3,
                double xi1,
                double xi2,
                double xi3) {
  double xval, yval, zval;
  BarycentricToCartesian(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3, xval, yval, zval);

  double rval = sqrt(xval * xval + yval * yval + zval * zval);
  double thetaval = acos(zval / rval);

  return sin(thetaval);
}

// ------------------------------------------------------------
// Momentum contra-vector divided by energy (in comoving frame)
double mom_by_energy(int mu,
                     double x1,
                     double y1,
                     double z1,
                     double x2,
                     double y2,
                     double z2,
                     double x3,
                     double y3,
                     double z3,
                     double xi1,
                     double xi2,
                     double xi3) {
  double result = 0.;
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

// ------------------------------------------------------------------------
// partial xi1 / partial phi
double pXi1pPhi(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double x3,
                double y3,
                double z3,
                double xi1,
                double xi2,
                double xi3) {
  return (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2)) /
      (x3 * (y1 - xi2 * y1 + xi2 * y2) + x2 * xi2 * (y1 - y3) - x1 * (xi2 * y2 + y3 - xi2 * y3));
}

// ------------------------------------------------------------------------
// partial xi2 / partial phi
double pXi2pPhi(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double x3,
                double y3,
                double z3,
                double xi1,
                double xi2,
                double xi3) {
  return (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2)) /
      (x3 * (xi1 * y1 + y2 - xi1 * y2) + x1 * xi1 * (y2 - y3) - x2 * (xi1 * (y1 - y3) + y3));
}

// ------------------------------------------------------------------------
// partial xi1 / partial theta
double pXi1pTheta(double x1,
                  double y1,
                  double z1,
                  double x2,
                  double y2,
                  double z2,
                  double x3,
                  double y3,
                  double z3,
                  double xi1,
                  double xi2,
                  double xi3) {
  return (-2
      * pow(pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2) +
          pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2), 1.5) *
      sqrt(1 - pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2) /
          (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2) +
              pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)))) /
      (-2 * (xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3) *
          ((x1 - x3) * (x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2))
              + (y1 - y3) * (xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3) +
              (z1 - z3) * (xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3)) + 2 * (z1 - z3) *
          (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) +
              pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2) +
              pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)));
}

// ------------------------------------------------------------------------
// partial xi2 / partial theta
double pXi2pTheta(double x1,
                  double y1,
                  double z1,
                  double x2,
                  double y2,
                  double z2,
                  double x3,
                  double y3,
                  double z3,
                  double xi1,
                  double xi2,
                  double xi3) {
  return (-2
      * pow(pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2) +
          pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2), 1.5) *
      sqrt(1 - pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2) /
          (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) + pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2) +
              pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)))) /
      (-2 * (xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3) *
          ((x2 - x3) * (x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2))
              + (y2 - y3) * (xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3) +
              (z2 - z3) * (xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3)) + 2 * (z2 - z3) *
          (pow(x1 * xi1 + x2 * xi2 - x3 * (-1 + xi1 + xi2), 2) +
              pow(xi1 * y1 + xi2 * y2 + y3 - (xi1 + xi2) * y3, 2) +
              pow(xi1 * z1 + xi2 * z2 + z3 - (xi1 + xi2) * z3, 2)));
}

double PartialFEMBasisAwithoute(int ihat,
                                int a,
                                int t1,
                                int t2,
                                int t3,
                                double x1,
                                double y1,
                                double z1,
                                double x2,
                                double y2,
                                double z2,
                                double x3,
                                double y3,
                                double z3,
                                double xi1,
                                double xi2,
                                double xi3,
                                int basis_choice) {

  int basis_index_a = (a == t1) * 1 + (a == t2) * 2 + (a == t3) * 3;

  double dFEMBasisdphi = dFEMBasisdxi(xi1, xi2, xi3, basis_index_a, basis_choice, 1)
      * pXi1pPhi(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) +
      dFEMBasisdxi(xi1, xi2, xi3, basis_index_a, basis_choice, 2)
          * pXi2pPhi(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3);
  double dFEMBasisdtheta = dFEMBasisdxi(xi1, xi2, xi3, basis_index_a, basis_choice, 1)
      * pXi1pTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) +
      dFEMBasisdxi(xi1, xi2, xi3, basis_index_a, basis_choice, 2)
          * pXi2pTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3);

  if (ihat == 1) {
    return -SinPhiCosecTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdphi +
        CosPhiCosTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdtheta;
  } else if (ihat == 2) {
    return CosPhiCosecTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdphi +
        SinPhiCosTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdtheta;
  } else if (ihat == 3) {
    return -SinTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, xi1, xi2, xi3) * dFEMBasisdtheta;
  } else {
    std::cout << "Incorrect choice of index ihat!" << std::endl;
    exit(EXIT_FAILURE);
  }
}
} // namespace radiationfemn