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
//         (c) IntegrateMatrix: Find the integrals over all angles for a given function

#include <cassert>
#include <complex>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include "athena.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

// ----------------------------------------------
// ONLY MODIFY THIS FUNCTION TO ADD NEW CASES
// Integrate a function over a spherical triangle
// Implemented functions:
// 0: Psi_A Psi_B
// 1: Cos Phi Sin Theta Psi_A Psi_B
// 2: Sin Phi Sin Theta Psi_A Psi_B
// 3. Cos Theta Psi_A Psi_B
// 4. G^nu^mu_ihat
// 5. F^nu^mu_ihat
KOKKOS_INLINE_FUNCTION
double
IntegrateMatrixSphericalTriangle(int a,
                                 int b,
                                 int basis,
                                 int t1,
                                 int t2,
                                 int t3,
                                 const HostArray1D<Real> &x,
                                 const HostArray1D<Real> &y,
                                 const HostArray1D<Real> &z,
                                 const HostArray1D<Real> &scheme_weights,
                                 const HostArray2D<Real> &scheme_points,
                                 int matrixnumber,
                                 int nu,
                                 int mu,
                                 int ihat) {

  double x1 = x(t1);
  double y1 = y(t1);
  double z1 = z(t1);

  double x2 = x(t2);
  double y2 = y(t2);
  double z2 = z(t2);

  double x3 = x(t3);
  double y3 = y(t3);
  double z3 = z(t3);

  double result{0.};

  // (0) Psi_A Psi_B
  if (matrixnumber == 0) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0),
                                                  scheme_points(i, 1), scheme_points(i, 2))) *
          FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1),
                          scheme_points(i, 2), basis) * scheme_weights(i);

    }
  }

    // (1) Cos Phi Sin Theta Psi_A Psi_B
  else if (matrixnumber == 1) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += CosPhiSinTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1),
                               scheme_points(i, 2)) *
          sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0),
                                            scheme_points(i, 1), scheme_points(i, 2))) *
          FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1),
                          scheme_points(i, 2), basis) * scheme_weights(i);
    }
  }

    // (2) Sin Phi Sin Theta Psi_A Psi_B
  else if (matrixnumber == 2) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += SinPhiSinTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1),
                               scheme_points(i, 2)) *
          sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0),
                                            scheme_points(i, 1), scheme_points(i, 2))) *
          FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1),
                          scheme_points(i, 2), basis) * scheme_weights(i);
    }
  }

    // (3) Cos Theta Psi_A Psi_B
  else if (matrixnumber == 3) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += CosTheta(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1),
                         scheme_points(i, 2)) *
          sqrt(CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0),
                                            scheme_points(i, 1), scheme_points(i, 2))) *
          FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1),
                          scheme_points(i, 2), basis) * scheme_weights(i);
    }
  }

    /* // (4) G^nu^mu_ihat
   else if (matrixnumber == 4) {
     for (size_t i = 0; i < scheme_points.size(); i++) {
       result += mom_by_energy(nu,
                               x1,
                               y1,
                               z1,
                               x2,
                               y2,
                               z2,
                               x3,
                               y3,
                               z3,
                               scheme_points(i, 0),
                               scheme_points(i, 1),
                               scheme_points(i, 2)) *
           mom_by_energy(mu,
                         x1,
                         y1,
                         z1,
                         x2,
                         y2,
                         z2,
                         x3,
                         y3,
                         z3,
                         scheme_points(i, 0),
                         scheme_points(i, 1),
                         scheme_points(i, 2)) *
           sqrt(CalculateDeterminantJacobian(x1,
                                             y1,
                                             z1,
                                             x2,
                                             y2,
                                             z2,
                                             x3,
                                             y3,
                                             z3,
                                             scheme_points(i, 0),
                                             scheme_points(i, 1),
                                             scheme_points(i, 2))) *
           FEMBasisA(a, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1), scheme_points(i, 2), basis) *
           PartialFEMBasisBwithoute(ihat,
                                    a,
                                    t1,
                                    t2,
                                    t3,
                                    x1,
                                    y1,
                                    z1,
                                    x2,
                                    y2,
                                    z2,
                                    x3,
                                    y3,
                                    z3,
                                    scheme_points(i, 0),
                                    scheme_points(i, 1),
                                    scheme_points(i, 2),
                                    basis) * scheme_weights(i);
     }
   } */

    // (5) F^nu^mu_ihat
  else if (matrixnumber == 5) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += mom_by_energy(nu,
                              x1,
                              y1,
                              z1,
                              x2,
                              y2,
                              z2,
                              x3,
                              y3,
                              z3,
                              scheme_points(i, 0),
                              scheme_points(i, 1),
                              scheme_points(i, 2)) *
          mom_by_energy(mu,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        x3,
                        y3,
                        z3,
                        scheme_points(i, 0),
                        scheme_points(i, 1),
                        scheme_points(i, 2)) *
          sqrt(CalculateDeterminantJacobian(x1,
                                            y1,
                                            z1,
                                            x2,
                                            y2,
                                            z2,
                                            x3,
                                            y3,
                                            z3,
                                            scheme_points(i, 0),
                                            scheme_points(i, 1),
                                            scheme_points(i, 2))) *
          FEMBasisABasisB(a, b, t1, t2, t3, scheme_points(i, 0), scheme_points(i, 1),
                          scheme_points(i, 2), basis) * scheme_weights(i) *
          mom_by_energy(ihat,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        x3,
                        y3,
                        z3,
                        scheme_points(i, 0),
                        scheme_points(i, 1),
                        scheme_points(i, 2)) * scheme_weights(i);
    }
  }

  result = 0.5 * result;

  return result;
}

// --------------------------------------------
// Calculate determinant of Jacobian term
KOKKOS_INLINE_FUNCTION
double
CalculateDeterminantJacobian(double x1,
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

  double result =
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
double
IntegrateMatrix(int a,                                    // matrix row (this is an angle pair index)
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
  double result = 0.;

  bool is_edge{false};
  HostArray2D<int> edge_triangles;
  FindTriangles(a, b, triangles, edge_triangles, is_edge);

  if (is_edge) {
    for (size_t i = 0; i < 6; i++) {
      if (edge_triangles(i, 0) >= 0) {
        int triangle_index_1 = edge_triangles(i, 0);
        int triangle_index_2 = edge_triangles(i, 1);
        int triangle_index_3 = edge_triangles(i, 2);

        double integrated_result =
            IntegrateMatrixSphericalTriangle(a, b, basis, triangle_index_1, triangle_index_2, triangle_index_3, x, y, z,
                                             scheme_weights, scheme_points, matrixchoice, nu, mu, ihat);
        result += integrated_result;
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------
// Calculate real spherical harmonics [arXiv:2212.01409, Eqn. (25)
//KOKKOS_INLINE_FUNCTION
double RealSphericalHarmonic(int l, int m, double phi, double theta) {
  std::complex<double> result(0., 0.);
  std::complex<double> sqrtminusone(0., 1.);
  if (m > 0) {
    result = (1. / sqrt(2.)) * (boost::math::spherical_harmonic<double, double>(l, m, theta, phi)
        + pow(-1., m) * boost::math::spherical_harmonic<double, double>(l, -m, theta, phi));
  } else if (m == 0) {
    result = boost::math::spherical_harmonic<double, double>(l, 0, theta, phi);
  } else {
    result = (1. / (sqrtminusone * sqrt(2.))) * (boost::math::spherical_harmonic<double, double>(l, -m, theta, phi)
        - pow(-1., m) * boost::math::spherical_harmonic<double, double>(l, m, theta, phi));
  }

  double imaginary_max = abs(imag(result));
  assert(imaginary_max < 1e-14);

  return real(result);
}

// ----------------------------------------------------------
// Find mass/stiffness and other associated matrices for FP_N
// The following matrices are implemented:
// 0: Psi_A Psi_B
// 1: Cos Phi Sin Theta Psi_A Psi_B
// 2: Sin Phi Sin Theta Psi_A Psi_B
// 3. Cos Theta Psi_A Psi_B
// 4. G^nu^mu_ihat
// 5. F^nu^mu_ihat
//KOKKOS_INLINE_FUNCTION
double IntegrateMatrixFPN(int la,
                          int ma,
                          int lb,
                          int mb,
                          const HostArray1D<Real> &scheme_weights,
                          const HostArray2D<Real> &scheme_points,
                          int matrixchoice, int nu,
                          int mu,
                          int ihat) {

  double result = 0.;

  if (matrixchoice == 0) {
    result = double((la == lb) * (ma == mb));
  } else if (matrixchoice == 1) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * cos(scheme_points(i, 0)) * sin(scheme_points(i, 1))
          * RealSphericalHarmonic(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * RealSphericalHarmonic(lb, mb, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  } else if (matrixchoice == 2) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * sin(scheme_points(i, 0)) * sin(scheme_points(i, 1))
          * RealSphericalHarmonic(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * RealSphericalHarmonic(lb, mb, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  } else if (matrixchoice == 3) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * cos(scheme_points(i, 1))
          * RealSphericalHarmonic(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * RealSphericalHarmonic(lb, mb, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  } else if (matrixchoice == 4) {

  } else if (matrixchoice == 5) {
    for (size_t i = 0; i < scheme_weights.size(); i++) {
      result += 4. * M_PI * mom_by_energy(nu, scheme_points(i, 0), scheme_points(i, 1)) *
          mom_by_energy(mu, scheme_points(i, 0), scheme_points(i, 1))
          * RealSphericalHarmonic(la, ma, scheme_points(i, 0), scheme_points(i, 1))
          * RealSphericalHarmonic(lb, mb, scheme_points(i, 0), scheme_points(i, 1))
          * mom_by_energy(ihat, scheme_points(i, 0), scheme_points(i, 1)) * scheme_weights(i);
    }
  }
  return result;
}
} // namespace radiationfemn