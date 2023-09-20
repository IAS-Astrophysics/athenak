//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_loadmatrices.cpp
//  \brief generate and load matrices for the angular grid

#include <iostream>

#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

void RadiationFEMN::LoadFEMNMatrices() {
  std::cout << "Geodesic grid: Generating ..." << std::endl;

  HostArray1D<Real> x;
  HostArray1D<Real> y;
  HostArray1D<Real> z;
  HostArray1D<Real> r;
  HostArray1D<Real> theta;
  HostArray1D<Real> phi;
  HostArray2D<int> edges;
  HostArray2D<int> triangles;

  GeodesicGridBaseGenerate(num_ref, num_points, num_edges, num_triangles, x, y, z, r, theta, phi, edges, triangles);

  std::cout << "Geodesic grid: Base grid generated! num_ref = " << num_ref << ", num_points = " << num_points
            << ", num_edges = " << num_edges << ", num_triangles = "
            << num_triangles << std::endl;

  if (refinement_level > 0) {
    for (size_t i = num_ref; i < refinement_level; i++) {
      GeodesicGridRefine(num_ref, num_points, num_edges, num_triangles, x, y, z, r, theta, phi, edges, triangles);
      std::cout << "Geodesic grid: Refined grid generated! num_ref = " << num_ref << ", num_points = " << num_points
                << ", num_edges = " << num_edges
                << ", num_triangles = " << num_triangles << std::endl;
    }
  }

  for (size_t i = 0; i < num_points; i++) {
    angular_grid(i, 0) = phi(i);
    angular_grid(i, 1) = theta(i);
  }

  auto &mm_ = mass_matrix;
  auto &sx_ = stiffness_matrix_x;
  auto &sy_ = stiffness_matrix_y;
  auto &sz_ = stiffness_matrix_z;
  auto &fmatrix_ = F_matrix;

  //Populate the mass matrix
  std::cout << "Computing the mass matrix ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      mm_(i, j) = radiationfemn::IntegrateMatrixFEMN(i,
                                                     j,
                                                     basis,
                                                     x,
                                                     y,
                                                     z,
                                                     scheme_weights,
                                                     scheme_points,
                                                     triangles,
                                                     0,
                                                     -42,
                                                     -42,
                                                     -42);
    }
  }

  // compute stiffness matrices
  std::cout << "Computing the stiffness matrices ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      sx_(i, j) = radiationfemn::IntegrateMatrixFEMN(i,
                                                     j,
                                                     basis,
                                                     x,
                                                     y,
                                                     z,
                                                     scheme_weights,
                                                     scheme_points,
                                                     triangles,
                                                     1,
                                                     -42,
                                                     -42,
                                                     -42);
      sy_(i, j) = radiationfemn::IntegrateMatrixFEMN(i,
                                                     j,
                                                     basis,
                                                     x,
                                                     y,
                                                     z,
                                                     scheme_weights,
                                                     scheme_points,
                                                     triangles,
                                                     2,
                                                     -42,
                                                     -42,
                                                     -42);
      sz_(i, j) = radiationfemn::IntegrateMatrixFEMN(i,
                                                     j,
                                                     basis,
                                                     x,
                                                     y,
                                                     z,
                                                     scheme_weights,
                                                     scheme_points,
                                                     triangles,
                                                     3,
                                                     -42,
                                                     -42,
                                                     -42);
    }
  }

  // compute the F matrices
  std::cout << "Computing the F matrices ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      for (int nu = 0; nu < 4; nu++) {
        for (int mu = 0; mu < 4; mu++) {
          fmatrix_(nu, mu, 0, i, j) =
              radiationfemn::IntegrateMatrixFEMN(i,
                                                 j,
                                                 basis,
                                                 x,
                                                 y,
                                                 z,
                                                 scheme_weights,
                                                 scheme_points,
                                                 triangles,
                                                 5,
                                                 nu,
                                                 mu,
                                                 1);
          fmatrix_(nu, mu, 1, i, j) = radiationfemn::IntegrateMatrixFEMN(i,
                                                                         j,
                                                                         basis,
                                                                         x,
                                                                         y,
                                                                         z,
                                                                         scheme_weights,
                                                                         scheme_points,
                                                                         triangles,
                                                                         5,
                                                                         nu,
                                                                         mu,
                                                                         2);
          fmatrix_(nu, mu, 2, i, j) = radiationfemn::IntegrateMatrixFEMN(i,
                                                                         j,
                                                                         basis,
                                                                         x,
                                                                         y,
                                                                         z,
                                                                         scheme_weights,
                                                                         scheme_points,
                                                                         triangles,
                                                                         5,
                                                                         nu,
                                                                         mu,
                                                                         3);
        }
      }
    }
  }

}

/* Generate matrices which are needed for the FP_N scheme
 *
 * Loads the mass matrix, stiffness matrices, F and G matrices
 */
void RadiationFEMN::LoadFPNMatrices() {

  std::cout << "Loading matrices for FPN ... " << std::endl;

  auto &mm_ = mass_matrix;
  auto &sx_ = stiffness_matrix_x;
  auto &sy_ = stiffness_matrix_y;
  auto &sz_ = stiffness_matrix_z;

  // populate angular grid with (l,m) values
  auto &lm_grid_ = angular_grid;
  for (int l = 0; l <= lmax; l++) {
    for (int m = -l; m <= l; m++) {
      lm_grid_(l * l + (l + m), 0) = l;
      lm_grid_(l * l + (l + m), 1) = m;
    }
  }

  // temporary host arrays
  HostArray2D<Real> temp_matrix;
  HostArray2D<Real> temp_matrix_2;
  HostArray2D<Real> temp_matrix_3;
  Kokkos::realloc(temp_matrix, num_points, num_points);
  Kokkos::realloc(temp_matrix_2, num_points, num_points);
  Kokkos::realloc(temp_matrix_3, num_points, num_points);

  // Compute the matrix
  std::cout << "Computing the mass matrix ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      temp_matrix(i, j) = radiationfemn::IntegrateMatrixFPN(int(lm_grid_(i, 0)), int(lm_grid_(i, 1)), int(lm_grid_(j, 0)), int(lm_grid_(j, 1)),
                                                    scheme_weights, scheme_points, 0, -42, -42, -42);
    }
  }
  Kokkos::deep_copy(mm_, temp_matrix);

  // compute stiffness matrices
  std::cout << "Computing the stiffness matrices ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      temp_matrix(i, j) = radiationfemn::IntegrateMatrixFPN(int(lm_grid_(i, 0)), int(lm_grid_(i, 1)), int(lm_grid_(j, 0)), int(lm_grid_(j, 1)),
                                                    scheme_weights, scheme_points, 1, -42, -42, -42);
      temp_matrix_2(i, j) = radiationfemn::IntegrateMatrixFPN(int(lm_grid_(i, 0)), int(lm_grid_(i, 1)), int(lm_grid_(j, 0)), int(lm_grid_(j, 1)),
                                                    scheme_weights, scheme_points, 2, -42, -42, -42);
      temp_matrix_3(i, j) = radiationfemn::IntegrateMatrixFPN(int(lm_grid_(i, 0)), int(lm_grid_(i, 1)), int(lm_grid_(j, 0)), int(lm_grid_(j, 1)),
                                                    scheme_weights, scheme_points, 3, -42, -42, -42);
    }
  }
  Kokkos::deep_copy(sx_, temp_matrix);
  Kokkos::deep_copy(sy_, temp_matrix_2);
  Kokkos::deep_copy(sz_, temp_matrix_3);

  // compute the F matrices
  /*
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      for (int nu = 0; nu < 4; nu++) {
        for (int mu = 0; mu < 4; mu++) {
          fmatrix_(nu, mu, 0, i, j) = radiationfemn::IntegrateMatrixFPN(int(lm_grid_(i, 0)),
                                                                        int(lm_grid_(i, 1)),
                                                                        int(lm_grid_(j, 0)),
                                                                        int(lm_grid_(j, 1)),
                                                                        scheme_weights,
                                                                        scheme_points,
                                                                        5, nu, mu, 1);
          fmatrix_(nu, mu, 1, i, j) = radiationfemn::IntegrateMatrixFPN(int(lm_grid_(i, 0)),
                                                                        int(lm_grid_(i, 1)),
                                                                        int(lm_grid_(j, 0)),
                                                                        int(lm_grid_(j, 1)),
                                                                        scheme_weights,
                                                                        scheme_points,
                                                                        5, nu, mu, 2);
          fmatrix_(nu, mu, 2, i, j) = radiationfemn::IntegrateMatrixFPN(int(lm_grid_(i, 0)),
                                                                        int(lm_grid_(i, 1)),
                                                                        int(lm_grid_(j, 0)),
                                                                        int(lm_grid_(j, 1)),
                                                                        scheme_weights,
                                                                        scheme_points,
                                                                        5, nu, mu, 3);
        }
      }
    }
  }*/

}
}  // namespace radiationfemn