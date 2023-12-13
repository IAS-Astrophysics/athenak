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

  HostArray2D<Real> temp_angular_grid;
  Kokkos::realloc(temp_angular_grid, num_points, 2);
  for (size_t i = 0; i < num_points; i++) {
    temp_angular_grid(i, 0) = phi(i);
    temp_angular_grid(i, 1) = theta(i);
  }
  Kokkos::deep_copy(angular_grid, temp_angular_grid);

  auto &mm_ = mass_matrix;
  auto &sx_ = stiffness_matrix_x;
  auto &sy_ = stiffness_matrix_y;
  auto &sz_ = stiffness_matrix_z;
  auto &fmatrix_ = F_matrix;
  auto &gmatrix_ = G_matrix;
  auto &e_source_ = e_source;
  auto &Q_matrix_ = Q_matrix;

  // temporary host arrays
  HostArray2D<Real> temp_matrix;
  HostArray2D<Real> temp_matrix_2;
  HostArray2D<Real> temp_matrix_3;
  HostArray5D<Real> temp_array_5d;
  Kokkos::realloc(temp_matrix, num_points, num_points);
  Kokkos::realloc(temp_matrix_2, num_points, num_points);
  Kokkos::realloc(temp_matrix_3, num_points, num_points);
  Kokkos::realloc(temp_array_5d, 4, 4, 3, num_points, num_points);

  // compute mass matrix
  std::cout << "Computing the mass matrix (FEM) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      temp_matrix(i, j) = radiationfemn::IntegrateMatrixFEMN(i, j, basis, x, y, z, scheme_weights, scheme_points, triangles, 0, -42, -42, -42);
    }
  }
  Kokkos::deep_copy(mm_, temp_matrix);

  // compute stiffness matrices
  std::cout << "Computing the stiffness matrices (FEM) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      temp_matrix(i, j) = radiationfemn::IntegrateMatrixFEMN(i, j, basis, x, y, z, scheme_weights, scheme_points, triangles, 1, -42, -42, -42);

      temp_matrix_2(i, j) = radiationfemn::IntegrateMatrixFEMN(i, j, basis, x, y, z, scheme_weights, scheme_points, triangles, 2, -42, -42, -42);

      temp_matrix_3(i, j) = radiationfemn::IntegrateMatrixFEMN(i, j, basis, x, y, z, scheme_weights, scheme_points, triangles, 3, -42, -42, -42);
    }
  }
  Kokkos::deep_copy(sx_, temp_matrix);
  Kokkos::deep_copy(sy_, temp_matrix_2);
  Kokkos::deep_copy(sz_, temp_matrix_3);

  // compute the F matrices
  std::cout << "Computing the F matrices (FEM) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      for (int nu = 0; nu < 4; nu++) {
        for (int mu = nu; mu < 4; mu++) {
          temp_array_5d(nu, mu, 0, i, j) = radiationfemn::IntegrateMatrixFEMN(i, j, basis, x, y, z, scheme_weights, scheme_points, triangles, 5, nu, mu, 1);
          temp_array_5d(mu, nu, 0, i, j) = temp_array_5d(nu, mu, 0, i, j);

          temp_array_5d(nu, mu, 1, i, j) = radiationfemn::IntegrateMatrixFEMN(i, j, basis, x, y, z, scheme_weights, scheme_points, triangles, 5, nu, mu, 2);
          temp_array_5d(mu, nu, 1, i, j) = temp_array_5d(nu, mu, 1, i, j);

          temp_array_5d(nu, mu, 2, i, j) = radiationfemn::IntegrateMatrixFEMN(i, j, basis, x, y, z, scheme_weights, scheme_points, triangles, 5, nu, mu, 3);
          temp_array_5d(mu, nu, 2, i, j) = temp_array_5d(nu, mu, 2, i, j);
        }
      }
    }
  }
  Kokkos::deep_copy(temp_array_5d, fmatrix_);

  HostArray1D<Real> e_source_temp;
  Kokkos::realloc(e_source_temp, num_points);
  HostArray2D<Real> Q_matrix_temp;
  Kokkos::realloc(Q_matrix_temp, 4, num_points);

  std::cout << "Computing the e-matrix (FEM) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    e_source_temp(i) = radiationfemn::IntegrateMatrixFEMN(i, i, basis, x, y, z, scheme_weights, scheme_points, triangles, 6, -42, -42, -42);
    Q_matrix_temp(0, i) = e_source_temp(i);
  }
  Kokkos::deep_copy(e_source_, e_source_temp);

  std::cout << "Computing the matrices for clp limiter (FEM) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    Q_matrix_temp(1, i) = radiationfemn::IntegrateMatrixFEMN(i, i, basis, x, y, z, scheme_weights, scheme_points, triangles, 7, -42, -42, -42);
    Q_matrix_temp(2, i) = radiationfemn::IntegrateMatrixFEMN(i, i, basis, x, y, z, scheme_weights, scheme_points, triangles, 8, -42, -42, -42);
    Q_matrix_temp(3, i) = radiationfemn::IntegrateMatrixFEMN(i, i, basis, x, y, z, scheme_weights, scheme_points, triangles, 9, -42, -42, -42);
  }
  Kokkos::deep_copy(Q_matrix_, Q_matrix_temp);

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
  auto &fmatrix_ = F_matrix;
  auto &gmatrix_ = G_matrix;
  auto &e_source_ = e_source;

  // populate angular grid with (l,m) values
  auto &lm_grid_ = angular_grid;
  HostArray2D<Real> temp_angular_grid;
  Kokkos::realloc(temp_angular_grid, num_points, 2);
  for (int l = 0; l <= lmax; l++) {
    for (int m = -l; m <= l; m++) {
      temp_angular_grid(l * l + (l + m), 0) = l;
      temp_angular_grid(l * l + (l + m), 1) = m;
    }
  }
  Kokkos::deep_copy(angular_grid, temp_angular_grid);

  // temporary host arrays
  HostArray2D<Real> temp_matrix;
  HostArray2D<Real> temp_matrix_2;
  HostArray2D<Real> temp_matrix_3;
  HostArray5D<Real> temp_array_5d;
  Kokkos::realloc(temp_matrix, num_points, num_points);
  Kokkos::realloc(temp_matrix_2, num_points, num_points);
  Kokkos::realloc(temp_matrix_3, num_points, num_points);
  Kokkos::realloc(temp_array_5d, 4, 4, 3, num_points, num_points);

  // Compute the mass matrix
  std::cout << "Computing the mass matrix (FPN) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      temp_matrix(i, j) =
          radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                            scheme_weights, scheme_points, 0, -42, -42, -42);
    }
  }
  Kokkos::deep_copy(mm_, temp_matrix);

  // compute stiffness matrices
  std::cout << "Computing the stiffness matrices (FPN) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      temp_matrix(i, j) =
          radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                            scheme_weights, scheme_points, 1, -42, -42, -42);
      temp_matrix_2(i, j) =
          radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                            scheme_weights, scheme_points, 2, -42, -42, -42);
      temp_matrix_3(i, j) =
          radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                            scheme_weights, scheme_points, 3, -42, -42, -42);
    }
  }
  Kokkos::deep_copy(sx_, temp_matrix);
  Kokkos::deep_copy(sy_, temp_matrix_2);
  Kokkos::deep_copy(sz_, temp_matrix_3);

  // compute the F matrices
  std::cout << "Computing the F matrices (FPN) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      for (int nu = 0; nu < 4; nu++) {
        for (int mu = nu; mu < 4; mu++) {
          temp_array_5d(nu, mu, 0, i, j) =
              radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                                scheme_weights, scheme_points, 5, nu, mu, 1);
          temp_array_5d(mu, nu, 0, i, j) = temp_array_5d(nu, mu, 0, i, j);

          temp_array_5d(nu, mu, 1, i, j) =
              radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                                scheme_weights, scheme_points, 5, nu, mu, 2);
          temp_array_5d(mu, nu, 1, i, j) = temp_array_5d(nu, mu, 1, i, j);

          temp_array_5d(nu, mu, 2, i, j) =
              radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                                scheme_weights, scheme_points, 5, nu, mu, 3);
          temp_array_5d(mu, nu, 2, i, j) = temp_array_5d(nu, mu, 2, i, j);
        }
      }
    }
  }
  Kokkos::deep_copy(fmatrix_, temp_array_5d);

  // compute the G matrices
  std::cout << "Computing the G matrices (FPN) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      for (int nu = 0; nu < 4; nu++) {
        for (int mu = nu; mu < 4; mu++) {
          temp_array_5d(nu, mu, 0, i, j) =
              radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                                scheme_weights, scheme_points, 4, nu, mu, 1);
          temp_array_5d(mu, nu, 0, i, j) = temp_array_5d(nu, mu, 0, i, j);

          temp_array_5d(nu, mu, 1, i, j) =
              radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                                scheme_weights, scheme_points, 4, nu, mu, 2);
          temp_array_5d(mu, nu, 1, i, j) = temp_array_5d(nu, mu, 1, i, j);

          temp_array_5d(nu, mu, 2, i, j) =
              radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(j, 0)), int(temp_angular_grid(j, 1)),
                                                scheme_weights, scheme_points, 4, nu, mu, 3);
          temp_array_5d(mu, nu, 2, i, j) = temp_array_5d(nu, mu, 2, i, j);
        }
      }
    }
  }
  Kokkos::deep_copy(gmatrix_, temp_array_5d);

  HostArray1D<Real> e_source_temp;
  Kokkos::realloc(e_source_temp, num_points);

  std::cout << "Computing the e-matrix (FPN) ... " << std::endl;
  for (int i = 0; i < num_points; i++) {
    e_source_temp(i) =
        radiationfemn::IntegrateMatrixFPN(int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)), int(temp_angular_grid(i, 0)), int(temp_angular_grid(i, 1)),
                                          scheme_weights, scheme_points, 6, -42, -42, -42);
  }
  Kokkos::deep_copy(e_source_, e_source_temp);

}
}  // namespace radiationfemn