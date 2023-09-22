//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_compute_pmatrices.cpp
//  \brief generate and load matrices for the angular grid

#include <iostream>

#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

void RadiationFEMN::ComputePMatrices() {
  std::cout << "Computing P matrices and modified P matrices ..." << std::endl;

  HostArray2D<Real> mass_temp;
  HostArray2D<Real> mass_inv_temp;
  HostArray2D<Real> stiff_x_temp;
  HostArray2D<Real> stiff_y_temp;
  HostArray2D<Real> stiff_z_temp;
  HostArray2D<Real> temp_array;
  HostArray2D<Real> temp_array_corrected;

  Kokkos::realloc(mass_temp, num_points, num_points);
  Kokkos::realloc(mass_inv_temp, num_points, num_points);
  Kokkos::realloc(stiff_x_temp, num_points, num_points);
  Kokkos::realloc(stiff_y_temp, num_points, num_points);
  Kokkos::realloc(stiff_z_temp, num_points, num_points);
  Kokkos::realloc(temp_array, num_points, num_points);
  Kokkos::realloc(temp_array_corrected, num_points, num_points);

  Kokkos::deep_copy(mass_temp, mass_matrix);
  Kokkos::deep_copy(stiff_x_temp, stiffness_matrix_x);
  Kokkos::deep_copy(stiff_y_temp, stiffness_matrix_y);
  Kokkos::deep_copy(stiff_z_temp, stiffness_matrix_z);

  // store inverse of lumped mass matrix in temp_array
  radiationfemn::LUInverse(mass_temp, mass_inv_temp);

  // M^-1 M
  radiationfemn::MatMultiply(mass_inv_temp, mass_temp, temp_array);
  auto P_matrix_0 = Kokkos::subview(P_matrix, 0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_0, temp_array);

  auto Pmod_matrix_0 = Kokkos::subview(Pmod_matrix, 0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_0, temp_array);

  // stilde-x matrix
  radiationfemn::MatMultiply(mass_inv_temp, stiff_x_temp, temp_array);
  auto P_matrix_1 = Kokkos::subview(P_matrix, 1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_1, temp_array);

  radiationfemn::ZeroSpeedCorrection(temp_array, temp_array_corrected, 1. / sqrt(3.));
  auto Pmod_matrix_1 = Kokkos::subview(Pmod_matrix, 1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_1, temp_array_corrected);

  // stilde-y matrix
  radiationfemn::MatMultiply(mass_inv_temp, stiff_y_temp, temp_array);
  auto P_matrix_2 = Kokkos::subview(P_matrix, 2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_2, temp_array);

  radiationfemn::ZeroSpeedCorrection(temp_array, temp_array_corrected, 1. / sqrt(3.));
  auto Pmod_matrix_2 = Kokkos::subview(Pmod_matrix, 2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_2, temp_array_corrected);

  // stilde-z matrix index
  radiationfemn::MatMultiply(mass_inv_temp, stiff_z_temp, temp_array);
  auto P_matrix_3 = Kokkos::subview(P_matrix, 3, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_3, temp_array);

  radiationfemn::ZeroSpeedCorrection(temp_array, temp_array_corrected, 1. / sqrt(3.));
  auto Pmod_matrix_3 = Kokkos::subview(Pmod_matrix, 3, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_3, temp_array_corrected);

}

}