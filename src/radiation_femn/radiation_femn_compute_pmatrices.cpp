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
#include "radiation_femn_matinv.hpp"

namespace radiationfemn {

// compute inverse of mass matrix (on host, then copy to device)
void RadiationFEMN::ComputeMassInverse() {

  std::cout << "Computing the inverse of the mass matrix ..." << std::endl;

  HostArray2D<Real> mass_matrix_host;
  HostArray2D<Real> mass_matrix_inv_host;
  HostArray2D<Real> lu_matrix_host;
  HostArray1D<Real> b_matrix_host;
  HostArray1D<Real> x_matrix_host;
  HostArray1D<int> pivots_host;

  Kokkos::realloc(mass_matrix_host, num_points, num_points);
  Kokkos::realloc(mass_matrix_inv_host, num_points, num_points);
  Kokkos::realloc(lu_matrix_host, num_points, num_points);
  Kokkos::realloc(b_matrix_host, num_points);
  Kokkos::realloc(x_matrix_host, num_points);
  Kokkos::realloc(pivots_host, num_points - 1);

  Kokkos::deep_copy(mass_matrix_host, mass_matrix);
  Kokkos::deep_copy(lu_matrix_host, mass_matrix);
  Kokkos::deep_copy(b_matrix_host, 0.);
  Kokkos::deep_copy(x_matrix_host, 0.);
  Kokkos::deep_copy(pivots_host, 0.);

  radiationfemn::LUInv<HostArray2D<Real>, HostArray1D<Real>, HostArray1D<int>>(mass_matrix_host, mass_matrix_inv_host, lu_matrix_host,
                                                                               x_matrix_host, b_matrix_host, pivots_host);
  Kokkos::deep_copy(mass_matrix_inv, mass_matrix_inv_host);

}
// compute P and Pmod matrices (on host, then copy to device)
void RadiationFEMN::ComputePMatrices() {

  std::cout << "Computing P matrices and modified P matrices ..." << std::endl;

  DvceArray2D<Real> temp_array;
  HostArray2D<Real> temp_array_host;
  HostArray2D<Real> temp_array_corrected;

  Kokkos::realloc(temp_array, num_points, num_points);
  Kokkos::realloc(temp_array_host, num_points, num_points);
  Kokkos::realloc(temp_array_corrected, num_points, num_points);

  // M^-1 M
  radiationfemn::MatMultiplyDvce(mass_matrix_inv, mass_matrix, temp_array);
  auto P_matrix_0 = Kokkos::subview(P_matrix, 0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_0, temp_array);
  auto Pmod_matrix_0 = Kokkos::subview(Pmod_matrix, 0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_0, temp_array);

  // stilde-x matrix
  radiationfemn::MatMultiplyDvce(mass_matrix_inv, stiffness_matrix_x, temp_array);
  auto P_matrix_1 = Kokkos::subview(P_matrix, 1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_1, temp_array);
  Kokkos::deep_copy(temp_array_host, temp_array);

  radiationfemn::ZeroSpeedCorrection(temp_array_host, temp_array_corrected, 1. / sqrt(3.));
  auto Pmod_matrix_1 = Kokkos::subview(Pmod_matrix, 1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_1, temp_array_corrected);

  // stilde-y matrix
  radiationfemn::MatMultiplyDvce(mass_matrix_inv, stiffness_matrix_y, temp_array);
  auto P_matrix_2 = Kokkos::subview(P_matrix, 2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_2, temp_array);
  Kokkos::deep_copy(temp_array_host, temp_array);

  radiationfemn::ZeroSpeedCorrection(temp_array_host, temp_array_corrected, 1. / sqrt(3.));
  auto Pmod_matrix_2 = Kokkos::subview(Pmod_matrix, 2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_2, temp_array_corrected);

  // stilde-z matrix index
  radiationfemn::MatMultiplyDvce(mass_matrix_inv, stiffness_matrix_z, temp_array);
  auto P_matrix_3 = Kokkos::subview(P_matrix, 3, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_3, temp_array);
  Kokkos::deep_copy(temp_array_host, temp_array);

  radiationfemn::ZeroSpeedCorrection(temp_array_host, temp_array_corrected, 1. / sqrt(3.));
  auto Pmod_matrix_3 = Kokkos::subview(Pmod_matrix, 3, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_3, temp_array_corrected);

}

}