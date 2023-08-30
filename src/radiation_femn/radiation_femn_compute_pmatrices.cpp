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

void RadiationFEMN::ComputePMatrix() {
  std::cout << "Computing P matrices ..." << std::endl;

  Kokkos::deep_copy(P_matrix, 0.);

  DvceArray2D<Real> temp_array;
  DvceArray2D<Real> temp_array_2;
  Kokkos::realloc(temp_array, num_points, num_points);
  Kokkos::realloc(temp_array_2, num_points, num_points);
  Kokkos::deep_copy(temp_array, 0.);

  // store inverse of lumped mass matrix in temp_array
  radiationfemn::LUInverse(mass_matrix_lumped, temp_array);

  // mass-matrix index (indentity matrix)
  for (int i = 0; i < num_points; i++) {
    P_matrix(0,i,i) = 1.;
  }

  // stilde-x matrix index
  Kokkos::deep_copy(temp_array_2, 0.);
  radiationfemn::MatMultiply(temp_array, stiffness_matrix_x, temp_array_2);
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      P_matrix(1,i,j) = temp_array_2(i,j);
    }
  }

  // stilde-y matrix index
  Kokkos::deep_copy(temp_array_2, 0.);
  radiationfemn::MatMultiply(temp_array, stiffness_matrix_y, temp_array_2);
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      P_matrix(2,i,j) = temp_array_2(i,j);
    }
  }

  // stilde-z matrix index
  Kokkos::deep_copy(temp_array_2, 0.);
  radiationfemn::MatMultiply(temp_array, stiffness_matrix_z, temp_array_2);
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      P_matrix(3,i,j) = temp_array_2(i,j);
    }
  }

}
}