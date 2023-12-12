//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file  radiation_femn_compute_sourcematrices.cpp
//  \brief compute matrices for the source terms

#include <iostream>

#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

void RadiationFEMN::ComputeSourceMatrices() {
  std::cout << "Computing source matrices ..." << std::endl;

  auto mm_ = mass_matrix;
  auto e_source_ = e_source;
  auto S_source_ = S_source;

  DvceArray2D<Real> mass_inv_temp;
  Kokkos::realloc(mass_inv_temp, num_points, num_points);
  radiationfemn::LUInverse(mm_, mass_inv_temp);

  DvceArray1D<Real> e_source_temp;
  DvceArray1D<Real> e_source_temp_mod;
  DvceArray2D<Real> S_source_temp;
  DvceArray2D<Real> S_source_temp_mod;
  Kokkos::realloc(e_source_temp, num_points);
  Kokkos::realloc(e_source_temp_mod, num_points);
  Kokkos::realloc(S_source_temp, num_points, num_points);
  Kokkos::realloc(S_source_temp_mod, num_points, num_points);

  Kokkos::deep_copy(e_source_temp, e_source_);
  Kokkos::deep_copy(e_source_nominv, e_source_);

  Kokkos::deep_copy(e_source_temp_mod, 0.);
  Kokkos::deep_copy(S_source_temp, 0.);
  Kokkos::deep_copy(S_source_temp_mod, 0.);

  std::cout << "Constructing the source matrices ..." << std::endl;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      e_source_temp_mod(i) += mass_inv_temp(i,j) * e_source_temp(j);
      S_source_temp(i,j) = e_source_temp(i) * e_source_temp(j);
    }
  }
  radiationfemn::MatMultiplyDvce(mass_inv_temp, S_source_temp, S_source_temp_mod);

  Kokkos::deep_copy(e_source_, e_source_temp_mod);
  Kokkos::deep_copy(S_source_, S_source_temp_mod);

}

}