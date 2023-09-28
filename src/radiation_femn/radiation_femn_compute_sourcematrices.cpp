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

  auto mm_ = pmy_pack->pradfemn->mass_matrix;

  HostArray2D<Real> mass_inv_temp;
  Kokkos::realloc(mass_inv_temp, num_points, num_points);
  radiationfemn::LUInverse(mm_, mass_inv_temp);

  HostArray1D<Real> e_source_temp;
  Kokkos::realloc(e_source_temp, num_points);
  HostArray2D<Real> S_source_temp;
  Kokkos::realloc(S_source_temp, num_points, num_points);

}

}