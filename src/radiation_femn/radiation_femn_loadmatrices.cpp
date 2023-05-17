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

}

// -------------------------------------------------
// Generate the matrices needed for FPN
void RadiationFEMN::LoadFPNMatrices() {
  std::cout << "Loading matrices for FPN ... " << std::endl;

  auto &lm_grid_ = angular_grid;

  // populate the angular grid with (l,m) values
  for (int l = 0; l <= lmax; l++) {
    for (int m = -l; m <= l; m++) {
      lm_grid_(l * l + (l + m), 0) = l;
      lm_grid_(l * l + (l + m), 1) = m;
    }
  }

  auto &mm_ = mass_matrix;
  //Populate the mass matrix
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; i < num_points; j++) {
      mm_(i,j) = IntegrateMatrixFPN(lm_grid_(i,0),lm_grid_(i,1),lm_grid_(j,0),lm_grid_(j,1),scheme_weights,scheme_points,0);
    }

  }
}
}  // namespace radiationfemn