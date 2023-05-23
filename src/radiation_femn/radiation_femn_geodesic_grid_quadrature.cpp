//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_quadrature.cpp
//  \brief contains T2 quadrature weights. DO NOT EDIT.

#include <fstream>
#include <string>
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

void LoadQuadrature(std::string &scheme_name,
                    int &scheme_num_points,
                    HostArray1D<Real> &scheme_weights,
                    HostArray2D<Real> &scheme_points) {

  double value;
  Kokkos::realloc(scheme_weights, scheme_num_points);
  Kokkos::realloc(scheme_points, scheme_num_points, 3);

  std::string file_path = __FILE__;
  std::string path = file_path.substr(0, file_path.find_last_of("/\\")) + "/quadrature/";
  std::cout << "Quadrature data path = " << path << std::endl;
  std::string filepath = path + scheme_name + "_" + std::to_string(scheme_num_points) + ".txt";
  std::ifstream file;
  file.open(filepath);
  if (!file) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The location of the quadrature rule cannot be found. Please check that the file exists"
              << " is entered correctly: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < scheme_num_points; i++) {
    file >> value;
    scheme_points(i, 0) = value;
    file >> value;
    scheme_points(i, 1) = value;
    file >> value;
    scheme_points(i, 2) = value;
    file >> value;
    scheme_weights(i) = value;

  }
  file.close();

}
}