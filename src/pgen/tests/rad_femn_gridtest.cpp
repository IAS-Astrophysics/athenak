//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_gridtest.cpp
//! \brief tests the geodesic grid and associated matrices for radiation FEM_N

// C++ headers
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>        // exp
#include <algorithm>    // max
#include <iomanip>
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

void ProblemGenerator::RadiationFEMNGridtest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The radiation FEM_N grid test can only be run with radiation-femn, but no " << std::endl
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string pathdir = pin->GetString("radiation-femn", "matrices_savedir");
  std::string filenamepart = "/femn_" + std::to_string(pmbp->pradfemn->num_points);
  if (pmbp->pradfemn->fpn) {
    filenamepart = "/fpn_" + std::to_string(pmbp->pradfemn->num_points);
  }

  // [1] save metadata, grid information and quadrature information
  std::ofstream fout(pathdir + filenamepart + "_metadata" + ".txt");
  fout << "FEMN_N metadata: " << std::endl;
  fout << std::endl;
  fout << "num_ref = " << pmbp->pradfemn->num_ref << std::endl;
  fout << "num_points = " << pmbp->pradfemn->num_points << std::endl;
  fout << "num_edges = " << pmbp->pradfemn->num_edges << std::endl;
  fout << "num_triangles = " << pmbp->pradfemn->num_triangles << std::endl;
  fout << "quadrature_num_points = " << pmbp->pradfemn->scheme_num_points << std::endl;
  fout << "quadrature_name = " << pmbp->pradfemn->scheme_name << std::endl;

  HostArray2D<Real> angular_grid_host;
  Kokkos::realloc(angular_grid_host, pmbp->pradfemn->num_points, 2);
  Kokkos::deep_copy(angular_grid_host, pmbp->pradfemn->angular_grid);
  std::ofstream fout2(pathdir + filenamepart + "_grid_coordinates" + ".txt");
  fout2 << "phi theta" << std::endl;
  for (size_t i = 0; i < pmbp->pradfemn->num_points; i++) {
    fout2 << angular_grid_host(i, 0) << " " << angular_grid_host(i, 1) << std::endl;
  }

  std::ofstream fout7(pathdir + filenamepart + "_quadrature_info" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->scheme_num_points; i++) {
    fout7 << pmbp->pradfemn->scheme_points(i, 0) << " " << pmbp->pradfemn->scheme_points(i, 1) << " " << pmbp->pradfemn->scheme_points(i, 2) << " "
          << pmbp->pradfemn->scheme_weights(i) << std::endl;
  }

  // [2] save mass matrix, mass matrix inverse
  HostArray2D<Real> mass_matrix_host;
  HostArray2D<Real> mass_matrix_inv_host;
  Kokkos::realloc(mass_matrix_host, pmbp->pradfemn->num_points, pmbp->pradfemn->num_points);
  Kokkos::realloc(mass_matrix_inv_host, pmbp->pradfemn->num_points, pmbp->pradfemn->num_points);
  Kokkos::deep_copy(mass_matrix_host, pmbp->pradfemn->mass_matrix);
  Kokkos::deep_copy(mass_matrix_inv_host, pmbp->pradfemn->mass_matrix_inv);

  double sum = 0.;
  std::ofstream fout3(pathdir + filenamepart + "_mass_matrix" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout3 << mass_matrix_host(i, j) << " ";
      sum += mass_matrix_host(i, j);
    }
    fout3 << std::endl;
  }
  fout << "sum of mass matrix = " << sum << std::endl;

  std::ofstream fout3i(pathdir + filenamepart + "_mass_matrix_inv" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout3i << mass_matrix_inv_host(i, j) << " ";
      sum += mass_matrix_inv_host(i, j);
    }
    fout3i << std::endl;
  }


  // [3] save stiffness matrices (no multiplication by inv of mass matrix)
  HostArray2D<Real> stiffness_matrix_x_host;
  HostArray2D<Real> stiffness_matrix_y_host;
  HostArray2D<Real> stiffness_matrix_z_host;
  Kokkos::realloc(stiffness_matrix_x_host, pmbp->pradfemn->num_points, pmbp->pradfemn->num_points);
  Kokkos::realloc(stiffness_matrix_y_host, pmbp->pradfemn->num_points, pmbp->pradfemn->num_points);
  Kokkos::realloc(stiffness_matrix_z_host, pmbp->pradfemn->num_points, pmbp->pradfemn->num_points);
  Kokkos::deep_copy(stiffness_matrix_x_host, pmbp->pradfemn->stiffness_matrix_x);
  Kokkos::deep_copy(stiffness_matrix_y_host, pmbp->pradfemn->stiffness_matrix_y);
  Kokkos::deep_copy(stiffness_matrix_z_host, pmbp->pradfemn->stiffness_matrix_z);

  std::ofstream fout4(pathdir + filenamepart + "_stiffness_x" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout4 << stiffness_matrix_x_host(i, j) << " ";
    }
    fout4 << std::endl;
  }

  std::ofstream fout5(pathdir + filenamepart + "_stiffness_y" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout5 << stiffness_matrix_y_host(i, j) << " ";
    }
    fout5 << std::endl;
  }

  std::ofstream fout6(pathdir + filenamepart + "_stiffness_z" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout6 << stiffness_matrix_z_host(i, j) << " ";
    }
    fout6 << std::endl;
  }

  // [4] save P matrices (after multiplying by inv of mass matrix)
  HostArray3D<Real> P_matrix_host;
  Kokkos::realloc(P_matrix_host, 4, pmbp->pradfemn->num_points, pmbp->pradfemn->num_points);
  Kokkos::deep_copy(P_matrix_host, pmbp->pradfemn->P_matrix);
  std::ofstream fout9(pathdir + filenamepart + "_P_matrix_0" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout9 << P_matrix_host(0, i, j) << " ";
    }
    fout9 << std::endl;
  }

  std::ofstream fout10(pathdir + filenamepart + "_P_matrix_1" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout10 << P_matrix_host(1, i, j) << " ";
    }
    fout10 << std::endl;
  }

  std::ofstream fout11(pathdir + filenamepart + "_P_matrix_2" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout11 << P_matrix_host(2, i, j) << " ";
    }
    fout11 << std::endl;
  }

  std::ofstream fout12(pathdir + filenamepart + "_P_matrix_3" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout12 << P_matrix_host(3, i, j) << " ";
    }
    fout12 << std::endl;
  }

  // save e-matrix
  HostArray1D<Real> e_source_host;
  Kokkos::realloc(e_source_host, pmbp->pradfemn->num_points);
  Kokkos::deep_copy(e_source_host, pmbp->pradfemn->e_source);
  std::ofstream fout13(pathdir + filenamepart + "_e_matrix" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    fout13 << e_source_host(i) << std::endl;
  }

  // save S-matrix
  HostArray2D<Real> S_source_host;
  Kokkos::realloc(S_source_host, pmbp->pradfemn->num_points, pmbp->pradfemn->num_points);
  Kokkos::deep_copy(S_source_host, pmbp->pradfemn->S_source);
  std::ofstream fout14(pathdir + filenamepart + "_s_matrix.txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout14 << S_source_host(i, j) << " ";
    }
    fout14 << std::endl;
  }

  return;
}