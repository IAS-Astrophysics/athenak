//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
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

void ProblemGenerator::RadiationFEMNGridtestFPN(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl << "The radiation FP_N grid test can only be run with radiation-femn, but no "
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pradfemn->fpn != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl << "The radiation FP_N grid test can only be run with fpn=1 in parameter file!"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // save directory for matrices
  std::string pathdir = pin->GetString("radiation-femn", "savedir");
  std::string filenamepart = "/fpn_lmax_" + std::to_string(pmbp->pradfemn->lmax);

  // save metadata
  std::ofstream fout(pathdir + filenamepart + "_metadata" + ".txt");
  fout << "FP_N metadata: " << std::endl;
  fout << std::endl;
  fout << "lmax = " << pmbp->pradfemn->lmax << std::endl;
  fout << "num_points = " << pmbp->pradfemn->num_points << std::endl;
  fout << "quadrature_num_points = " << pmbp->pradfemn->scheme_num_points << std::endl;
  fout << "quadrature_name = " << pmbp->pradfemn->scheme_name << std::endl;

  // save quadrature information
  std::ofstream fout1(pathdir + filenamepart + "_quadrature_points_weights" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->scheme_num_points; i++) {
    fout1 << pmbp->pradfemn->scheme_points(i, 0) << " " << pmbp->pradfemn->scheme_points(i, 1) << " " << pmbp->pradfemn->scheme_points(i, 2) << " "
          << pmbp->pradfemn->scheme_weights(i) << std::endl;
  }

  // save (l,m) information
  std::ofstream fout2(pathdir + filenamepart + "_l_m" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    fout2 << pmbp->pradfemn->angular_grid(i, 0) << " " << pmbp->pradfemn->angular_grid(i, 1) << std::endl;
  }

  // save mass matrix
  std::ofstream fout3(pathdir + filenamepart + "_mass_matrix" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout3 << pmbp->pradfemn->mass_matrix(i, j) << " ";
    }
    fout3 << std::endl;
  }

  // save stiffness-x matrix
  std::ofstream fout4(pathdir + filenamepart + "_stiffness_x" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout4 << pmbp->pradfemn->stiffness_matrix_x(i, j) << " ";
    }
    fout4 << std::endl;
  }

  // save stiffness-y matrix
  std::ofstream fout5(pathdir + filenamepart + "_stiffness_y" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout5 << pmbp->pradfemn->stiffness_matrix_y(i, j) << " ";
    }
    fout5 << std::endl;
  }

  // save stiffness-z matrix
  std::ofstream fout6(pathdir + filenamepart + "_stiffness_z" + ".txt");
  for (int i = 0; i < pmbp->pradfemn->num_points; i++) {
    for (int j = 0; j < pmbp->pradfemn->num_points; j++) {
      fout6 << pmbp->pradfemn->stiffness_matrix_z(i, j) << " ";
    }
    fout6 << std::endl;
  }

  return;
}