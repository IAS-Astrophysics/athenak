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

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

void ProblemGenerator::RadiationFEMNLinalgtest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The radiation FEM_N grid test can only be run with radiation-femn, but no "
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Entered Linear algebra test problem. " << std::endl;

  DvceArray2D<Real> matrix;
  DvceArray2D<Real> matrix_answer;
  DvceArray2D<Real> lm_matrix;
  DvceArray1D<int> pivots;

  // Test 1: Check the matrix from Golub & Van Loan example 3.4.1

  Kokkos::realloc(matrix, 3, 3);
  Kokkos::realloc(matrix_answer, 3, 3);
  Kokkos::realloc(lm_matrix, 3, 3);
  Kokkos::realloc(pivots, 2);

  matrix(0, 0) = 3.;
  matrix(0, 1) = 17.;
  matrix(0, 2) = 10.;
  matrix(1, 0) = 2.;
  matrix(1, 1) = 4.;
  matrix(1, 2) = -2.;
  matrix(2, 0) = 6.;
  matrix(2, 1) = 18.;
  matrix(2, 2) = -12.;

  matrix_answer(0, 0) = 6.;
  matrix_answer(0, 1) = 18.;
  matrix_answer(0, 2) = -12.;
  matrix_answer(1, 0) = 1. / 3.;
  matrix_answer(1, 1) = 8.;
  matrix_answer(1, 2) = 16.;
  matrix_answer(2, 0) = 1. / 2.;
  matrix_answer(2, 1) = -1. / 4.;
  matrix_answer(2, 2) = 6.;

  std::cout << std::endl;
  std::cout << "Test 1: Check LU decomposed matrix" << std::endl;
  std::cout << std::endl;
  std::cout << "Matrix information:" << std::endl;
  std::cout << "Number of rows: " << matrix.extent(0) << std::endl;
  std::cout << "Number of cols: " << matrix.extent(1) << std::endl;
  std::cout << std::endl;

  std::cout << "Matrix:" << std::endl;
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      std::cout << matrix(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }

  radiationfemn::LUDecomposition(matrix, lm_matrix, pivots);

  std::cout << std::endl;
  std::cout << "LU Matrix (row swapped):" << std::endl;
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      std::cout << lm_matrix(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Correct LU Matrix:" << std::endl;
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      std::cout << matrix_answer(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }

  double error = -42.;
  std::cout << std::endl;
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      if (fabs(lm_matrix(i, j) - matrix_answer(i, j)) > error) {
        error = fabs(lm_matrix(i, j) - matrix_answer(i, j));
      }
    }
  }
  std::cout << "Maximum error: " << error << std::endl;

  std::cout << std::endl;
  std::cout << "Pivot information" << std::endl;
  for (int i = 0; i < pivots.extent(0); i++) {
    std::cout << pivots(i) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Test 2: Solve A x = b using LU decomposition" << std::endl;
  DvceArray1D<Real> b_array;
  DvceArray1D<Real> solution;
  DvceArray1D<Real> solution_correct;
  Kokkos::realloc(b_array, 3);
  Kokkos::realloc(solution, 3);
  Kokkos::realloc(solution_correct, 3);

  b_array(0) = 4.;
  b_array(1) = 9.;
  b_array(2) = 6.;

  solution_correct(0) = 247./24.;
  solution_correct(1) = -55./24.;
  solution_correct(2) = 29./24.;

  radiationfemn::LUSolve(lm_matrix, pivots, b_array, solution);

  std::cout << "Solution \t Correct solution \t |Difference|: " << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << solution(i) << " \t " << solution_correct(i) << " \t\t " << fabs(solution(i)-solution_correct(i)) << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Test 3: Compute inverse of the matrix" << std::endl;
  DvceArray2D<Real> matrix_inverse;
  DvceArray2D<Real> matrix_inverse_answer;

  Kokkos::realloc(matrix_inverse, 3, 3);
  Kokkos::realloc(matrix_inverse_answer, 3, 3);

  matrix_inverse_answer(0,0) = -6./144.;
  matrix_inverse_answer(0,1) = 192./144.;
  matrix_inverse_answer(0,2) = -37./144.;
  matrix_inverse_answer(1,0) = 6./144.;
  matrix_inverse_answer(1,1) = -48./144.;
  matrix_inverse_answer(1,2) = 13./144.;
  matrix_inverse_answer(2,0) = 6./144.;
  matrix_inverse_answer(2,1) = 24./144.;
  matrix_inverse_answer(2,2) = -11./144.;

  radiationfemn::LUInverse(matrix, matrix_inverse);

  error = -42.;
  std::cout << "Matrix inverse:" << std::endl;
  for (int i = 0; i < matrix_inverse.extent(0); i++) {
    for (int j = 0; j < matrix_inverse.extent(1); j++) {
      std::cout << matrix_inverse(i, j) << " " << std::flush;
      if(fabs(matrix_inverse(i,j)-matrix_inverse_answer(i,j)) > error) {
        error = fabs(matrix_inverse(i,j)-matrix_inverse_answer(i,j));
      }
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Correct inverse:" << std::endl;
  for (int i = 0; i < matrix_inverse_answer.extent(0); i++) {
    for (int j = 0; j < matrix_inverse_answer.extent(1); j++) {
      std::cout << matrix_inverse_answer(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Maximum error in computing inverse: " << error << std::endl;

  return;
}