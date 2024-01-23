//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_gridtest.cpp
//! \brief tests the linear algebra routines for radiation FEM_N

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
#include "radiation_femn/radiation_femn_linalg.hpp"
#include "radiation_femn/radiation_femn_matinv.hpp"

void ProblemGenerator::RadiationFEMNLinalgtest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The radiation FEM_N grid test can only be run with radiation-femn, but no "
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
	/*
  std::cout << "Testing linear algebra routines for radiation FEMN:" << std::endl;
  std::cout << std::endl;

  HostArray2D <Real> matrix;
  HostArray2D <Real> matrix_answer;
  HostArray2D <Real> lu_matrix;
  HostArray1D<int> pivots;
  HostArray1D<Real> x_array;
  HostArray1D<Real> b_array;

  Kokkos::realloc(matrix, 3, 3);
  Kokkos::realloc(matrix_answer, 3, 3);
  Kokkos::realloc(lu_matrix, 3, 3);
  Kokkos::realloc(pivots, 2);
  Kokkos::realloc(x_array, 3);
  Kokkos::realloc(b_array, 3);

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

  Kokkos::deep_copy(lu_matrix, matrix);


  // [1] Check the LU decomposition of matrix from Golub & Van Loan example 3.4.1
  std::cout << "Test 1: Check LU decomposition of matrix" << std::endl;
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
  std::cout << std::endl;

  radiationfemn::LUDec<HostArray2D<Real>, HostArray1D<int>>(matrix, lu_matrix, pivots);

  std::cout << "LUDec result:" << std::endl;
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      std::cout << lu_matrix(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Answer:" << std::endl;
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      std::cout << matrix_answer(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  double error = -42.;
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      if (fabs(lu_matrix(i, j) - matrix_answer(i, j)) > error) {
        error = fabs(lu_matrix(i, j) - matrix_answer(i, j));
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

  // [2] Solve a linear system of equations with the LU decomposed matrix
  std::cout << "Test 2: Solve A x = b using the LU matrix" << std::endl;
  std::cout << std::endl;
  DvceArray1D <Real> solution;
  DvceArray1D <Real> solution_correct;
  Kokkos::realloc(solution, 3);
  Kokkos::realloc(solution_correct, 3);

  b_array(0) = 4.;
  b_array(1) = 9.;
  b_array(2) = 6.;
  Kokkos::deep_copy(solution, b_array);

  solution_correct(0) = 247. / 24.;
  solution_correct(1) = -55. / 24.;
  solution_correct(2) = 29. / 24.;

  radiationfemn::LUSolve<HostArray2D<Real>, HostArray1D<Real>, HostArray1D<int>>(lu_matrix, pivots, b_array, solution);

  std::cout << "b:" << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << b_array(i) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Solution \t Correct solution \t |Difference|: " << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << solution(i) << " \t " << solution_correct(i) << " \t\t " << fabs(solution(i) - solution_correct(i)) << std::endl;
  }
  std::cout << std::endl;

  // test matrix inverse routine
  std::cout << "Test 3: Compute inverse of the matrix" << std::endl;
  std::cout << std::endl;
  HostArray2D <Real> matrix_inverse;
  HostArray2D <Real> matrix_inverse_answer;

  Kokkos::realloc(matrix_inverse, 3, 3);
  Kokkos::realloc(matrix_inverse_answer, 3, 3);
  Kokkos::deep_copy(x_array, 0.);
  Kokkos::deep_copy(b_array, 0.);
  Kokkos::deep_copy(pivots, 0);
  Kokkos::deep_copy(lu_matrix, matrix);

  matrix_inverse_answer(0, 0) = -6. / 144.;
  matrix_inverse_answer(0, 1) = 192. / 144.;
  matrix_inverse_answer(0, 2) = -37. / 144.;
  matrix_inverse_answer(1, 0) = 6. / 144.;
  matrix_inverse_answer(1, 1) = -48. / 144.;
  matrix_inverse_answer(1, 2) = 13. / 144.;
  matrix_inverse_answer(2, 0) = 6. / 144.;
  matrix_inverse_answer(2, 1) = 24. / 144.;
  matrix_inverse_answer(2, 2) = -11. / 144.;


  radiationfemn::LUInv<HostArray2D<Real>, HostArray1D<Real>, HostArray1D<int>>(matrix, matrix_inverse, lu_matrix, x_array, b_array, pivots);
  radiationfemn::LUInverse(matrix, matrix_inverse);

  error = -42.;
  std::cout << "Matrix inverse:" << std::endl;
  for (int i = 0; i < matrix_inverse.extent(0); i++) {
    for (int j = 0; j < matrix_inverse.extent(1); j++) {
      std::cout << matrix_inverse(i, j) << " " << std::flush;
      if (fabs(matrix_inverse(i, j) - matrix_inverse_answer(i, j)) > error) {
        error = fabs(matrix_inverse(i, j) - matrix_inverse_answer(i, j));
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

  // test matrix multiplication
  std::cout << std::endl;
  std::cout << "Test 4: Compute product of two square matrices" << std::endl;
  std::cout << std::endl;

  HostArray2D <Real> mat_a;
  HostArray2D <Real> mat_b;
  HostArray2D <Real> mat_ab;
  HostArray2D <Real> mat_ab_correct;

  Kokkos::realloc(mat_a, 3, 3);
  Kokkos::realloc(mat_b, 3, 3);
  Kokkos::realloc(mat_ab, 3, 3);
  Kokkos::realloc(mat_ab_correct, 3, 3);

  mat_ab_correct(0, 0) = 23;
  mat_ab_correct(0, 1) = 13;
  mat_ab_correct(0, 2) = 14;
  mat_ab_correct(1, 0) = 21;
  mat_ab_correct(1, 1) = 21;
  mat_ab_correct(1, 2) = 33;
  mat_ab_correct(2, 0) = 9;
  mat_ab_correct(2, 1) = 6;
  mat_ab_correct(2, 2) = 4;

  mat_a(0, 0) = 2;
  mat_a(0, 1) = 7;
  mat_a(0, 2) = 3;
  mat_a(1, 0) = 1;
  mat_a(1, 1) = 5;
  mat_a(1, 2) = 8;
  mat_a(2, 0) = 0;
  mat_a(2, 1) = 4;
  mat_a(2, 2) = 1;

  mat_b(0, 0) = 3;
  mat_b(0, 1) = 0;
  mat_b(0, 2) = 1;
  mat_b(1, 0) = 2;
  mat_b(1, 1) = 1;
  mat_b(1, 2) = 0;
  mat_b(2, 0) = 1;
  mat_b(2, 1) = 2;
  mat_b(2, 2) = 4;

  std::cout << "Matrix A:" << std::endl;
  for (int i = 0; i < mat_a.extent(0); i++) {
    for (int j = 0; j < mat_a.extent(1); j++) {
      std::cout << mat_a(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Matrix B:" << std::endl;
  for (int i = 0; i < mat_b.extent(0); i++) {
    for (int j = 0; j < mat_b.extent(1); j++) {
      std::cout << mat_b(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  radiationfemn::MatMultiplyHost(mat_a, mat_b, mat_ab);

  error = -42.;
  std::cout << "Computed product:" << std::endl;
  for (int i = 0; i < mat_ab.extent(0); i++) {
    for (int j = 0; j < mat_ab.extent(1); j++) {
      std::cout << mat_ab(i, j) << " " << std::flush;
      if (fabs(mat_ab(i, j) - mat_ab_correct(i, j)) > error) {
        error = fabs(mat_ab(i, j) - mat_ab_correct(i, j));
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Answer:" << std::endl;
  for (int i = 0; i < mat_ab.extent(0); i++) {
    for (int j = 0; j < mat_ab.extent(1); j++) {
      std::cout << mat_ab_correct(i, j) << " " << std::flush;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Maximum error in computing product: " << error << std::endl;
  std::cout << std::endl;

  // test eigenvalue eigenvector routine
  std::cout << "Test 5: Compute eigenvalues and eigenvectors" << std::endl;

  std::vector<std::vector<double>> mat = {{-1.0, 1.0, -1.0, 1.0},
                                          {-8.0, 4.0, -2.0, 1.0},
                                          {27.0, 9.0, 3.0, 1.0},
                                          {64.0, 16.0, 4.0, 1.0 }};

  std::vector<std::complex<double>> eigval;
  std::vector<std::vector<std::complex<double>>> eigvec;
  radiationfemn::MatEig(mat, eigval, eigvec);

  std::cout << std::endl;
  std::cout << "Matrix (for which we compute eigenvalues/eigenvectors):" << std::endl;
  for (int i = 0; i < mat.size(); i++) {
    for (int j = 0; j < mat[0].size(); j++) {
      std::cout << mat[i][j] << " " << std::flush;
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Eigenvalues: " << std::endl;
  for (int i = 0; i < eigval.size(); i++){
    std::cout << eigval[i] << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Eigenvectors (column wise):" << std::endl;
  for (int i = 0; i < eigvec.size(); i++) {
    for (int j = 0; j < eigvec[0].size(); j++) {
      std::cout << eigvec[j][i] << " " << std::flush;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // test zero speed more corrections
  std::cout << "Test 6: Compute the zero speed mode corrections" << std::endl;
  std::cout << std::endl;

  HostArray2D<Real> zerosp_matrix;
  HostArray2D<Real> zerosp_matrix_corrected;
  HostArray2D<Real> zerosp_answer;

  Kokkos::realloc(zerosp_matrix, 3, 3);
  Kokkos::realloc(zerosp_matrix_corrected, 3, 3);
  Kokkos::realloc(zerosp_answer, 3, 3);
  double v = 1./ sqrt(3);

  zerosp_matrix(0,0) = 2.1;
  zerosp_matrix(0,1) = 3.;
  zerosp_matrix(0,2) = 4.5;
  zerosp_matrix(1,0) = 5.3;
  zerosp_matrix(1,1) = 6.9;
  zerosp_matrix(1,2) = 7.1;
  zerosp_matrix(2,0) = 1.1;
  zerosp_matrix(2,1) = 3.4;
  zerosp_matrix(2,2) = 5.6;

  zerosp_answer(0,0) = 2.76788416;
  zerosp_answer(0,1) = 2.99146524;
  zerosp_answer(0,2) = 3.89624893;
  zerosp_answer(1,0) = 3.77051678;
  zerosp_answer(1,1) = 7.13741257;
  zerosp_answer(1,2) = 8.06392804;
  zerosp_answer(2,0) = 1.96201983;
  zerosp_answer(2,1) = 3.22170985;
  zerosp_answer(2,2) = 5.14221535;

  radiationfemn::ZeroSpeedCorrection(zerosp_matrix, zerosp_matrix_corrected, v);

  std::cout << "Matrix for zero speed correction:" << std::endl;
  for (int i = 0; i < zerosp_matrix.extent(0); i++) {
    for (int j = 0; j < zerosp_matrix.extent(1); j++) {
      std::cout << zerosp_matrix(i,j) << " " << std::flush;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Answer:" << std::endl;
  for (int i = 0; i < zerosp_answer.extent(0); i++) {
    for (int j = 0; j < zerosp_answer.extent(1); j++) {
      std::cout << zerosp_answer(i,j) << " " << std::flush;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  error = -42.;
  std::cout << "Computed zero speed mode corrected matrix:" << std::endl;
  for (int i = 0; i < zerosp_matrix.extent(0); i++) {
    for (int j = 0; j < zerosp_matrix.extent(0); j++) {
      std::cout << zerosp_matrix_corrected(i,j) << " " << std::flush;
      error = fabs(zerosp_matrix_corrected(i, j) - zerosp_answer(i, j));
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Maximum error in computing zero speed corrected matrices: " << error << std::endl;
	*/
  return;
}
