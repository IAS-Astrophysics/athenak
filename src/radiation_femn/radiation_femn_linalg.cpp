//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_linalg.cpp
//  \brief implementation of the matrix inverse routines

#include <cmath>
#include <complex>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include "athena.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

// Perform LU decomposition on a square matrix
// Generates a row-wise permutated matrix lu_matrix containing the LU decomposed matrix and the permutation in pivot_indices
// square_matrix: [NxN] matrix of reals, input matrix
// lu_matrix:     [NxN] matrix of reals, the row-permuted LU decomposition matrix
// pivot_indices: [N-1] array of reals, the permutation information
void LUDecomposition(DvceArray2D<Real> square_matrix, DvceArray2D<Real> lu_matrix, DvceArray1D<int> pivot_indices) {

  int num_rows = square_matrix.extent(0);

  Kokkos::realloc(pivot_indices, num_rows - 1);
  Kokkos::deep_copy(lu_matrix, square_matrix);

  double largest_pivot;
  double swap_value;

  // begin construction of the row shifted square matrix
  for (int k = 0; k < num_rows - 1; k++) {

    // get the best pivot
    largest_pivot = 0.;
    for (int i = k; i < num_rows; i++) {
      if (fabs(lu_matrix(i, k)) > largest_pivot) {
        largest_pivot = fabs(lu_matrix(i, k));
        pivot_indices(k) = i;
      }
    }

    // if new pivot is found, swap rows
    if (pivot_indices[k] != k) {
      for (int j = k; j < num_rows; j++) {
        swap_value = lu_matrix(pivot_indices[k], j);
        lu_matrix(pivot_indices[k], j) = lu_matrix(k, j);
        lu_matrix(k, j) = swap_value;
      }
    }

    for (int i = k + 1; i < num_rows; i++) {
      lu_matrix(i, k) = lu_matrix(i, k) / lu_matrix(k, k);
      for (int j = k + 1; j < num_rows; j++) {
        lu_matrix(i, j) = lu_matrix(i, j) - lu_matrix(i, k) * lu_matrix(k, j);
      }
    }
  }
}

// Solve a set of equations A[N,N] x[N] = b[N] using an LU decomposed matrix and pivot information
// lu_matrix:     [NxN] matrix, the LU decomposed matrix
// pivot_indices: [N] array, the pivot information
// b_array:       [N] array, the array b
// x_array:       [N] array, the solution x
void LUSolve(const DvceArray2D<Real> lu_matrix, const DvceArray1D<int> pivot_indices, const DvceArray1D<Real> b_array, DvceArray1D<Real> x_array) {

  int num_rows = b_array.extent(0);
  int index;
  double swap_value;
  double temp_value;

  Kokkos::realloc(x_array, num_rows);
  Kokkos::deep_copy(x_array, b_array);

  // Forward substitution to find solution to L y = b
  for (int k = 0; k < num_rows - 1; k++) {
    swap_value = x_array(pivot_indices(k));
    x_array(pivot_indices(k)) = x_array(k);
    x_array(k) = swap_value;
    for (int j = k + 1; j < num_rows; j++) {
      x_array(j) = x_array(j) - lu_matrix(j, k) * x_array(k);
    }
  }

  // Back substitution to find solution to U x = y
  x_array(num_rows - 1) = x_array(num_rows - 1) / lu_matrix(num_rows - 1, num_rows - 1);
  for (int k = num_rows - 2; k >= 0; k--) {
    for (int j = k + 1; j < num_rows; j++) {
      x_array(k) = x_array(k) - lu_matrix(k, j) * x_array(j);
    }
    x_array(k) = x_array(k) / lu_matrix(k, k);
  }

}

// Compute the inverse of a square matrix
// A_matrix:          [NxN] a square matrix
// A_matrix_inverse:  [NxN] the inverse
void LUInverse(DvceArray2D<Real> A_matrix, DvceArray2D<Real> A_matrix_inverse) {

  int n = A_matrix.extent(0);

  Kokkos::realloc(A_matrix_inverse, n, n);
  Kokkos::deep_copy(A_matrix_inverse, 0.);

  DvceArray1D<Real> x_array;
  DvceArray1D<Real> b_array;
  DvceArray2D<Real> lu_matrix;
  DvceArray1D<int> pivots;

  Kokkos::realloc(x_array, n);
  Kokkos::realloc(b_array, n);
  Kokkos::realloc(lu_matrix, n, n);
  Kokkos::realloc(pivots, n - 1);

  radiationfemn::LUDecomposition(A_matrix, lu_matrix, pivots);

  for (int i = 0; i < n; i++) {
    Kokkos::deep_copy(b_array, 0.);
    b_array(i) = 1.;
    radiationfemn::LUSolve(lu_matrix, pivots, b_array, x_array);
    for (int j = 0; j < n; j++) {
      A_matrix_inverse(j, i) = x_array(j);
    }
  }

}

// Multiply two matrices
// A_matrix:          [NxN] a square matrix
// B_matrix:          [NxN] a square matrix
// result:            [NxN] the product
void MatMultiply(HostArray2D<Real> A_matrix, HostArray2D<Real> B_matrix, HostArray2D<Real> result) {

  int N = A_matrix.extent(0);

  Kokkos::deep_copy(result, 0.);
  par_for("radiation_femn_matrix_multiply", DevExeSpace(), 0, N - 1, 0, N - 1, 0, N - 1,
          KOKKOS_LAMBDA(const int i, const int j, const int k) {

            result(i, j) += A_matrix(i, k) * B_matrix(k, j);
          });
}

// Multiply two complex matrices
// A_matrix:          [NxN] a square matrix
// B_matrix:          [NxN] a square matrix
// result:            [NxN] the product
void MatMultiplyComplex(std::vector<std::vector<std::complex<double>>> &A_matrix,
                        std::vector<std::vector<std::complex<double>>> &B_matrix,
                        std::vector<std::vector<std::complex<double>>> &result) {

  int N = A_matrix.size();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        result[i][j] = 0.;
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        result[i][j] += A_matrix[i][k] * B_matrix[k][j];
      }
    }
  }

}

// Support for mass lumping
// A_matrix:          [NxN] a square matrix
void MatLumping(DvceArray2D<Real> A_matrix) {

  int N = A_matrix.extent(0);

  DvceArray2D<Real> result;
  Kokkos::realloc(result, N, N);
  Kokkos::deep_copy(result, 0.);
  par_for("radiation_femn_matrix_lumping", DevExeSpace(), 0, N - 1, 0, N - 1,
          KOKKOS_LAMBDA(const int i, const int j) {
            result(i, i) += A_matrix(i, j);
          });
  Kokkos::deep_copy(A_matrix, result);
}

// Compute eigenvalues and eigenvectors of a real square matrix (GSL)
// matrix:        [NxN] square matrix
// eigval:        [N] complex array of eigenvalues
// eigvec:        [NxN] complex array of eigenvectors
void MatEig(std::vector<std::vector<double>> &matrix, std::vector<std::complex<double>> &eigval, std::vector<std::vector<std::complex<double>>> &eigvec) {

  int N = matrix.size();
  double matrix_flattened[N * N];

  int idx = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrix_flattened[idx] = matrix[i][j];
      idx++;
    }
  }

  std::cout << std::endl;

  gsl_matrix_view matview = gsl_matrix_view_array(matrix_flattened, N, N);
  gsl_vector_complex *eigval_gsl = gsl_vector_complex_alloc(N);
  gsl_matrix_complex *eigvec_gsl = gsl_matrix_complex_alloc(N, N);
  gsl_eigen_nonsymmv_workspace *nonsymw = gsl_eigen_nonsymmv_alloc(N);

  gsl_eigen_nonsymmv(&matview.matrix, eigval_gsl, eigvec_gsl, nonsymw);
  gsl_eigen_nonsymmv_sort(eigval_gsl, eigvec_gsl, GSL_EIGEN_SORT_ABS_DESC);
  gsl_eigen_nonsymmv_free(nonsymw);

  for (int i = 0; i < N; i++) {
    gsl_complex eval_i = gsl_vector_complex_get(eigval_gsl, i);
    gsl_vector_complex_view evec_i = gsl_matrix_complex_column(eigvec_gsl, i);

    eigval.push_back(std::complex<double>(GSL_REAL(eval_i), GSL_IMAG(eval_i)));

    std::vector<std::complex<double>> eigvecs_vec;
    for (int j = 0; j < N; j++) {
      gsl_complex evec_ij = gsl_vector_complex_get(&evec_i.vector, j);
      eigvecs_vec.push_back(std::complex<double>(GSL_REAL(evec_ij), GSL_IMAG(evec_ij)));
    }
    eigvec.push_back(eigvecs_vec);
  }

  gsl_vector_complex_free(eigval_gsl);
  gsl_matrix_complex_free(eigvec_gsl);
}

// Compute the correction for the zero speed modes
// matrix:            [NxN] a real square matrix
// matrix_corrected:  [NxN] the corrected square matrix
// v:                       the zero speed mode correction factor
void ZeroSpeedCorrection(HostArray2D<Real> matrix, HostArray2D<Real> matrix_corrected, double v) {

  // store the matrix in a vector<vector> for use with MatEig
  std::vector<Real> mat_row(matrix.extent(0));
  std::vector<std::vector<Real>> mat(matrix.extent(1), mat_row);
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      mat[i][j] = matrix(i, j);
    }
  }

  // compute eigenvalues and eigenvectors of matrix
  std::vector<std::complex<double>> eigval;
  std::vector<std::vector<std::complex<double>>> eigvec;
  MatEig(mat, eigval, eigvec);

  std::cout << std::endl;
  double eigvec_data[matrix.extent(0) * matrix.extent(1) * 2];
  int index = 0;
  for (int i = 0; i < matrix.extent(0); i++) {
    for (int j = 0; j < matrix.extent(1); j++) {
      std::cout << eigvec[j][i] << " " << std::flush;
      eigvec_data[index] = std::real(eigvec[j][i]);
      eigvec_data[index + 1] = std::imag(eigvec[j][i]);
      index = index + 2;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // print eigenvector data
  for (int i = 0; i < matrix.extent(0) * matrix.extent(1) * 2; i++) {
    std::cout << eigvec_data[i] << " " << std::flush;
  }
  std::cout << std::endl;

  // calculate the matrix of left eigenvectors
  int size = matrix.extent(0);
  gsl_matrix_complex_view m = gsl_matrix_complex_view_array(eigvec_data, size, size);
  gsl_matrix_complex *minv = gsl_matrix_complex_alloc(size, size);
  gsl_permutation *p = gsl_permutation_alloc(size);
  int s;

  gsl_linalg_complex_LU_decomp(&m.matrix, p, &s);
  gsl_linalg_complex_LU_invert(&m.matrix, p, minv);
  gsl_permutation_free(p);

  // compute the matrix of left eigenvectors (std::vector<std::vector<std::complex<double>>>)
  std::vector<std::complex<double>> lefteigvec_row(matrix.extent(0));
  std::vector<std::vector<std::complex<double>>> lefteigvec(matrix.extent(1), lefteigvec_row);
  std::cout << std::endl;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      gsl_complex minv_ij = gsl_matrix_complex_get(minv, i, j);
      lefteigvec[i][j] = std::complex<double>(GSL_REAL(minv_ij), GSL_IMAG(minv_ij));
      std::cout << GSL_REAL(minv_ij) << " + " << GSL_IMAG(minv_ij) << "j \t\t" << std::flush;
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      std::cout << lefteigvec[i][j] << " " << std::flush;
    }
    std::cout << std::endl;
  }

  // construct the matrix of right eigenvectors (std::vector<std::vector<std::complex<double>>>)
  std::vector<std::complex<double>> righteigvec_row(matrix.extent(0));
  std::vector<std::vector<std::complex<double>>> righteigvec(matrix.extent(1), righteigvec_row);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      righteigvec[i][j] = eigvec[j][i];
    }
  }

  std::cout << std::endl;
  std::cout << "Right eigenvectors: " << std::endl;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      std::cout << righteigvec[i][j] << " " << std::flush;
    }
    std::cout << std::endl;
  }

  // construct the zero speed mode corrected eigenvalue matrix
  std::vector<std::complex<double>> eigval_corrected_row(matrix.extent(0));
  std::vector<std::vector<std::complex<double>>> eigval_corrected(matrix.extent(1), eigval_corrected_row);
  std::cout << std::endl;
  std::cout << "Eigenvalues and corrections: " << std::endl;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      eigval_corrected[i][j] = std::complex<double>(0., 0.);
      if (i == j) {
        eigval_corrected[i][i] = std::max<double>(v, std::abs(eigval[i]));
        std::cout << "Eigenvalue : " << eigval[i] << " Corrected: " << eigval_corrected[i][i] << " | " << std::endl;
      }
    }
  }

  // reconstruct the corrected matrix
  std::vector<std::complex<double>> temp_reconstruction_row(matrix.extent(0));
  std::vector<std::vector<std::complex<double>>> temp_reconstruction(matrix.extent(1), temp_reconstruction_row);

  MatMultiplyComplex(righteigvec, eigval_corrected, temp_reconstruction);

  //std::cout << std::endl;
  //std::cout << "Product of right eigenvector matrix with modified eigenvalue matrix:" << std::endl;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      //std::cout << temp_reconstruction[i][j] << " " << std::flush;
    }
    //std::cout << std::endl;
  }

  std::vector<std::complex<double>> temp_reconstruction_row_2(matrix.extent(0));
  std::vector<std::vector<std::complex<double>>> temp_reconstruction_2(matrix.extent(1), temp_reconstruction_row_2);

  MatMultiplyComplex(temp_reconstruction, lefteigvec, temp_reconstruction_2);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      matrix_corrected(i,j) = std::real(temp_reconstruction_2[i][j]);
    }
  }

}
} // namespace radiationfemn