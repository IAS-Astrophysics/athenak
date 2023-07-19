//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_linalg.cpp
//  \brief implementation of the matrix inverse routines

#include <cmath>
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
} // namespace radiationfemn