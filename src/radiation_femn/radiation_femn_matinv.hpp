#ifndef ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_MATINV_HPP_
#define ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_MATINV_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_matinv.cpp
//  \brief implementation of routines for matrix inversion

#include <cmath>
#include "athena.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

// Perform LU decomposition on a square matrix
// Generates a row-wise permutated matrix lu_matrix containing the LU decomposed matrix and the permutation in pivot_indices
//
// Note: lu_matrix and square_matrix should be the same before using this function!
//
// square_matrix: [NxN] matrix of reals, input matrix
// lu_matrix:     [NxN] matrix of reals, the row-permuted LU decomposition matrix
// pivot_indices: [N-1] array of reals, the permutation information
template<typename T1, typename T3>
KOKKOS_INLINE_FUNCTION void LUDec(T1 square_matrix, T1 lu_matrix, T3 pivot_indices) {

  int num_rows = square_matrix.extent(0);

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
//
// Note: b_array and x_array should be the same before using this function!
//
// lu_matrix:     [NxN] matrix, the LU decomposed matrix
// pivot_indices: [N] array, the pivot information
// b_array:       [N] array, the array b
// x_array:       [N] array, the solution x
template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION void LUSolve(const T1 lu_matrix, const T3 pivot_indices, const T2 b_array, T2 x_array) {

  int num_rows = b_array.extent(0);
  double swap_value;

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
// lu_matrix:         [NxN] a copy of A_matrix
// b_array:           [N] the rhs of linear equations
// x_array:           [N] a copy of b_array
// pivots:            [N-1] an integer array
template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION void LUInv(TeamMember_t member, T1 A_matrix, T1 A_matrix_inverse, T1 lu_matrix, T2 x_array, T2 b_array, T3 pivots) {

  int n = A_matrix.extent(0);

  radiationfemn::LUDec<T1, T3>(A_matrix, lu_matrix, pivots);

  par_for_inner(member, 0, n - 1, [&](const int i) {
    //Kokkos::deep_copy(b_array, 0.);
    //Kokkos::deep_copy(x_array, 0.);
    b_array(i) = 1.;
    x_array(i) = 1;

    radiationfemn::LUSolve<T1, T2, T3>(lu_matrix, pivots, b_array, x_array);
    for (int j = 0; j < n; j++) {
      A_matrix_inverse(j, i) = x_array(j);
    }
  });

}

template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION void LUInv(T1 A_matrix, T1 A_matrix_inverse, T1 lu_matrix, T2 x_array, T2 b_array, T3 pivots) {

  int n = A_matrix.extent(0);

  radiationfemn::LUDec<T1, T3>(A_matrix, lu_matrix, pivots);

  for (int i = 0; i <= n - 1; i++) {
    b_array(i) = 1.;
    x_array(i) = 1;

    radiationfemn::LUSolve<T1, T2, T3>(lu_matrix, pivots, b_array, x_array);
    for (int j = 0; j < n; j++) {
      A_matrix_inverse(j, i) = x_array(j);
    }

    b_array(i) = 0.;
    x_array(i) = 0.;
  }

}

}

#endif //ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_MATINV_HPP_
