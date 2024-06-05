#ifndef ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_MATINV_HPP_
#define ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_MATINV_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_matinv.cpp
//  \brief implementation of routines for matrix inversion

#include <math.h>

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

  Real largest_pivot;
  Real swap_value;

  // begin construction of the row shifted square matrix
  for (int k = 0; k < num_rows - 1; k++) {

    // get the best pivot
    largest_pivot = 0.;
    for (int i = k; i < num_rows; i++) {
      if (Kokkos::fabs(lu_matrix(i, k)) > largest_pivot) {
        largest_pivot = Kokkos::fabs(lu_matrix(i, k));
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
  Real swap_value;

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
// b_array:           [N] must be populated with 0
// x_array:           [N] must be populated with 0
// pivots:            [N-1] an integer array
template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION void LUInv(TeamMember_t member, T1 A_matrix, T1 A_matrix_inverse, T1 lu_matrix, T2 x_array, T2 b_array, T3 pivots) {

  int n = A_matrix.extent(0);

  radiationfemn::LUDec<T1, T3>(A_matrix, lu_matrix, pivots);

  par_for_inner(member, 0, n - 1, [&](const int i) {
    for (int j = 0; j < n; j++) {
      b_array(j) = 0.;
      x_array(j) = 0.;
    }
    b_array(i) = 1.;
    x_array(i) = 1;

    radiationfemn::LUSolve<T1, T2, T3>(lu_matrix, pivots, b_array, x_array);
    for (int j = 0; j < n; j++) {
      A_matrix_inverse(j, i) = x_array(j);
    }

  });

}

template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION void LUSolveAxb(TeamMember_t member, T1 A_matrix, T1 lu_matrix, T2 x_array, T2 b_array, T3 pivots) {

  int n = A_matrix.extent(0);

  radiationfemn::LUDec<T1, T3>(A_matrix, lu_matrix, pivots);
  radiationfemn::LUSolve<T1, T2, T3>(lu_matrix, pivots, b_array, x_array);

}

template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION void LUInv(T1 A_matrix, T1 A_matrix_inverse, T1 lu_matrix, T2 x_array, T2 b_array, T3 pivots) {

  int n = A_matrix.extent(0);

  radiationfemn::LUDec<T1, T3>(A_matrix, lu_matrix, pivots);

  for (int i = 0; i <= n - 1; i++) {
    for (int j = 0; j < n; j++) {
      b_array(j) = 0.;
      x_array(j) = 0.;
    }
    b_array(i) = 1.;
    x_array(i) = 1;

    radiationfemn::LUSolve<T1, T2, T3>(lu_matrix, pivots, b_array, x_array);
    for (int j = 0; j < n; j++) {
      A_matrix_inverse(j, i) = x_array(j);
    }

  }

}

}

template<typename T2>
KOKKOS_INLINE_FUNCTION Real dot(TeamMember_t member, T2 a, T2 b) {
  Real result = 0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, a.extent(0)), [&](const int j, Real &partial_sum) {
    partial_sum += a(j) * b(j);
  }, result);
  member.team_barrier();
  return result;
}

template<typename T1, typename T2>
KOKKOS_INLINE_FUNCTION Real dot(int i, T1 a, T2 b) {
  Real result = 0.;
  for (int j = 0; j < a.extent(0); j++) {
    result += a(i, j) * b(j);
  }
  return result;
}

template<typename T1, typename T2>
KOKKOS_INLINE_FUNCTION Real dot(TeamMember_t member, int i, T1 a, T2 b) {
  Real result = 0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, a.extent(0)), [&](const int j, Real &partial_sum) {
    partial_sum += a(i, j) * b(j);
  }, result);
  member.team_barrier();
  return result;
}

template<typename T1, typename T2>
KOKKOS_INLINE_FUNCTION void dot(TeamMember_t member, T1 a, T2 b, T2 result) {
  for (int i = 0; i < a.extent(0); i++) {
    result(i) = 0;
    for (int j = 0; j < a.extent(1); j++) {
      result(i) += a(i, j) * b(j);
    }
  }
  member.team_barrier();
}

template<typename T1, typename T2>
KOKKOS_INLINE_FUNCTION void uBiCGSTAB(T1 A_test, T2 b_test, T2 x0, T2 rhat0, T2 r0, T2 rho0, T2 p0, T2 v, T2 h, T2 s, T2 t, Real tol = 1e-6, int tot_iter = 200) {

  int niter = 0;
  int num_rows = A_test.extent(0);

  for (int i = 0; i < num_rows; i++) {
    x0(i) = 0;
    rhat0(i) = 0;
    r0(i) = b_test(i) - dot<T1, T2>(i, A_test, x0);
  }

  rho0 = dot<T2>(rhat0, r0);
  Kokkos::deep_copy(p0, r0);

  for (int i = 0; i < tot_iter; i++) {
    niter++;

    dot<T1, T2>(A_test, p0, v);
    auto alpha = rho0 / dot<T2>(rhat0, v);

    for (int j = 0; j < num_rows; j++) {
      h(j) = x0(j) + alpha * p0(j);
      s(j) = r0(j) - alpha * v(j);
    }

    if (dot<T2>(s, s) < tol) {
      Kokkos::deep_copy(x0, h);
      break;
    }

    dot<T1, T2>(A_test, s, t);
    auto omega = dot<T2, T2>(t, s) / dot<T2, T2>(t, t);

    for (int j = 0; j < num_rows; j++) {
      x0(j) = h(j) + omega * s(j);
      r0(j) = s(j) - omega * t(j);
    }

    if (dot<T2, T2>(r0, r0) < tol) {
      break;
    }

    auto rho1 = dot<T2, T2>(rhat0, r0);
    auto beta = (rho1 / rho0) * (alpha / omega);
    rho0 = rho1;

    for (int j = 0; j < num_rows; j++) {
      p0(j) = r0(j) + beta * (p0(j) - omega * v(j));
    }
  }
}

#endif //ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_MATINV_HPP_
