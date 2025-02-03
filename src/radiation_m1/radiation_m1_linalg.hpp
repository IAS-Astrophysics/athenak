#ifndef RADIATION_M1_LINALG_HPP
#define RADIATION_M1_LINALG_HPP

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_linalg.hpp
//  \brief Linear algebra routines for M1

#include "radiation_m1_macro.hpp"

#include <athena.hpp>

namespace radiationm1 {

KOKKOS_INLINE_FUNCTION
Real l2_norm(const Real (&V)[M1_MULTIROOTS_DIM]) {
  Real result = 0;
  for (double i : V) {
    result += i * i;
  }
  return Kokkos::sqrt(result);
}

KOKKOS_INLINE_FUNCTION
Real dot(const Real (&U)[M1_MULTIROOTS_DIM],
         const Real (&V)[M1_MULTIROOTS_DIM]) {
  Real result = 0;
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    result += U[i] * V[i];
  }
  return result;
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::compute_diag
//  \brief computes the columnwise L2 norm of a matrix J abd store in diag
KOKKOS_INLINE_FUNCTION
void qr_factorize(const Real (&A)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                  Real (&Q)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                  Real (&R)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM]) {
  Real V[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM]{};

  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    for (int j = 0; j < M1_MULTIROOTS_DIM; j++) {
      Q[i][j] = 0;
      R[i][j] = 0;
      V[i][j] = A[i][j];
    }
  }

  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    Real Vi[M1_MULTIROOTS_DIM]{};
    Real Qi[M1_MULTIROOTS_DIM]{};

    for (int k = 0; k < M1_MULTIROOTS_DIM; k++) {
      Vi[k] = V[k][i];
    }
    R[i][i] = l2_norm(Vi);
    for (int k = 0; k < M1_MULTIROOTS_DIM; k++) {
      Q[k][i] = Vi[k] / R[i][i];
      Qi[k] = Q[k][i];
    }

    for (int j = i + 1; j < M1_MULTIROOTS_DIM; j++) {
      Real Vj[M1_MULTIROOTS_DIM]{};
      for (int k = 0; k < M1_MULTIROOTS_DIM; k++) {
        Vj[k] = V[k][j];
      }
      R[i][j] = dot(Qi, Vj);
      for (int k = 0; k < M1_MULTIROOTS_DIM; k++) {
        V[k][j] = V[k][j] - R[i][j] * Qi[k];
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void qr_update(Real (&Q)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
               Real (&R)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
               Real (&W)[M1_MULTIROOTS_DIM], Real (&V)[M1_MULTIROOTS_DIM]) {}

KOKKOS_INLINE_FUNCTION
void qr_R_solve(const Real (&r)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                const Real (&qtf)[M1_MULTIROOTS_DIM],
                Real (&p)[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    p[i] = qtf[i];
  }
  p[M1_MULTIROOTS_DIM - 1] = qtf[M1_MULTIROOTS_DIM - 1] /
                             r[M1_MULTIROOTS_DIM - 1][M1_MULTIROOTS_DIM - 1];
  for (int k = M1_MULTIROOTS_DIM - 2; k >= 0; k--) {
    for (int j = k + 1; j < M1_MULTIROOTS_DIM; j++) {
      p[k] = p[k] - r[k][j] * p[j];
    }
    p[k] = p[k] / r[k][k];
  }
}
} // namespace radiationm1
#endif // RADIATION_M1_LINALG_HPP
