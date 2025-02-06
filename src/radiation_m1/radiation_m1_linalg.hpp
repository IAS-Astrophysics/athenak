#ifndef RADIATION_M1_LINALG_HPP
#define RADIATION_M1_LINALG_HPP

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_linalg.hpp
//  \brief Linear algebra routines for M1

#include "athena.hpp"
#include "radiation_m1_macro.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::norm_l2
//  \brief computes the L2 norm of a vector
KOKKOS_INLINE_FUNCTION
Real norm_l2(const Real (&V)[M1_MULTIROOTS_DIM]) {
  Real result = 0;
  for (double i : V) {
    result += i * i;
  }
  return Kokkos::sqrt(result);
}

KOKKOS_INLINE_FUNCTION
Real norm_l2(const Real (&V)[M1_MULTIROOTS_DIM - 1]) {
  Real result = 0;
  for (double i : V) {
    result += i * i;
  }
  return Kokkos::sqrt(result);
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::dot
//  \brief computes the dot product of two vectors
KOKKOS_INLINE_FUNCTION
Real dot(const Real (&U)[M1_MULTIROOTS_DIM],
         const Real (&V)[M1_MULTIROOTS_DIM]) {
  Real result = 0;
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    result += U[i] * V[i];
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
Real sign(Real a) { return (a >= 0) ? +1 : -1; }

KOKKOS_INLINE_FUNCTION
void dscal(Real a, Real (&v)[M1_MULTIROOTS_DIM - 1]) {
  for (double &i : v) {
    i = a * i;
  }
}

KOKKOS_INLINE_FUNCTION
Real householder_transform(Real (&v)[M1_MULTIROOTS_DIM]) {

  if (M1_MULTIROOTS_DIM == 1) {
    return 0.;
  } else {
    Real x[M1_MULTIROOTS_DIM - 1];
    for (int i = 0; i < M1_MULTIROOTS_DIM - 1; i++) {
      x[i] = v[i + 1];
    }

    double xnorm = norm_l2(x);

    if (xnorm == 0) {
      return 0.;
    }

    Real alpha = v[0];
    Real beta = -sign(alpha) * Kokkos::hypot(alpha, xnorm);
    Real tau = (beta - alpha) / beta;

    {
      Real s = (alpha - beta);

      if (Kokkos::fabs(s) > DBL_MIN) {
        dscal(1.0 / s, x);
        v[0] = beta;
      } else {
        dscal(DBL_EPSILON / s, x);
        dscal(1.0 / DBL_EPSILON, x);
        v[0] = beta;
      }
    }
    return tau;
  }
}

KOKKOS_INLINE_FUNCTION
void gsl_linalg_QR_decomp(Real (&A)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                          Real (&tau)[M1_MULTIROOTS_DIM]) {
    /*
    for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
      /* Compute the Householder transformation to reduce the j-th
         column of the matrix to a multiple of the j-th unit vector */
      /*
      gsl_vector_view c = gsl_matrix_subcolumn(A, i, i, M - i);
      double tau_i = gsl_linalg_householder_transform(&(c.vector));
      double *ptr = gsl_vector_ptr(&(c.vector), 0);

      gsl_vector_set(tau, i, tau_i);

      /* Apply the transformation to the remaining columns and
         update the norms */
    /*
      if (i + 1 < N) {
        gsl_matrix_view m = gsl_matrix_submatrix(A, i, i + 1, M - i, N - i - 1);
        gsl_vector_view work = gsl_vector_subvector(tau, i + 1, N - i - 1);
        double tmp = *ptr;

        *ptr = 1.0;
        gsl_linalg_householder_left(tau_i, &(c.vector), &(m.matrix),
                                    &(work.vector));
        *ptr = tmp;
      }
    }

    return GSL_SUCCESS; */

}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::qr_factorize
//  \brief computes the QR decomposition of a square matrix
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
    R[i][i] = norm_l2(Vi);
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

//----------------------------------------------------------------------------------------
//! \fn radiationm1::givens_rotation
//  \brief generates a Givens rotation v=(x,y) => (|v|,0)
KOKKOS_INLINE_FUNCTION
void givens_rotation(const Real &a, const Real &b, Real &c, Real &s) {
  if (b == 0) {
    c = 1;
    s = 0;
  } else if (Kokkos::fabs(b) > Kokkos::fabs(a)) {
    Real t = -a / b;
    Real s1 = 1.0 / sqrt(1 + t * t);
    s = s1;
    c = s1 * t;
  } else {
    Real t = -b / a;
    Real c1 = 1.0 / Kokkos::sqrt(1. + t * t);
    c = c1;
    s = c1 * t;
  }
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::givens_apply_gv
//  \brief applies rotation v' = G^T v
KOKKOS_INLINE_FUNCTION
void givens_apply_gv(Real (&v)[M1_MULTIROOTS_DIM], const int &i, const int &j,
                     const Real &c, const Real &s) {
  Real vi = v[i];
  Real vj = v[j];
  v[i] = c * vi - s * vj;
  v[j] = s * vi + c * vj;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::givens_apply_qr
//  \brief applies rotation Q' = Q G
KOKKOS_INLINE_FUNCTION
void givens_apply_qr(Real (&Q)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                     Real (&R)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                     const int &i, const int &j, const Real &c, const Real &s) {
  // Apply rotation Q' = Q G
  for (int k = 0; k < M1_MULTIROOTS_DIM; k++) {
    Real qki = Q[k][i];
    Real qkj = Q[k][j];
    Q[k][i] = qki * c - qkj * s;
    Q[k][j] = qki * s + qkj * c;
  }
  // Apply rotation R' = G^T R
  for (int k = Kokkos::min(i, j); k < M1_MULTIROOTS_DIM; k++) {
    Real rik = R[i][k];
    Real rjk = R[j][k];
    R[i][k] = c * rik - s * rjk;
    R[j][k] = s * rik + c * rjk;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::qr_update
//  \brief update a QR factorisation QR = B => B' = B + u v^T
//         Ref: (12.5.1) from Golub & Van Loan
KOKKOS_INLINE_FUNCTION
void qr_update(Real (&Q)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
               Real (&R)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
               Real (&W)[M1_MULTIROOTS_DIM], Real (&V)[M1_MULTIROOTS_DIM]) {

  // apply Given's rotation to W
  for (int k = M1_MULTIROOTS_DIM - 1; k > 0; k--) {
    Real c, s;
    Real Wk = W[k];
    Real Wkm1 = W[k - 1];

    givens_rotation(Wkm1, Wk, c, s);
    givens_apply_gv(W, k - 1, k, c, s);
    givens_apply_qr(Q, R, k - 1, k, c, s);
  }

  // add in w v^T
  for (int j = 0; j < M1_MULTIROOTS_DIM; j++) {
    Real r0j = R[0][j];
    Real vj = V[j];
    R[0][j] = r0j + W[0] * vj;
  }

  // apply Given's transformations
  for (int k = 1; k < M1_MULTIROOTS_DIM; k++) {
    Real c, s;
    Real diag = R[k - 1][k - 1];
    Real offdiag = R[k][k - 1];

    givens_rotation(diag, offdiag, c, s);
    givens_apply_qr(Q, R, k - 1, k, c, s);

    R[k][k - 1] = 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::qr_R_solve
//  \brief computes solution to the system R x = b where R is right triangular
//  matrix
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
