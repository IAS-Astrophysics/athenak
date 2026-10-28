#ifndef PGEN_FLUIDS_REACTIVE_BUBBLE_HPP_
#define PGEN_FLUIDS_REACTIVE_BUBBLE_HPP_

#include <cmath>

#include "athena.hpp"
#include "nuclear/network/network.hpp"

//----------------------------------------------------------------------------------------
//! \fn LUDecompose()
//! \brief In-place LU decomposition with partial pivoting (Doolittle). On return the
//! strict lower part of A holds L (unit diagonal implied), the upper part holds U, and
//! piv[k] is the row swapped into position k. Returns false if A is singular.

template<int N>
KOKKOS_INLINE_FUNCTION
bool LUDecompose(Real A[][N], int piv[N]) {
  for (int k=0; k<N; k++) {
    // find pivot row
    int p = k;
    Real amax = Kokkos::fabs(A[k][k]);
    for (int r=k+1; r<N; r++) {
      if (Kokkos::fabs(A[r][k]) > amax) {
        amax = Kokkos::fabs(A[r][k]);
        p = r;
      }
    }
    piv[k] = p;
    if (amax < 1.0e-300) return false;

    if (p != k) {
      for (int c=0; c<N; c++) {
        Real tmp = A[k][c];
        A[k][c] = A[p][c];
        A[p][c] = tmp;
      }
    }

    // eliminate below the pivot, storing multipliers in L
    Real pinv = 1.0/A[k][k];
    for (int r=k+1; r<N; r++) {
      A[r][k] *= pinv;
      for (int c=k+1; c<N; c++) {
        A[r][c] -= A[r][k]*A[k][c];
      }
    }
  }
  return true;
}


//----------------------------------------------------------------------------------------
//! \fn LUSolve()
//! \brief Solves A x = b using the factors from LUDecompose(). LU and piv are not
//! modified, so they can be reused for several right-hand sides.

template<int N>
KOKKOS_INLINE_FUNCTION
void LUSolve(const Real LU[][N], const int piv[N], const Real b[N], Real x[N]) {
  for (int n=0; n<N; n++) {
    x[n] = b[n];
  }

  // apply row permutation, in the same order as the factorization
  for (int k=0; k<N; k++) {
    if (piv[k] != k) {
      Real tmp = x[k];
      x[k] = x[piv[k]];
      x[piv[k]] = tmp;
    }
  }

  // forward substitution: L y = P b
  for (int r=1; r<N; r++) {
    for (int c=0; c<r; c++) {
      x[r] -= LU[r][c]*x[c];
    }
  }

  // back substitution: U x = y
  for (int r=N-1; r>=0; r--) {
    for (int c=r+1; c<N; c++) {
      x[r] -= LU[r][c]*x[c];
    }
    x[r] /= LU[r][r];
  }
}


//----------------------------------------------------------------------------------------
//! \fn FactorNewtonMatrix()
//! \brief Builds A = I - dt*J and LU-factorizes it in place into LU.

KOKKOS_INLINE_FUNCTION
bool FactorNewtonMatrix(const Real dt, const Real J[][NX], Real LU[][NX], int piv[NX]) {
  for (int m=0; m<NX; m++) {
    for (int n=0; n<NX; n++) {
      LU[m][n] = ((m == n) ? 1.0 : 0.0) - dt*J[m][n];
    }
  }
  return LUDecompose<NX>(LU, piv);
}


//----------------------------------------------------------------------------------------
//! \fn BackEulerIteration()
//! \brief One (simplified) Newton iteration of the backward Euler step. Solves
//!   (I - dt*J) DUX0 = UX - UX0 + dt*F(UX0)
//! with the stored LU factors, where UX is the state at the start of the step and UX0
//! the current iterate. The caller then sets UX0 += DUX0. Errors are measured from the
//! Newton step: index 0 is the energy component, index 1 the max over mass fractions.

KOKKOS_INLINE_FUNCTION
void BackEulerIteration(const Real rho, const Real dt, const Real UX[NX],
                        const Real UX0[NX], Real DUX0[NX], const Real F[NX],
                        const Real LU[][NX], const int piv[NX],
                        const Real e_atol, const Real e_rtol,
                        const Real y_atol, const Real y_rtol,
                        Real abs_res[2], Real rel_res[2], Real err[2]) {
  const Real tiny = 1.0e-30;
  const Real x_floor = 1.0e-30;

  Real b[NX];
  for (int n=0; n<NX; n++) {
    b[n] = UX[n] - UX0[n] + dt*F[n];
  }

  LUSolve<NX>(LU, piv, b, DUX0);

  // Convergence errors of the conserved variables, normalized by the mixed tolerance
  // atol + rtol*|x| (converged when err <= 1):
  //   err[0]: rho*e  [erg/cm^3]  with e_atol, e_rtol
  //   err[1]: max_n rho*X[n] [g/cm^3] with y_atol, y_rtol
  // abs_res/rel_res are the plain absolute/relative updates (used to detect stalls).

  // Energy component
  Real e_new = Kokkos::fabs(UX0[IE] + DUX0[IE]);
  abs_res[0] = Kokkos::fabs(DUX0[IE]);
  rel_res[0] = abs_res[0] / Kokkos::fmax(e_new, tiny);
  err[0] = abs_res[0] / (e_atol + e_rtol*e_new);

  // Partial density (rho*X[n]) components
  abs_res[1] = 0.0;
  rel_res[1] = 0.0;
  err[1] = 0.0;
  for (int n=0; n<NSPECIES; n++) {
    Real dx = Kokkos::fabs(DUX0[n]);
    Real x_new = Kokkos::fabs(UX0[n] + DUX0[n]);
    abs_res[1] = Kokkos::fmax(abs_res[1], dx);
    rel_res[1] = Kokkos::fmax(rel_res[1], dx / Kokkos::fmax(x_new, rho*x_floor));
    err[1] = Kokkos::fmax(err[1], dx / (y_atol + y_rtol*x_new));
  }
}

#endif  // PGEN_FLUIDS_REACTIVE_BUBBLE_HPP_

