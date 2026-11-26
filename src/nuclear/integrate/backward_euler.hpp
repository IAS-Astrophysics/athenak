#ifndef NUCLEAR_INTEGRATE_BACKWARD_EULER_HPP_
#define NUCLEAR_INTEGRATE_BACKWARD_EULER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file backward_euler.hpp
//! \brief Backward Euler integrator of the reaction network for one cell: the dense LU
//! solver, the simplified Newton iteration and the per-cell driver BackwardEulerBurn().
//! Only the partial densities rho*X[n] are integrated (operator splitting); the energy
//! release is carried by the EOS through the binding energy in eps, which is held
//! fixed. Everything here is in cgs, T in K.

#include <cmath>

#include "athena.hpp"
#include "nuclear/network/network.hpp"
#include "nuclear/microphysics.hpp"
#include "nuclear/nuclear.hpp"

namespace nuclear {

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
bool FactorNewtonMatrix(const Real dt, const Real J[][NSPECIES], Real LU[][NSPECIES],
                        int piv[NSPECIES]) {
  for (int m=0; m<NSPECIES; m++) {
    for (int n=0; n<NSPECIES; n++) {
      LU[m][n] = ((m == n) ? 1.0 : 0.0) - dt*J[m][n];
    }
  }
  return LUDecompose<NSPECIES>(LU, piv);
}


//----------------------------------------------------------------------------------------
//! \fn BackEulerIteration()
//! \brief One (simplified) Newton iteration of the backward Euler step. Solves
//!   (I - dt*J) DUX0 = UX - UX0 + dt*F(UX0)
//! with the stored LU factors, where UX is the state at the start of the step and UX0
//! the current iterate. The caller then sets UX0 += DUX0. Only the partial densities
//! rho*X[n] are integrated (operator splitting): the energy release is carried by the
//! Helmholtz EOS, so there is no energy component.

KOKKOS_INLINE_FUNCTION
void BackEulerIteration(const Real rho, const Real dt, const Real UX[NSPECIES],
                        const Real UX0[NSPECIES], Real DUX0[NSPECIES],
                        const Real F[NSPECIES], const Real LU[][NSPECIES],
                        const int piv[NSPECIES], const Real y_atol, const Real y_rtol,
                        Real &abs_res, Real &rel_res, Real &err) {
  const Real x_floor = 1.0e-30;

  Real b[NSPECIES];
  for (int n=0; n<NSPECIES; n++) {
    b[n] = UX[n] - UX0[n] + dt*F[n];
  }

  LUSolve<NSPECIES>(LU, piv, b, DUX0);

  // Convergence error of rho*X[n] [g/cm^3], max over species, normalized by the mixed
  // tolerance y_atol + y_rtol*|x| (converged when err <= 1). abs_res/rel_res are the
  // plain absolute/relative updates (used to detect stalls).
  abs_res = 0.0;
  rel_res = 0.0;
  err = 0.0;
  for (int n=0; n<NSPECIES; n++) {
    Real dx = Kokkos::fabs(DUX0[n]);
    Real x_new = Kokkos::fabs(UX0[n] + DUX0[n]);
    abs_res = Kokkos::fmax(abs_res, dx);
    rel_res = Kokkos::fmax(rel_res, dx / Kokkos::fmax(x_new, rho*x_floor));
    err = Kokkos::fmax(err, dx / (y_atol + y_rtol*x_new));
  }
}


//----------------------------------------------------------------------------------------
//! \fn BackwardEulerBurn()
//! \brief Burns one cell over dt with a single backward Euler step of the conserved
//! rho*X[n], solved by simplified Newton iterations (Jacobian reused, refactored when
//! convergence stalls or every prm.be_n_refactor iterations). rho and eps are held
//! fixed; T follows from (rho, eps, X) through the EOS at each iterate.
//!
//! All in cgs:
//!   rho [g/cm^3], eps [erg/g], dt [s]
//!   X[n]   in: mass fractions at the start of the step
//!          out: burned mass fractions, renormalized so that sum_n X[n] = 1
//!   state  out: thermodynamic state at the burned composition (T [K], cv, dedX, ...)
//!   t_heat out: heating timescale cv*T/|edot| [s] at the burned state
//! eos must be the cgs copy of the EOS (SetCodeUnitSystem(Primitive::MakeCGS()), see
//! nuclear/microphysics.hpp). Returns true if the Newton iteration converged and the
//! result is finite; on false, X, state and t_heat hold the last iterate.

template <typename EOS>
KOKKOS_INLINE_FUNCTION
bool BackwardEulerBurn(const Real rho, const Real eps, const Real dt, Real X[NSPECIES],
                       const EOS &eos, const NuclearParams &prm, Thermo &state,
                       Real &t_heat) {
  // RHS, d(eps)/dX[n], and the conserved rho*X[n] vector: UX is the state at the
  // start of the step, UX0 the current Newton iterate and DUX0 the Newton update.
  Real F[NSPECIES];
  Real ex[NSPECIES];
  Real DUX0[NSPECIES];
  Real UX0[NSPECIES];
  Real UX[NSPECIES];

  // Jacobian and its LU factors (reused across iterations)
  Real J[NSPECIES][NSPECIES];
  Real LU[NSPECIES][NSPECIES];
  int piv[NSPECIES];

  Real abs_res, rel_res, err;
  Real enuc_dot;

  state.rho = rho;
  for (int n=0; n<NSPECIES; n++) {
    UX[n] = rho*X[n];
    UX0[n] = UX[n];
  }

  // Composition, T (at fixed eps), cv and d(e)/dX[n] of the current iterate UX0
  auto update_state = [&]() {
    for (int n=0; n<NSPECIES; n++) {
      state.xn[n] = UX0[n]/rho;
    }
    state.e = eps;
    burn(eos_input_re, state, eos);
    for (int n=0; n<NSPECIES; n++) {
      ex[n] = state.dedX[n];
    }
  };

  // Jacobian and LU factorization at the start of the step
  update_state();
  PopulateJac(rho, state.t, state.xn, state.cv, ex, J);
  bool factor_ok = FactorNewtonMatrix(dt, J, LU, piv);
  int iters_since_factor = 0;
  Real rel_res_old = 0.0;
  bool converged = false;

  for (int iter=0; (iter<prm.max_iter) && factor_ok; iter++) {
    ActualRHS(rho, state.t, state.xn, ex, F, enuc_dot);

    // Newton iteration, reusing the L and U factors
    BackEulerIteration(rho, dt, UX, UX0, DUX0, F, LU, piv, prm.y_atol, prm.y_rtol,
                       abs_res, rel_res, err);

    for (int n=0; n<NSPECIES; n++) {
      UX0[n] += DUX0[n];
    }
    update_state();
    iters_since_factor++;

    // Converged when, for every species, |d(rho*X)| <= y_atol + y_rtol*|rho*X|
    if (err <= 1.0) {
      converged = true;
      break;
    }

    // Refactor when convergence stalls or after be_n_refactor iterations
    bool stalled = (iters_since_factor > 1) && (rel_res > 0.5*rel_res_old);
    rel_res_old = rel_res;
    if (stalled || (iters_since_factor >= prm.be_n_refactor)) {
      PopulateJac(rho, state.t, state.xn, state.cv, ex, J);
      factor_ok = FactorNewtonMatrix(dt, J, LU, piv);
      iters_since_factor = 0;
    }
  }

  // Failure: singular matrix, no convergence within max_iter, or a non-finite result
  // (also T, e.g. eps outside the table)
  bool finite = Kokkos::isfinite(state.t);
  for (int n=0; n<NSPECIES; n++) {
    finite = finite && Kokkos::isfinite(UX0[n]);
  }

  // Mass fractions of the last iterate, renormalized so that sum_n X[n] = 1
  // (corrects the drift from the burn and from advecting the species separately)
  Real sum_rhox = 0.0;
  for (int n=0; n<NSPECIES; n++) {
    sum_rhox += UX0[n];
  }
  for (int n=0; n<NSPECIES; n++) {
    X[n] = UX0[n]/sum_rhox;
  }

  // Heating timescale cv*T/|edot| [s] at the burned state (state is up to date with
  // UX0 after the last update_state() call)
  t_heat = heating_timescale(rho, state.t, state.xn, ex, state.cv);

  return converged && finite;
}

}  // namespace nuclear

#endif  // NUCLEAR_INTEGRATE_BACKWARD_EULER_HPP_
