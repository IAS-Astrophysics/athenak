#ifndef RADIATION_M1_ROOTS_HYBRIDJ_HPP
#define RADIATION_M1_ROOTS_HYBRIDJ_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_roots_hybridj.hpp
//  \brief functions for Powell's multiroot solver

#include "athena.hpp"
#include "radiation_m1_macro.hpp"

namespace radiationm1 {

struct HybridsjState {
  size_t iter;
  size_t ncfail;
  size_t ncsuc;
  size_t nslow1;
  size_t nslow2;
  Real fnorm;
  Real delta;
  Real q[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM];
  Real r[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM];
  Real tau[M1_MULTIROOTS_DIM];
  Real diag[M1_MULTIROOTS_DIM];
  Real qtf[M1_MULTIROOTS_DIM];
  Real newton[M1_MULTIROOTS_DIM];
  Real gradient[M1_MULTIROOTS_DIM];
  Real x_trial[M1_MULTIROOTS_DIM];
  Real f_trial[M1_MULTIROOTS_DIM];
  Real df[M1_MULTIROOTS_DIM];
  Real qtdf[M1_MULTIROOTS_DIM];
  Real rdx[M1_MULTIROOTS_DIM];
  Real w[M1_MULTIROOTS_DIM];
  Real v[M1_MULTIROOTS_DIM];

  Real x[M1_MULTIROOTS_DIM];
  Real f[M1_MULTIROOTS_DIM];
  Real J[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM];
  Real dx[M1_MULTIROOTS_DIM];
};

enum HybridsjSignal {
  HYBRIDSJ_ENOPROGJ,
  HYBRIDSJ_EBADFUNC,
  HYBRIDSJ_EINVAL,
  HYBRIDSJ_SUCCESS,
  HYBRIDSJ_CONTINUE,
};

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::copy_vector
//  \brief copy a vector from src to dest
KOKKOS_INLINE_FUNCTION
void copy_vector(Real (&dest)[M1_MULTIROOTS_DIM],
                 const Real (&src)[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    dest[i] = src[i];
  }
}

KOKKOS_INLINE_FUNCTION
void
//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::enorm
//  \brief computes the L2 norm of a vector f
KOKKOS_INLINE_FUNCTION
Real enorm(const Real (&f)[M1_MULTIROOTS_DIM]) {
  Real result2 = 0;
  for (double i : f) {
    result2 += i * i;
  }
  return Kokkos::sqrt(result2);
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::enorm_sum
//  \brief computes the L2 norm of the sum of two vector a,b
KOKKOS_INLINE_FUNCTION
Real enorm_sum(const Real (&a)[M1_MULTIROOTS_DIM],
               const Real (&b)[M1_MULTIROOTS_DIM]) {
  Real result2 = 0;
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    result2 += (a[i] + b[i]) * (a[i] + b[i]);
  }
  return Kokkos::sqrt(result2);
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::scaled_enorm
//  \brief computes the scaled L2 norm of a vector f
KOKKOS_INLINE_FUNCTION
Real scaled_enorm(const Real (&d)[M1_MULTIROOTS_DIM],
                  const Real (&f)[M1_MULTIROOTS_DIM]) {
  Real result2 = 0;
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    result2 += (d[i] * f[i]) * (d[i] * f[i]);
  }
  return Kokkos::sqrt(result2);
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::compute_diag
//  \brief computes the columnwise L2 norm of a matrix J
KOKKOS_INLINE_FUNCTION
void compute_diag(Real (&J)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                  Real (&diag)[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    Real sum = 0;
    for (int j = 0; j < M1_MULTIROOTS_DIM; j++) {
      sum += J[j][i] * J[j][i];
    }
    if (sum == 0) {
      sum = 1.0;
    }
    diag[i] = Kokkos::sqrt(sum);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::update_diag
//  \brief updates the columnwise L2 norm of a matrix J
KOKKOS_INLINE_FUNCTION
void update_diag(const Real (&J)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                 Real (&diag)[M1_MULTIROOTS_DIM]) {
  for (int j = 0; j < M1_MULTIROOTS_DIM; j++) {
    Real cnorm, sum = 0;
    for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
      sum += J[i][j] * J[i][j];
    }
    if (sum == 0) {
      sum = 1.0;
    }
    cnorm = Kokkos::sqrt(sum);

    if (cnorm > diag[j])
      diag[j] = cnorm;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::compute_delta
//  \brief updates the columnwise L2 norm of a matrix J
KOKKOS_INLINE_FUNCTION
Real compute_delta(const Real (&diag)[M1_MULTIROOTS_DIM],
                   const Real (&x)[M1_MULTIROOTS_DIM]) {
  Real Dx = scaled_enorm(diag, x);
  Real factor = 100;

  return (Dx > 0) ? factor * Dx : factor;
}

KOKKOS_INLINE_FUNCTION
void minimum_step(const Real gnorm, const Real (&diag)[M1_MULTIROOTS_DIM],
                  Real (&g)[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    g[i] = (g[i] / gnorm) / diag[i];
  }
}
KOKKOS_INLINE_FUNCTION
void compute_qtf(const Real (&q)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                 const Real (&f)[M1_MULTIROOTS_DIM],
                 Real (&qtf)[M1_MULTIROOTS_DIM]) {
  for (int j = 0; j < M1_MULTIROOTS_DIM; j++) {
    Real sum = 0;
    for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
      sum += q[i][j] * f[i];
    }
    qtf[j] = sum;
  }
}

KOKKOS_INLINE_FUNCTION
int newton_direction(const Real r[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                     const Real qtf[M1_MULTIROOTS_DIM],
                     Real p[M1_MULTIROOTS_DIM]) {
  int status;

  status = gsl_linalg_R_solve(r, qtf, p);

  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    Real pi = p[i];
    p[i] = -pi;
  }

  return status;
}

KOKKOS_INLINE_FUNCTION void
gradient_direction(const Real r[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                   const Real qtf[M1_MULTIROOTS_DIM],
                   const Real diag[M1_MULTIROOTS_DIM],
                   Real g[M1_MULTIROOTS_DIM]) {
  for (int j = 0; j < M1_MULTIROOTS_DIM; j++) {
    Real sum = 0;
    Real dj;
    for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
      sum += r[i][j] * qtf[i];
    }
    dj = diag[j];
    g[j] = -sum / dj;
  }
}

KOKKOS_INLINE_FUNCTION void compute_df(const Real f_trial[M1_MULTIROOTS_DIM],
                                       const Real f[M1_MULTIROOTS_DIM],
                                       Real df[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    df[i] = f_trial[i] - f[i];
  }
}

KOKKOS_INLINE_FUNCTION
void compute_wv(const Real qtdf[M1_MULTIROOTS_DIM],
                const Real rdx[M1_MULTIROOTS_DIM],
                const Real dx[M1_MULTIROOTS_DIM],
                const Real diag[M1_MULTIROOTS_DIM], Real pnorm,
                Real w[M1_MULTIROOTS_DIM], Real v[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    w[i] = (qtdf[i] - rdx[i]) / pnorm;
    v[i] = diag[i] * diag[i] * dx[i] / pnorm;
  }
}

KOKKOS_INLINE_FUNCTION
void compute_Rg(const Real (&r)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                const Real (&gradient)[M1_MULTIROOTS_DIM],
                Real (&Rg)[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    double sum = 0;
    for (int j = i; j < M1_MULTIROOTS_DIM; j++) {
      sum += r[i][j] * gradient[j];
    }
    Rg[i] = sum;
  }
}

KOKKOS_INLINE_FUNCTION
Real compute_actual_reduction(const Real fnorm, const Real fnorm1) {
  Real actred{};
  if (fnorm1 < fnorm) {
    Real u = fnorm1 / fnorm;
    actred = 1 - u * u;
  } else {
    actred = -1;
  }
  return actred;
}

KOKKOS_INLINE_FUNCTION
Real compute_predicted_reduction(const Real fnorm, const Real fnorm1) {
  Real prered{};
  if (fnorm1 < fnorm) {
    Real u = fnorm1 / fnorm;
    prered = 1 - u * u;
  } else {
    prered = 0;
  }
  return prered;
}

KOKKOS_INLINE_FUNCTION
void compute_rdx(const Real r[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                 const Real dx[M1_MULTIROOTS_DIM],
                 Real rdx[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    Real sum = 0;
    for (int j = i; j < M1_MULTIROOTS_DIM; j++) {
      sum += r[i][j] * dx[j];
    }
    rdx[i] = sum;
  }
}

KOKKOS_INLINE_FUNCTION
void scaled_addition(Real alpha, const Real newton[M1_MULTIROOTS_DIM],
                     Real beta, const Real gradient[M1_MULTIROOTS_DIM],
                     Real p[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    p[i] = alpha * newton[i] + beta * gradient[i];
  }
}

KOKKOS_INLINE_FUNCTION
HybridsjSignal dogleg(const Real (&r)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                      const Real (&qtf)[M1_MULTIROOTS_DIM],
                      const Real (&diag)[M1_MULTIROOTS_DIM], const Real &delta,
                      Real (&newton)[M1_MULTIROOTS_DIM],
                      Real (&gradient)[M1_MULTIROOTS_DIM],
                      Real (&p)[M1_MULTIROOTS_DIM]) {
  Real qnorm, gnorm, sgnorm, bnorm, temp;

  newton_direction(r, qtf, newton);

  qnorm = scaled_enorm(diag, newton);
  if (qnorm <= delta) {
    copy_vector(p, newton);
    return HYBRIDSJ_SUCCESS;
  }

  gradient_direction(r, qtf, diag, gradient);
  gnorm = enorm(gradient);
  if (gnorm == 0) {
    double alpha = delta / qnorm;
    double beta = 0;
    scaled_addition(alpha, newton, beta, gradient, p);
    return HYBRIDSJ_SUCCESS;
  }

  minimum_step(gnorm, diag, gradient);

  // compute Rg and store it temporarily in p
  compute_Rg(r, gradient, p);

  temp = enorm(p);
  sgnorm = (gnorm / temp) / temp;

  if (sgnorm > delta) {
    double alpha = 0;
    double beta = delta;
    scaled_addition(alpha, newton, beta, gradient, p);
    return HYBRIDSJ_SUCCESS;
  }

  bnorm = enorm(qtf);

  {
    Real bg = bnorm / gnorm;
    Real bq = bnorm / qnorm;
    Real dq = delta / qnorm;
    Real dq2 = dq * dq;
    Real sd = sgnorm / delta;
    Real sd2 = sd * sd;

    Real t1 = bg * bq * sd;
    Real u = t1 - dq;
    Real t2 = t1 - dq * sd2 + sqrt(u * u + (1 - dq2) * (1 - sd2));

    Real alpha = dq * (1 - sd2) / t2;
    Real beta = (1 - alpha) * sgnorm;

    scaled_addition(alpha, newton, beta, gradient, p);
  }

  return HYBRIDSJ_SUCCESS;
}

KOKKOS_INLINE_FUNCTION
void compute_trial_step(const Real x[M1_MULTIROOTS_DIM],
                        const Real dx[M1_MULTIROOTS_DIM],
                        Real x_trial[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    x_trial[i] = x[i] + dx[i];
  }
}

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::HybridsjInitialize
//  \brief Initialize the solver state for Powell's hybrid method
template <class Functor, class... Types>
KOKKOS_INLINE_FUNCTION HybridsjSignal HybridsjInitialize(Functor &&fdf,
                                                         HybridsjState &state) {
  // @TODO: call function multiroot
  // GSL_MULTIROOT_FN_EVAL_F_DF(fdf, x, f, J);

  state.iter = 1;
  state.fnorm = enorm(state.f);
  state.ncfail = 0;
  state.ncsuc = 0;
  state.nslow1 = 0;
  state.nslow2 = 0;

  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    state.df[i] = 0;
  }

  // store column norms in diag
  compute_diag(state.J, state.diag);

  // Set delta to factor |D x| or to factor if |D x| is zero
  state.delta = compute_delta(state.diag, state.x);

  // QR factorization of J
  // gsl_linalg_QR_decomp(J, state.tau);
  // gsl_linalg_QR_unpack(J, state.tau, state.q, state.r);

  return HYBRIDSJ_SUCCESS;
}

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::HybridsjIterate
//  \brief Iterate the solver state for Powell's hybrid method
template <class Functor, class... Types>
KOKKOS_INLINE_FUNCTION HybridsjSignal HybridsjIterate(Functor &&fdf,
                                                      HybridsjState &state) {
  Real prered{}, actred{};
  Real pnorm{}, fnorm1{}, fnorm1p{};
  Real ratio{};
  Real p1 = 0.1, p5 = 0.5, p001 = 0.001, p0001 = 0.0001;

  // Compute qtf = Q^T f
  compute_qtf(state.q, state.f, state.qtf);

  // Compute dogleg step
  dogleg(state.r, state.qtf, state.diag, state.delta, state.newton,
         state.gradient, state.dx);

  /* Take a trial step */
  compute_trial_step(state.x, state.dx, state.x_trial);

  pnorm = scaled_enorm(state.diag, state.dx);

  if (state.iter == 1) {
    if (pnorm < state.delta) {
      state.delta = pnorm;
    }
  }

  /* Evaluate function at x + p */

  {
    int status = GSL_MULTIROOT_FN_EVAL_F(fdf, x_trial, f_trial);

    if (status != HYBRIDSJ_SUCCESS) {
      return HYBRIDSJ_EBADFUNC;
    }
  }

  /* Set df = f_trial - f */

  compute_df(f_trial, f, df);

  /* Compute the scaled actual reduction */

  fnorm1 = enorm(f_trial);

  actred = compute_actual_reduction(fnorm, fnorm1);

  /* Compute rdx = R dx */

  compute_rdx(r, dx, rdx);

  /* Compute the scaled predicted reduction phi1p = |Q^T f + R dx| */

  fnorm1p = enorm_sum(qtf, rdx);

  prered = compute_predicted_reduction(fnorm, fnorm1p);

  /* Compute the ratio of the actual to predicted reduction */

  if (prered > 0) {
    ratio = actred / prered;
  } else {
    ratio = 0;
  }

  /* Update the step bound */

  if (ratio < p1) {
    state.ncsuc = 0;
    state.ncfail++;
    state.delta *= p5;
  } else {
    state.ncfail = 0;
    state.ncsuc++;

    if (ratio >= p5 || state.ncsuc > 1)
      state.delta = GSL_MAX(state.delta, pnorm / p5);
    if (fabs(ratio - 1) <= p1)
      state->delta = pnorm / p5;
  }

  /* Test for successful iteration */

  if (ratio >= p0001) {
    gsl_vector_memcpy(x, x_trial);
    gsl_vector_memcpy(f, f_trial);
    state->fnorm = fnorm1;
    state->iter++;
  }

  /* Determine the progress of the iteration */

  state->nslow1++;
  if (actred >= p001)
    state->nslow1 = 0;

  if (actred >= p1)
    state->nslow2 = 0;

  if (state->ncfail == 2) {
    {
      int status = GSL_MULTIROOT_FN_EVAL_DF(fdf, x, J);

      if (status != GSL_SUCCESS) {
        return GSL_EBADFUNC;
      }
    }

    state->nslow2++;

    if (state->iter == 1) {
      if (scale)
        compute_diag(J, diag);
      state->delta = compute_delta(diag, x);
    } else {
      if (scale)
        update_diag(J, diag);
    }

    /* Factorize J into QR decomposition */

    gsl_linalg_QR_decomp(J, tau);
    gsl_linalg_QR_unpack(J, tau, q, r);
    return GSL_SUCCESS;
  }

  /* Compute qtdf = Q^T df, w = (Q^T df - R dx)/|dx|,  v = D^2 dx/|dx| */

  compute_qtf(q, df, qtdf);

  compute_wv(qtdf, rdx, dx, diag, pnorm, w, v);

  /* Rank-1 update of the jacobian Q'R' = Q(R + w v^T) */

  // gsl_linalg_QR_update(q, r, w, v); @TODO: implement this

  // No progress as measured by jacobian evaluations
  if (state.nslow2 == 5) {
    return GSL_ENOPROGJ;
  }

  // No progress as measured by function evaluations
  if (state.nslow1 == 10) {
    return HYBRIDSJ_ENOPROG;
  }
  return HYBRIDSJ_SUCCESS;
}
} // namespace radiationm1
#endif // RADIATION_M1_ROOTS_HYBRIDJ_HPP
