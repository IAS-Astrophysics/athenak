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
#include "radiation_m1_linalg.hpp"
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
  Real diag[M1_MULTIROOTS_DIM];
  Real qtf[M1_MULTIROOTS_DIM];
  Real newton[M1_MULTIROOTS_DIM];
  Real gradient[M1_MULTIROOTS_DIM];
  Real x_trial[M1_MULTIROOTS_DIM];
  Real f_trial[M1_MULTIROOTS_DIM];
  Real J_trial[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM];
  Real df[M1_MULTIROOTS_DIM];
  Real qtdf[M1_MULTIROOTS_DIM];
  Real rdx[M1_MULTIROOTS_DIM];
  Real w[M1_MULTIROOTS_DIM];
  Real v[M1_MULTIROOTS_DIM];
};

struct HybridsjParams {
  Real x[M1_MULTIROOTS_DIM]; // current solution {x_i} in N dimensional space
  Real f[M1_MULTIROOTS_DIM]; // N function values f_i({x_i})
  Real J[M1_MULTIROOTS_DIM]
        [M1_MULTIROOTS_DIM];  // Jacobian values J_ij = \p f_i/\p x_j at {x_i}
  Real dx[M1_MULTIROOTS_DIM]; // stores the dogleg step J dx = - f
};

enum HybridsjSignal {
  HYBRIDSJ_ENOPROGJ,
  HYBRIDSJ_EBADFUNC,
  HYBRIDSJ_EINVAL,
  HYBRIDSJ_SUCCESS,
  HYBRIDSJ_CONTINUE,
  HYBRIDSJ_EBADTOL,
};

KOKKOS_INLINE_FUNCTION Real set_dthin(Real chi) { return 1.5 * chi - 0.5; }
KOKKOS_INLINE_FUNCTION Real set_dthick(Real chi) { return 1.5 * (1 - chi); }

// Low level kernel computing the Jacobian matrix
KOKKOS_INLINE_FUNCTION void source_jacobian(
    const Real qpre[4], AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>(&F_u), Real &F2,
    const Real &chi, const Real &kapa, const Real &kaps,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>(&v_u),
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>(&v_d), const Real &v2,
    const Real &W, const Real &alpha, const Real &cdt, const Real qstar[4],
    Real (&J)[4][4]) {
  const Real kapas = kapa + kaps;
  const Real alpW = alpha * W;

  const Real dthin = set_dthin(chi);
  const Real dthick = set_dthick(chi);

  const Real vx = v_d(1);
  const Real vy = v_d(2);
  const Real vz = v_d(3);
  const Real W2 = SQ(W);
  const Real W3 = W2 * W;

  const Real vdotF = F_u(1) * v_d(1) + F_u(2) * v_d(2) + F_u(3) * v_d(3);
  const Real normF = Kokkos::sqrt(F2);
  const Real inormF = (normF > 0 ? 1 / normF : 0);
  const Real vdothatf = vdotF * inormF;
  const Real vdothatf2 = SQ(vdothatf);
  const Real hatfx = qpre[1] * inormF;  // hatf_i
  const Real hatfy = qpre[2] * inormF;
  const Real hatfz = qpre[3] * inormF;
  const Real hatfupx = F_u(1) * inormF;  // hatf^i
  const Real hatfupy = F_u(2) * inormF;
  const Real hatfupz = F_u(3) * inormF;
  const Real e = qpre[0];
  const Real eonormF = Kokkos::min<Real>(e * inormF, 1.0);
  // with factor dthin ...

  // drvts of J
  Real JdE =
      W2 + dthin * vdothatf2 * W2 + (dthick * (3 - 2 * W2) * (-1 + W2)) / (1 + 2 * W2);

  Real JdFv =
      2 * W2 *
      (-1 + (dthin * eonormF * vdothatf) + (2 * dthick * (-1 + W2)) / (1 + 2 * W2));
  Real JdFf = (-2 * dthin * eonormF * vdothatf2 * W2);

  Real JdFx = JdFv * v_u(1) + JdFf * hatfupx;
  Real JdFy = JdFv * v_u(2) + JdFf * hatfupy;
  Real JdFz = JdFv * v_u(3) + JdFf * hatfupz;

  // drvts of Hi
  Real HdEv = W3 * (-1 - dthin * vdothatf2 + (dthick * (-3 + 2 * W2)) / (1 + 2 * W2));
  Real HdEf = -(dthin * vdothatf * W);

  Real HxdE = HdEv * vx + HdEf * hatfx;
  Real HydE = HdEv * vy + HdEf * hatfy;
  Real HzdE = HdEv * vz + HdEf * hatfz;

  Real HdFdelta = (1 - dthick * v2 - (dthin * eonormF * vdothatf)) * W;
  Real HdFvv = (2 * (1 - dthin * eonormF * vdothatf) * W3) +
               dthick * W * (2 - 2 * W2 + 1 / (-1 - 2 * W2));
  Real HdFff = (2 * dthin * eonormF * vdothatf * W);
  Real HdFvf = (2 * dthin * eonormF * vdothatf2 * W3);
  Real HdFfv = -(dthin * eonormF * W);

  Real HxdFx = HdFdelta + HdFvv * vx * v_u(1) + HdFff * hatfx * hatfupx +
               HdFvf * vx * hatfupx + HdFfv * hatfx * v_u(1);
  Real HydFx = HdFvv * vy * v_u(1) + HdFff * hatfy * hatfupx + HdFvf * vy * hatfupx +
               HdFfv * hatfy * v_u(1);
  Real HzdFx = HdFvv * vz * v_u(1) + HdFff * hatfz * hatfupx + HdFvf * vz * hatfupx +
               HdFfv * hatfz * v_u(1);

  Real HxdFy = HdFvv * vx * v_u(2) + HdFff * hatfx * hatfupy + HdFvf * vx * hatfupy +
               HdFfv * hatfx * v_u(2);
  Real HydFy = HdFdelta + HdFvv * vy * v_u(2) + HdFff * hatfy * hatfupy +
               HdFvf * vy * hatfupy + HdFfv * hatfy * v_u(2);
  Real HzdFy = HdFvv * vz * v_u(2) + HdFff * hatfz * hatfupy + HdFvf * vz * hatfupy +
               HdFfv * hatfz * v_u(2);

  Real HxdFz = HdFvv * vx * v_u(3) + HdFff * hatfx * hatfupz + HdFvf * vx * hatfupz +
               HdFfv * hatfx * v_u(3);
  Real HydFz = HdFvv * vy * v_u(3) + HdFff * hatfy * hatfupz + HdFvf * vy * hatfupz +
               HdFfv * hatfy * v_u(3);
  Real HzdFz = HdFdelta + HdFvv * vz * v_u(3) + HdFff * hatfz * hatfupz +
               HdFvf * vz * hatfupz + HdFfv * hatfz * v_u(3);

  // Build the Jacobian
  Real J00 = -alpW * (kapas - kaps * JdE);

  Real J0x = +alpW * kaps * JdFx + alpW * kapas * v_u(1);
  Real J0y = +alpW * kaps * JdFy + alpW * kapas * v_u(2);
  Real J0z = +alpW * kaps * JdFz + alpW * kapas * v_u(3);

  Real Jx0 = -alpha * (kapas * HxdE + W * kapa * vx * JdE);
  Real Jy0 = -alpha * (kapas * HydE + W * kapa * vy * JdE);
  Real Jz0 = -alpha * (kapas * HzdE + W * kapa * vz * JdE);

  Real Jxx = -alpha * (kapas * HxdFx + W * kapa * vx * JdFx);
  Real Jxy = -alpha * (kapas * HxdFy + W * kapa * vx * JdFy);
  Real Jxz = -alpha * (kapas * HxdFz + W * kapa * vx * JdFz);

  Real Jyy = -alpha * (kapas * HydFx + W * kapa * vy * JdFx);
  Real Jyx = -alpha * (kapas * HydFy + W * kapa * vy * JdFy);
  Real Jyz = -alpha * (kapas * HydFz + W * kapa * vy * JdFz);

  Real Jzx = -alpha * (kapas * HzdFx + W * kapa * vz * JdFx);
  Real Jzy = -alpha * (kapas * HzdFy + W * kapa * vz * JdFy);
  Real Jzz = -alpha * (kapas * HzdFz + W * kapa * vz * JdFz);

  // Store Jacobian into J
  J[0][0] = 1 - cdt * J00;
  J[0][1] = -cdt * J0x;
  J[0][2] = -cdt * J0y;
  J[0][2] = -cdt * J0z;
  J[1][0] = -cdt * Jx0;
  J[1][1] = 1 - cdt * Jxx;
  J[1][2] = -cdt * Jxy;
  J[1][3] = -cdt * Jxz;
  J[2][0] = -cdt * Jy0;
  J[2][1] = -cdt * Jyx;
  J[2][2] = 1 - cdt * Jyy;
  J[2][3] = -cdt * Jyz;
  J[3][0] = -cdt * Jz0;
  J[3][1] = -cdt * Jzx;
  J[3][2] = -cdt * Jzy;
  J[3][3] = 1 - cdt * Jzz;
}

class HybridsjFunctor {
public:
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real (&x)[M1_MULTIROOTS_DIM], Real (&f)[M1_MULTIROOTS_DIM],
                  Real (&J)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM], HybridsjState &state,
                  HybridsjParams &pars, SrcParams &p, const RadiationM1Params &params_) {
    // Function to rootfind for
    //    f(q) = q - q^* - dt S[q]
    // auto ierr = RadiationM1::prepare(x, p); @TODO: fix
    f[0] = x[0] - p.Estar - p.cdt * p.Edot;
    f[1] = x[1] - p.Fstar_d(1) - p.cdt * p.tS_d(1);
    f[2] = x[2] - p.Fstar_d(2) - p.cdt * p.tS_d(2);
    f[3] = x[3] - p.Fstar_d(3) - p.cdt * p.tS_d(3);

    Real m_q[] = {p.E, p.F_d(1), p.F_d(2), p.F_d(3)};
    Real m_F2 = tensor_dot(p.F_u, p.F_d);
    Real m_v2 = tensor_dot(p.v_u, p.v_d);
    Real m_qstar[] = {p.Estar, p.Fstar_d(1), p.Fstar_d(2), p.Fstar_d(3)};

    source_jacobian(m_q, p.F_u, m_F2, p.chi, p.kscat, p.kabs, p.v_u, p.v_d, m_v2, p.W,
                    p.alp, p.cdt, m_qstar, J);
  }
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
//  \brief store columnwise L2 norm of J in diag
KOKKOS_INLINE_FUNCTION
void compute_diag(const Real (&J)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
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
    if (cnorm > diag[j]) {
      diag[j] = cnorm;
    }
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
void minimum_step(const Real &gnorm, const Real (&diag)[M1_MULTIROOTS_DIM],
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
void newton_direction(const Real (&r)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                      const Real (&qtf)[M1_MULTIROOTS_DIM],
                      Real (&p)[M1_MULTIROOTS_DIM]) {
  qr_R_solve(r, qtf, p);

  for (double &i : p) {
    Real pi = i;
    i = -pi;
  }
}

KOKKOS_INLINE_FUNCTION void
gradient_direction(const Real (&r)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                   const Real (&qtf)[M1_MULTIROOTS_DIM],
                   const Real (&diag)[M1_MULTIROOTS_DIM],
                   Real (&g)[M1_MULTIROOTS_DIM]) {
  for (int j = 0; j < M1_MULTIROOTS_DIM; j++) {
    Real sum = 0;
    for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
      sum += r[i][j] * qtf[i];
    }
    g[j] = -sum / diag[j];
  }
}

KOKKOS_INLINE_FUNCTION void compute_df(const Real (&f_trial)[M1_MULTIROOTS_DIM],
                                       const Real (&f)[M1_MULTIROOTS_DIM],
                                       Real (&df)[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    df[i] = f_trial[i] - f[i];
  }
}

KOKKOS_INLINE_FUNCTION
void compute_wv(const Real (&qtdf)[M1_MULTIROOTS_DIM],
                const Real (&rdx)[M1_MULTIROOTS_DIM],
                const Real (&dx)[M1_MULTIROOTS_DIM],
                const Real (&diag)[M1_MULTIROOTS_DIM], Real &pnorm,
                Real (&w)[M1_MULTIROOTS_DIM], Real (&v)[M1_MULTIROOTS_DIM]) {
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
    Real sum = 0;
    for (int j = i; j < M1_MULTIROOTS_DIM; j++) {
      sum += r[i][j] * gradient[j];
    }
    Rg[i] = sum;
  }
}

KOKKOS_INLINE_FUNCTION
Real compute_actual_reduction(const Real &fnorm, const Real &fnorm1) {
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
Real compute_predicted_reduction(const Real &fnorm, const Real &fnorm1) {
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
void compute_rdx(const Real (&r)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                 const Real (&dx)[M1_MULTIROOTS_DIM],
                 Real (&rdx)[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    Real sum = 0;
    for (int j = i; j < M1_MULTIROOTS_DIM; j++) {
      sum += r[i][j] * dx[j];
    }
    rdx[i] = sum;
  }
}

KOKKOS_INLINE_FUNCTION
void scaled_addition(Real &alpha, const Real (&newton)[M1_MULTIROOTS_DIM],
                     Real &beta, const Real (&gradient)[M1_MULTIROOTS_DIM],
                     Real (&p)[M1_MULTIROOTS_DIM]) {
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
  newton_direction(r, qtf, newton);
  Real qnorm = scaled_enorm(diag, newton);
  if (qnorm <= delta) {
    copy_vector(p, newton);
    return HYBRIDSJ_SUCCESS;
  }

  gradient_direction(r, qtf, diag, gradient);
  Real gnorm = enorm(gradient);
  if (gnorm == 0) {
    Real alpha = delta / qnorm;
    Real beta = 0;
    scaled_addition(alpha, newton, beta, gradient, p);
    return HYBRIDSJ_SUCCESS;
  }

  minimum_step(gnorm, diag, gradient);

  // compute Rg and store it temporarily in p
  compute_Rg(r, gradient, p);

  Real temp = enorm(p);
  Real sgnorm = (gnorm / temp) / temp;

  if (sgnorm > delta) {
    Real alpha = 0;
    Real beta = delta;
    scaled_addition(alpha, newton, beta, gradient, p);
    return HYBRIDSJ_SUCCESS;
  }

  Real bnorm = enorm(qtf);

  Real bg = bnorm / gnorm;
  Real bq = bnorm / qnorm;
  Real dq = delta / qnorm;
  Real dq2 = dq * dq;
  Real sd = sgnorm / delta;
  Real sd2 = sd * sd;

  Real t1 = bg * bq * sd;
  Real u = t1 - dq;
  Real t2 = t1 - dq * sd2 + Kokkos::sqrt(u * u + (1 - dq2) * (1 - sd2));

  Real alpha = dq * (1 - sd2) / t2;
  Real beta = (1 - alpha) * sgnorm;

  scaled_addition(alpha, newton, beta, gradient, p);

  return HYBRIDSJ_SUCCESS;
}

KOKKOS_INLINE_FUNCTION
void compute_trial_step(const Real (&x)[M1_MULTIROOTS_DIM],
                        const Real (&dx)[M1_MULTIROOTS_DIM],
                        Real (&x_trial)[M1_MULTIROOTS_DIM]) {
  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    x_trial[i] = x[i] + dx[i];
  }
}

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::HybridsjInitialize
//  \brief Initialize the solver state for Powell's hybrid method
template <class Functor, class... Types>
KOKKOS_INLINE_FUNCTION HybridsjSignal HybridsjInitialize(Functor &&fdf,
                                                         HybridsjState &state,
                                                         HybridsjParams &pars) {
  // populate f, J for a given x
  fdf(pars.x, pars.f, pars.J, state, pars);

  state.iter = 1;
  state.fnorm = enorm(pars.f);
  state.ncfail = 0;
  state.ncsuc = 0;
  state.nslow1 = 0;
  state.nslow2 = 0;

  for (double &i : state.df) {
    i = 0;
  }

  // store column norms, set delta and QR factorize J
  compute_diag(pars.J, state.diag);
  state.delta = compute_delta(state.diag, pars.x);
  qr_factorize(pars.J, state.q, state.r);

  return HYBRIDSJ_SUCCESS;
}

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::HybridsjIterate
//  \brief Iterate the solver state once for Powell's hybrid method
template <class Functor, class... Types>
KOKKOS_INLINE_FUNCTION HybridsjSignal HybridsjIterate(Functor &&fdf,
                                                      HybridsjState &state,
                                                      HybridsjParams &pars) {
  Real p1 = 0.1, p5 = 0.5, p001 = 0.001, p0001 = 0.0001;

  // Q^T f & dogleg
  compute_qtf(state.q, pars.f, state.qtf);
  HybridsjSignal dl = dogleg(state.r, state.qtf, state.diag, state.delta,
                             state.newton, state.gradient, pars.dx);

  // compute trial step
  compute_trial_step(pars.x, pars.dx, state.x_trial);
  Real pnorm = scaled_enorm(state.diag, pars.dx);
  if (state.iter == 1) {
    if (pnorm < state.delta) {
      state.delta = pnorm;
    }
  }

  // evaluate f at x + p
  fdf(state.x_trial, state.f_trial, state.J_trial, state, pars);

  // df = f_trial - f
  compute_df(state.f_trial, pars.f, state.df);
  // scaled actual reduction
  Real fnorm1 = enorm(state.f_trial);
  Real actred = compute_actual_reduction(state.fnorm, fnorm1);
  // rdx = R dx
  compute_rdx(state.r, pars.dx, state.rdx);
  // scaled predicted reduction
  Real fnorm1p = enorm_sum(state.qtf, state.rdx);
  Real prered = compute_predicted_reduction(state.fnorm, fnorm1p);
  // Ratio actual/predicted reduction
  Real ratio = (prered > 0) ? actred / prered : 0;

  // update step bound
  if (ratio < p1) {
    state.ncsuc = 0;
    state.ncfail++;
    state.delta *= p5;
  } else {
    state.ncfail = 0;
    state.ncsuc++;

    if (ratio >= p5 || state.ncsuc > 1) {
      state.delta = Kokkos::max<Real>(state.delta, pnorm / p5);
    }
    if (Kokkos::fabs(ratio - 1) <= p1) {
      state.delta = pnorm / p5;
    }
  }

  // test if iteration successful
  if (ratio >= p0001) {
    copy_vector(pars.x, state.x_trial);
    copy_vector(pars.f, state.f_trial);
    state.fnorm = fnorm1;
    state.iter++;
  }

  // determine iteration progress
  state.nslow1++;
  if (actred >= p001) {
    state.nslow1 = 0;
  }
  if (actred >= p1) {
    state.nslow2 = 0;
  }
  if (state.ncfail == 2) {
    {
      fdf(pars.x, pars.f, pars.J, state, pars);
    }

    state.nslow2++;

    if (state.iter == 1) {
      compute_diag(pars.J, state.diag);
      state.delta = compute_delta(state.diag, pars.x);
    } else {
      update_diag(pars.J, state.diag);
    }

    // QR factorization
    qr_factorize(pars.J, state.q, state.r);
    return HYBRIDSJ_SUCCESS;
  }

  compute_qtf(state.q, state.df, state.qtdf);
  compute_wv(state.qtdf, state.rdx, pars.dx, state.diag, pnorm, state.w,
             state.v);

  qr_update(state.q, state.r, state.w, state.v);

  // No progress conditions
  if (state.nslow2 == 5 || state.nslow1 == 10) {
    return HYBRIDSJ_ENOPROGJ;
  }
  return HYBRIDSJ_SUCCESS;
}

KOKKOS_INLINE_FUNCTION
HybridsjSignal HybridsjTestDelta(const Real (&dx)[M1_MULTIROOTS_DIM],
                                const Real (&x)[M1_MULTIROOTS_DIM], Real epsabs,
                                Real epsrel) {
  int ok = 1;

  if (epsrel < 0.0) {
    return HYBRIDSJ_EBADTOL;
  }

  for (int i = 0; i < M1_MULTIROOTS_DIM; i++) {
    double tolerance = epsabs + epsrel * Kokkos::fabs(x[i]);

    if (Kokkos::fabs(dx[i]) < tolerance || dx[i] == 0) {
      ok = 1;
    } else {
      ok = 0;
      break;
    }
  }

  if (ok) {
    return HYBRIDSJ_SUCCESS;
  }
  return HYBRIDSJ_CONTINUE;
}

} // namespace radiationm1
#endif // RADIATION_M1_ROOTS_HYBRIDJ_HPP
