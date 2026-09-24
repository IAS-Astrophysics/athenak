#ifndef RADIATION_M1_ROOTS_FNS_H
#define RADIATION_M1_ROOTS_FNS_H

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "radiation_m1/radiation_m1_closure.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1_roots_hybridsj.hpp"

namespace radiationm1 {
class BrentFunctor;

KOKKOS_IMPL_FUNCTION
MathSignal prepare(const BrentFunctor &BrentFunc, const Real q[4], SrcParams &src_params,
                   const RadiationM1Params &m1_params,
                   const RadiationM1Closure &closure_type);
//----------------------------------------------------------------------------------------
//! \class BrentFunctor
//  \brief Function to rootfind in order to determine the closure
class BrentFunctor {
 public:
  KOKKOS_INLINE_FUNCTION
  Real operator()(Real xi, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
                  const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
                  const Real &w_lorentz,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud,
                  const Real &E,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                  const RadiationM1Params &params,
                  const RadiationM1Closure &closure_type) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                  closure_fun(xi, closure_type), P_dd, params);

    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd{};
    assemble_rT(n_d, E, F_d, P_dd, rT_dd);

    Real J = calc_J_from_rT(rT_dd, u_u);
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
    calc_H_from_rT(rT_dd, u_u, proj_ud, H_d);
    apply_floor(g_uu, J, H_d, params);

    const Real H2 = tensor_dot(g_uu, H_d, H_d);
    return J * J * xi * xi - H2;
  }
};

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::dclosure_dxi
//  \brief derivative of the Minerbo Eddington factor chi(xi) w.r.t. xi
KOKKOS_INLINE_FUNCTION Real dclosure_dxi(const Real xi) {
  return xi * (12.0 - 6.0 * xi + 24.0 * xi * xi) / 15.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::NewtonClosure
//  \brief Pure Newton-Raphson rootfinder for the Minerbo closure.  Solves
//  f(xi) = J(xi)^2 xi^2 - H(xi)^2 = 0 on xi in [0,1] using the analytic
//  derivative df/dxi.  Seeded with the relativistic-aberration guess
//  xi0 = |F - v E| / E supplied by the caller.
//
//  closure_newton_bracket = true (default, recommended): a bracket [x_lo,x_hi]
//    with f(x_lo) <= 0 <= f(x_hi) is maintained and the step is the Newton point
//    only if it stays inside the bracket AND is reducing |f| fast enough
//    (|2 f| <= |dx_prev f'|), else a bisection (rtsafe; Numerical Recipes).  The
//    bracket guarantees convergence to the PHYSICAL root.  This is essential
//    because the Minerbo residual is bimodal in the optically-thick-but-advected
//    regime (the aberration guess can land at xi ~ 1 while the physical root is
//    xi -> 0); pure Newton there converges to the spurious streaming root and
//    corrupts the closure in the core.
//  closure_newton_bracket = false: pure Newton, clipped to [0,1] (faster, but
//    only safe where the guess sits in the physical root's basin).
//
//  f(0) = -H(0)^2 <= 0 always.  In bracket mode, if f(1) < 0 there is no sign
//  change in [0,1] -> fall back to the endpoint with the smaller |f|; if a
//  bracket exists but Newton has not converged at maxiter, return the bracket
//  midpoint (a bounded estimate, never an endpoint snap).  In pure mode a
//  non-converged solve falls back to the nearer endpoint.
//  Pthin/Pthick (and dP = Pthin - Pthick) are xi-independent and hoisted.
KOKKOS_INLINE_FUNCTION Real NewtonClosure(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d, const Real &w_lorentz,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &E,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
    const RadiationM1Params &params, const Real xi0) {
  BrentFunctor BrentFunc{};
  const bool use_bracket = params.closure_newton_bracket;

  // bracket check: need f(1) >= 0 (f(0) <= 0 always) for a sign change on [0,1].
  if (use_bracket) {
    const Real f_thin = BrentFunc(1.0, g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E,
                                  F_d, params, Minerbo);
    if (!(f_thin > 0.0)) {
      const Real f_thick = BrentFunc(0.0, g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
                                     E, F_d, params, Minerbo);
      return (Kokkos::abs(f_thin) < Kokkos::abs(f_thick)) ? 1.0 : 0.0;
    }
  }

  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pthin_dd{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pthick_dd{};
  calc_Pthin(g_uu, E, F_d, Pthin_dd);
  calc_Pthick(g_dd, g_uu, n_d, w_lorentz, v_d, E, F_d, Pthick_dd);
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> dP_dthin{};
  for (int a = 0; a < 4; ++a)
    for (int b = a; b < 4; ++b)
      dP_dthin(a, b) = Pthin_dd(a, b) - Pthick_dd(a, b);

  Real x_lo = 0.0;  // f(x_lo) <= 0 invariant (bracket mode)
  Real x_hi = 1.0;  // f(x_hi) >= 0 invariant (bracket mode)
  Real xi = Kokkos::fmin(1.0, Kokkos::fmax(0.0, xi0));
  Real dx_prev = 1.0;  // magnitude of last step (rtsafe safety check)
  bool converged = false;
  for (int iter = 0; iter < params.closure_maxiter; ++iter) {
    const Real chi = closure_fun(xi, Minerbo);
    const Real dthick = 1.5 * (1.0 - chi);
    const Real dthin = 1.0 - dthick;
    const Real ddthin = 1.5 * dclosure_dxi(xi);  // d(dthin)/dxi

    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
    for (int a = 0; a < 4; ++a)
      for (int b = a; b < 4; ++b)
        P_dd(a, b) = dthick * Pthick_dd(a, b) + dthin * Pthin_dd(a, b);

    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd{};
    assemble_rT(n_d, E, F_d, P_dd, rT_dd);
    Real J = calc_J_from_rT(rT_dd, u_u);
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
    calc_H_from_rT(rT_dd, u_u, proj_ud, H_d);
    apply_floor(g_uu, J, H_d, params);

    const Real H2 = tensor_dot(g_uu, H_d, H_d);
    const Real fval = J * J * xi * xi - H2;

    if (use_bracket) {
      x_lo = (fval < 0.0) ? xi : x_lo;
      x_hi = (fval >= 0.0) ? xi : x_hi;
    }

    // analytic df/dxi (Pthin/Pthick enter linearly through dthin).
    // H^2 = g^{ab} H_a H_b  =>  d(H^2)/d(dthin) = 2 g^{ab} H_a dH_b; the full
    // dH/d(dthin) vector is assembled first so the metric contraction is correct
    // for non-diagonal g_uu (curved space).
    const Real dJ_dthin = tensor_dot(dP_dthin, u_u, u_u);
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> dH_dthin_d{};
    for (int a = 0; a < 4; ++a) {
      Real dH_a = 0.0;
      for (int b = 0; b < 4; ++b)
        for (int c = 0; c < 4; ++c)
          dH_a -= proj_ud(b, a) * u_u(c) * dP_dthin(b, c);
      dH_dthin_d(a) = dH_a;
    }
    const Real H_dot_dH_dthin = tensor_dot(g_uu, H_d, dH_dthin_d);
    const Real dfval = 2.0 * J * J * xi +
                       (2.0 * J * dJ_dthin * xi * xi - 2.0 * H_dot_dH_dthin) * ddthin;

    const Real newton_xi = (Kokkos::abs(dfval) > 0.0) ? xi - fval / dfval : xi;
    Real xi_new;
    if (use_bracket) {
      // rtsafe: Newton if inside bracket and converging fast enough, else bisect.
      const Real lo = Kokkos::fmin(x_lo, x_hi);
      const Real hi = Kokkos::fmax(x_lo, x_hi);
      const Real bisect_xi = 0.5 * (x_lo + x_hi);
      const bool newton_ok = (newton_xi > lo) && (newton_xi < hi) &&
                             (2.0 * Kokkos::abs(fval) <= Kokkos::abs(dx_prev * dfval));
      xi_new = newton_ok ? newton_xi : bisect_xi;
    } else {
      xi_new = Kokkos::fmin(1.0, Kokkos::fmax(0.0, newton_xi));
    }
    dx_prev = Kokkos::abs(xi_new - xi);

    const bool bracket_conv =
        use_bracket && (Kokkos::abs(x_hi - x_lo) < params.closure_epsilon);
    if (dx_prev < params.closure_epsilon || bracket_conv) {
      xi = xi_new;
      converged = true;
      break;
    }
    xi = xi_new;
  }

  if (!converged) {
    if (use_bracket) {
      // A bracket exists (checked up front) and has narrowed over the iterations;
      // the best estimate is its midpoint.  Do NOT snap to an endpoint here --
      // that would discard a good estimate and can pick the wrong limit.
      xi = 0.5 * (x_lo + x_hi);
    } else {
      // pure Newton did not converge: fall back to the endpoint with smaller |f|.
      const Real f_thick = BrentFunc(0.0, g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
                                     E, F_d, params, Minerbo);
      const Real f_thin = BrentFunc(1.0, g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
                                    E, F_d, params, Minerbo);
      xi = (Kokkos::abs(f_thin) < Kokkos::abs(f_thick)) ? 1.0 : 0.0;
    }
  }
  return xi;
}

KOKKOS_INLINE_FUNCTION Real set_dthin(Real chi) { return 1.5 * chi - 0.5; }
KOKKOS_INLINE_FUNCTION Real set_dthick(Real chi) { return 1.5 * (1 - chi); }

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::source_jacobian
//  \brief low level kernel computing the Jacobian matrix
KOKKOS_INLINE_FUNCTION void source_jacobian(
    const Real qpre[4], AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_u, Real &F2,
    const Real &chi, const Real &kapa, const Real &kaps,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d, const Real &v2,
    const Real &W, const Real &alpha, const Real &cdt, const Real qstar[4],
    Real J[4][4]) {
  const Real kapas = kapa + kaps;
  const Real alpW = alpha * W;

  const Real dthin = set_dthin(chi);
  const Real dthick = set_dthick(chi);

  const Real vx = v_d(1);
  const Real vy = v_d(2);
  const Real vz = v_d(3);
  const Real W2 = W * W;
  const Real W3 = W2 * W;

  const Real vdotF = F_u(1) * v_d(1) + F_u(2) * v_d(2) + F_u(3) * v_d(3);
  const Real normF = Kokkos::sqrt(F2);
  const Real inormF = (normF > 0 ? 1 / normF : 0);
  const Real vdothatf = vdotF * inormF;
  const Real vdothatf2 = vdothatf * vdothatf;
  const Real hatfx = qpre[1] * inormF;  // hatf_i
  const Real hatfy = qpre[2] * inormF;
  const Real hatfz = qpre[3] * inormF;
  const Real hatfupx = F_u(1) * inormF;  // hatf^i
  const Real hatfupy = F_u(2) * inormF;
  const Real hatfupz = F_u(3) * inormF;
  const Real e = qpre[0];
  const Real eonormF = Kokkos::fmin(e * inormF, 1.0);  // with factor dthin

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

  Real Jyx = -alpha * (kapas * HydFx + W * kapa * vy * JdFx);
  Real Jyy = -alpha * (kapas * HydFy + W * kapa * vy * JdFy);
  Real Jyz = -alpha * (kapas * HydFz + W * kapa * vy * JdFz);

  Real Jzx = -alpha * (kapas * HzdFx + W * kapa * vz * JdFx);
  Real Jzy = -alpha * (kapas * HzdFy + W * kapa * vz * JdFy);
  Real Jzz = -alpha * (kapas * HzdFz + W * kapa * vz * JdFz);

  // Store Jacobian into J
  J[0][0] = 1 - cdt * J00;
  J[0][1] = -cdt * J0x;
  J[0][2] = -cdt * J0y;
  J[0][3] = -cdt * J0z;
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
  void operator()(const Real x[M1_MULTIROOTS_DIM], Real f[M1_MULTIROOTS_DIM],
                  Real J[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM], BrentFunctor BrentFunc,
                  SrcParams &src_params, RadiationM1Params &m1_params) const {
    // Function to rootfind for
    //    f(q) = q - q^* - dt S[q]
    auto ierr = prepare(BrentFunc, x, src_params, m1_params, m1_params.closure_type);
    f[0] = x[0] - src_params.Estar - src_params.cdt * src_params.Edot;
    f[1] = x[1] - src_params.Fstar_d(1) - src_params.cdt * src_params.tS_d(1);
    f[2] = x[2] - src_params.Fstar_d(2) - src_params.cdt * src_params.tS_d(2);
    f[3] = x[3] - src_params.Fstar_d(3) - src_params.cdt * src_params.tS_d(3);

    Real m_q[4] = {src_params.E, src_params.F_d(1), src_params.F_d(2), src_params.F_d(3)};
    Real m_F2 = tensor_dot(src_params.F_u, src_params.F_d);
    Real m_v2 = tensor_dot(src_params.v_u, src_params.v_d);
    Real m_qstar[4] = {src_params.Estar, src_params.Fstar_d(1), src_params.Fstar_d(2),
                      src_params.Fstar_d(3)};

    source_jacobian(m_q, src_params.F_u, m_F2, src_params.chi, src_params.kabs,
                    src_params.kscat, src_params.v_u, src_params.v_d, m_v2, src_params.W,
                    src_params.alp, src_params.cdt, m_qstar, J);
  }
};

}  // namespace radiationm1

#endif  // RADIATION_M1_ROOTS_FNS_H
