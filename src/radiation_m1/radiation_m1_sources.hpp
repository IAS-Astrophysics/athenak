#ifndef RADIATION_M1_SOURCES_HPP
#define RADIATION_M1_SOURCES_HPP

#include <athena_tensor.hpp>

#include "athena.hpp"
#include "radiation_m1_params.hpp"
#include "radiation_m1_roots_brent.hpp"
#include "radiation_m1_roots_hybridsj.hpp"

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_macro.hpp
//  \brief macros for Grey M1 radiation class

namespace radiationm1 {

struct SrcParams {
  SrcParams(const Real _cdt, const Real _alp,
            const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &_g_dd,
            const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &_g_uu,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_n_d,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_n_u,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &_gamma_ud,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_u_d,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_u_u,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_v_d,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_v_u,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &_proj_ud,
            const Real _W, const Real Estar,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d,
            const Real _chi, const Real _eta, const Real _kabs, const Real _kscat)
      : cdt(_cdt),
        alp(_alp),
        g_dd(_g_dd),
        g_uu(_g_uu),
        n_d(_n_d),
        n_u(_n_u),
        gamma_ud(_gamma_ud),
        u_d(_u_d),
        u_u(_u_u),
        v_d(_v_d),
        v_u(_v_u),
        proj_ud(_proj_ud),
        W(_W),
        Estar(Estar),
        Fstar_d(Fstar_d),
        chi(_chi),
        eta(_eta),
        kabs(_kabs),
        kscat(_kscat) {}
  const Real cdt;
  const Real alp;
  const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd;
  const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud;
  const Real W;
  const Real Estar;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d;
  Real chi;
  const Real eta;
  const Real kabs;
  const Real kscat;

  Real E{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
  Real J{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> S_d{};
  Real Edot{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> tS_d{};
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

HybridsjSignal RadiationM1::prepare_closure(const Real q[4], SrcParams &p,
                                            const RadiationM1Params &params) {
  p.E = Kokkos::max(q[0], 0.);
  if (p.E < 0) {
    return HYBRIDSJ_EBADFUNC;
  }
  pack_F_d(-p.alp * p.n_u(1), -p.alp * p.n_u(2), -p.alp * p.n_u(3), q[1], q[2], q[3],
           p.F_d);

  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u{};
  tensor_contract(p.g_uu, p.F_d, F_u);

  calc_closure(p.g_dd, p.g_uu, p.n_d, p.W, p.u_u, p.v_d, p.proj_ud, p.E, p.F_d, p.chi,
               p.P_dd, params);

  return HYBRIDSJ_SUCCESS;
}

HybridsjSignal RadiationM1::prepare_sources(const Real q[4], SrcParams &p) {
  assemble_rT(p.n_d, p.E, p.F_d, p.P_dd, p.T_dd);

  p.J = calc_J_from_rT(p.T_dd, p.u_u);
  calc_H_from_rT(p.T_dd, p.u_u, p.proj_ud, p.H_d);

  calc_rad_sources(p.eta, p.kabs, p.kscat, p.u_d, p.J, p.H_d, p.S_d);

  p.Edot = calc_rE_source(p.alp, p.n_u, p.S_d);
  calc_rF_source(p.alp, p.gamma_ud, p.S_d, p.tS_d);

  return HYBRIDSJ_SUCCESS;
}

HybridsjSignal RadiationM1::prepare(const Real q[4], SrcParams &p,
                                    const RadiationM1Params &params) {
  auto ierr = prepare_closure(q, p, params);
  if (ierr != HYBRIDSJ_SUCCESS) {
    return ierr;
  }

  ierr = prepare_sources(q, p);
  if (ierr != HYBRIDSJ_SUCCESS) {
    return ierr;
  }
  return HYBRIDSJ_SUCCESS;
}

KOKKOS_INLINE_FUNCTION
void explicit_update(SrcParams &p, Real &Enew,
                     AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d) {
  Enew = p.Estar + p.cdt * p.Edot;
  Fnew_d(1) = p.Fstar_d(1) + p.cdt * p.tS_d(1);
  Fnew_d(2) = p.Fstar_d(2) + p.cdt * p.tS_d(2);
  Fnew_d(3) = p.Fstar_d(3) + p.cdt * p.tS_d(3);
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  Fnew_d(0) = -p.alp * p.n_u(1) * Fnew_d(1) - p.alp * p.n_u(2) * Fnew_d(2) -
              p.alp * p.n_u(3) * Fnew_d(3);
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

// Solves the implicit problem
// .  q^new = q^star + dt S[q^new]
// The source term is S^a = (eta - ka J) u^a - (ka + ks) H^a and includes
// also emission.
SrcSignal RadiationM1::source_update(
    const Real &cdt, const Real &alp,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &W,
    const Real &Eold, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fold_d,
    const Real &Estar, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d,
    const Real &eta, const Real &kabs, const Real &kscat, Real &chi, Real &Enew,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d,
    const RadiationM1Params &params_) {
  SrcParams p(cdt, alp, g_dd, g_uu, n_d, n_u, gamma_ud, u_d, u_u, v_d, v_u, proj_ud, W,
              Estar, Fstar_d, chi, eta, kabs, kscat);
  // Old solution
  Real qold[] = {Eold, Fold_d(1), Fold_d(2), Fold_d(3)};
  Real xold[4];
  for (int i = 0; i < 4; i++) {
    xold[i] = qold[i];
  }

  // Non stiff limit, use explicit update
  if (cdt * kabs < 1 && cdt * kscat < 1) {
    prepare(xold, p, params_);
    explicit_update(p, Enew, Fnew_d);

    Real q[4] = {Enew, Fnew_d(1), Fnew_d(2), Fnew_d(3)};
    Real x[4];
    for (int i = 0; i < 4; i++) {
      x[i] = q[i];
    }

    prepare_closure(x, p, params_);
    chi = p.chi;

    return SrcThin;
  }

  // Our scheme cannot capture this dynamics (tau << dt), so we go
  // directly to the equilibrium
  if (params_.source_thick_limit > 0 &&
      SQ(cdt) * (kabs * (kabs + kscat)) > SQ(params_.source_thick_limit)) {
    return SrcEquil;
  }

  // This handles the scattering dominated limit
  if (params_.source_scat_limit > 0 && cdt * kscat > params_.source_scat_limit) {
    return SrcScat;
  }

  // Initial guess for the solution
  HybridsjState state{};
  HybridsjParams pars{};
  Real q[4] = {Enew, Fnew_d(1), Fnew_d(2), Fnew_d(3)};
  Real x[4];
  for (int i = 0; i < 4; i++) {
    x[i] = q[i];
    pars.x[i] = x[i];
  }

  int ierr = HybridsjInitialize(HybridsjFunc, state, pars);
  int iter = 0;
  /*
  do {
    ierr = radiationm1::HybridsjIterate(func, state, pars);
    print_f(pars.x, pars.f, pars.J);
    iter++;

    ierr = radiationm1::HybridsjTestDelta(pars.dx, pars.x, epsabs, epsrel);
  } while (ierr == radiationm1::HYBRIDSJ_CONTINUE && iter < maxiter); */

  do {
    if (iter < params_.source_maxiter) {
      ierr = HybridsjIterate();
      iter++;
    }
    // The nonlinear solver is stuck.
    if (ierr == HYBRIDSJ_ENOPROGJ || ierr == HYBRIDSJ_EBADFUNC || iter >= params_.source_maxiter) {
      // If we are here, then we are in trouble

      // We are optically thick, suggest to retry with Eddington closure
      if (closure_fun != eddington) {
#ifdef WARN_FOR_SRC_FIX
        ss << "Eddington closure\n";
        print_stuff(cctkGH, i, j, k, ig, &p, ss);
        Printer::print_warn(ss.str());
#endif
        ierr = source_update(cctkGH, i, j, k, ig, eddington, gsl_solver_1d, gsl_solver_nd,
                             cdt, alp, g_dd, g_uu, n_d, n_u, gamma_ud, u_d, u_u, v_d, v_u,
                             proj_ud, W, Eold, Fold_d, Estar, Fstar_d, eta, kabs, kscat,
                             chi, Enew, Fnew_d);
        if (ierr == THC_M1_SOURCE_OK) {
          return THC_M1_SOURCE_EDDINGTON;
        } else {
          return ierr;
        }
      } else {
#ifdef WARN_FOR_SRC_FIX
        ss << "using initial guess\n";
        print_stuff(cctkGH, i, j, k, ig, &p, ss);
        Printer::print_warn(ss.str());
#endif
        return THC_M1_SOURCE_FAIL;
      }
    } else if (ierr != GSL_SUCCESS) {
      char msg[BUFSIZ];
      snprintf(msg, BUFSIZ,
               "Unexpected error in "
               "gsl_multirootroot_fdfsolver_iterate, error code \"%d\"",
               ierr);
#pragma omp critical
      CCTK_ERROR(msg);
    }
    ierr = gsl_multiroot_test_delta(gsl_solver_nd->dx, gsl_solver_nd->x, source_epsabs,
                                    source_epsrel);
  } while (ierr == GSL_CONTINUE);

  *Enew = gsl_vector_get(gsl_solver_nd->x, 0);
  Fnew_d->at(1) = gsl_vector_get(gsl_solver_nd->x, 1);
  Fnew_d->at(2) = gsl_vector_get(gsl_solver_nd->x, 2);
  Fnew_d->at(3) = gsl_vector_get(gsl_solver_nd->x, 3);
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  Fnew_d->at(0) = -alp * n_u(1) * Fnew_d->at(1) - alp * n_u(2) * Fnew_d->at(2) -
                  alp * n_u(3) * Fnew_d->at(3);

  prepare_closure(gsl_solver_nd->x, &p);
  *chi = p.chi;

  return SrcOk;
}
}  // namespace radiationm1
#endif  // RADIATION_M1_SOURCES_HPP