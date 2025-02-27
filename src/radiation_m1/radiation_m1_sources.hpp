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