#ifndef RADIATION_M1_SOURCES_HPP
#define RADIATION_M1_SOURCES_HPP

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_roots_hybridsj.hpp"

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_macro.hpp
//  \brief macros for Grey M1 radiation class

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::RadiationM1::prepare_closure
//  \brief Sets F_d, F_u, E and computes chi, P_dd in src_params
KOKKOS_INLINE_FUNCTION
HybridsjSignal prepare_closure(const BrentFunctor &BrentFunc, const Real q[4],
                               SrcParams &src_params, const RadiationM1Params &m1_params,
                               const RadiationM1Closure &closure_type) {
  src_params.E = Kokkos::max(q[0], 0.);
  if (src_params.E < 0) {
    return HYBRIDSJ_EBADFUNC;
  }
  pack_F_d(-src_params.alp * src_params.n_u(1), -src_params.alp * src_params.n_u(2),
           -src_params.alp * src_params.n_u(3), q[1], q[2], q[3], src_params.F_d);

  tensor_contract(src_params.g_uu, src_params.F_d, src_params.F_u);

  calc_closure(BrentFunc, src_params.g_dd, src_params.g_uu, src_params.n_d, src_params.W,
               src_params.u_u, src_params.v_d, src_params.proj_ud, src_params.E,
               src_params.F_d, src_params.chi, src_params.P_dd, m1_params, closure_type);
  return HYBRIDSJ_SUCCESS;
}

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::RadiationM1::prepare_sources
//  \brief Sets T_dd, J, H_d, S_d, Edot and tS_d in src_params
KOKKOS_INLINE_FUNCTION
HybridsjSignal prepare_sources(const Real q[4], SrcParams &src_params) {
  assemble_rT(src_params.n_d, src_params.E, src_params.F_d, src_params.P_dd,
              src_params.T_dd);

  src_params.J = calc_J_from_rT(src_params.T_dd, src_params.u_u);
  calc_H_from_rT(src_params.T_dd, src_params.u_u, src_params.proj_ud, src_params.H_d);

  calc_rad_sources(src_params.eta, src_params.kabs, src_params.kscat, src_params.u_d,
                   src_params.J, src_params.H_d, src_params.S_d);

  src_params.Edot = calc_rE_source(src_params.alp, src_params.n_u, src_params.S_d);
  calc_rF_source(src_params.alp, src_params.gamma_ud, src_params.S_d, src_params.tS_d);

  return HYBRIDSJ_SUCCESS;
}

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::RadiationM1::prepare
//  \brief Calls prepare_closure and prepare_sources
KOKKOS_INLINE_FUNCTION
HybridsjSignal prepare(const BrentFunctor &BrentFunc, const Real q[4],
                       SrcParams &src_params, const RadiationM1Params &m1_params,
                       const RadiationM1Closure &closure_type) {
  auto ierr = prepare_closure(BrentFunc, q, src_params, m1_params, closure_type);
  if (ierr != HYBRIDSJ_SUCCESS) {
    return ierr;
  }
  ierr = prepare_sources(q, src_params);
  if (ierr != HYBRIDSJ_SUCCESS) {
    return ierr;
  }
  return HYBRIDSJ_SUCCESS;
}

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::explicit_update
//  \brief Computes (E/F)_new = (E/F)* + cdt * tS_d[E/F]
KOKKOS_INLINE_FUNCTION
void explicit_update(const SrcParams &src_params, Real &Enew,
                     AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d) {
  Enew = src_params.Estar + src_params.cdt * src_params.Edot;
  Fnew_d(1) = src_params.Fstar_d(1) + src_params.cdt * src_params.tS_d(1);
  Fnew_d(2) = src_params.Fstar_d(2) + src_params.cdt * src_params.tS_d(2);
  Fnew_d(3) = src_params.Fstar_d(3) + src_params.cdt * src_params.tS_d(3);
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  Fnew_d(0) = -src_params.alp * src_params.n_u(1) * Fnew_d(1) -
              src_params.alp * src_params.n_u(2) * Fnew_d(2) -
              src_params.alp * src_params.n_u(3) * Fnew_d(3);
}

// Solves the implicit problem
// .  q^new = q^star + dt S[q^new]
// The source term is S^a = (eta - ka J) u^a - (ka + ks) H^a and includes
// also emission.
KOKKOS_INLINE_FUNCTION
SrcSignal source_update(
    const BrentFunctor &BrentFunc, const HybridsjFunctor &HybridsjFunc, const Real &cdt,
    const Real &alp, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
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
    const RadiationM1Params &m1_params, const RadiationM1Closure &closure_type) {
  SrcParams src_params(cdt, alp, g_dd, g_uu, n_d, n_u, gamma_ud, u_d, u_u, v_d, v_u,
                       proj_ud, W, Estar, Fstar_d, chi, eta, kabs, kscat);
  // old solution
  Real qold[] = {Eold, Fold_d(1), Fold_d(2), Fold_d(3)};
  Real xold[4];
  for (int i = 0; i < 4; i++) {
    xold[i] = qold[i];
  }

  // non stiff limit, explicit update
  if (cdt * kabs < 1 && cdt * kscat < 1) {
    prepare(BrentFunc, xold, src_params, m1_params, closure_type);
    explicit_update(src_params, Enew, Fnew_d);

    Real q[4] = {Enew, Fnew_d(1), Fnew_d(2), Fnew_d(3)};
    Real x[4];
    for (int i = 0; i < 4; i++) {
      x[i] = q[i];
    }

    prepare_closure(BrentFunc, x, src_params, m1_params, closure_type);
    chi = src_params.chi;
    return SrcThin;
  }

  // cannot capture case tau << dt, go to equilibrium
  if (m1_params.source_thick_limit > 0 &&
      SQ(cdt) * (kabs * (kabs + kscat)) > SQ(m1_params.source_thick_limit)) {
    return SrcEquil;
  }

  // scattering dominated limit
  if (m1_params.source_scat_limit > 0 && cdt * kscat > m1_params.source_scat_limit) {
    return SrcScat;
  }

  // begin multiroots solve
  HybridsjState hybridsj_state{};
  HybridsjParams hybridsj_params{};

  // initial guess
  Real q[4] = {Enew, Fnew_d(1), Fnew_d(2), Fnew_d(3)};
  Real x[4];
  for (int i = 0; i < 4; i++) {
    x[i] = q[i];
    hybridsj_params.x[i] = x[i];
  }
  auto ierr =
      HybridsjInitialize(HybridsjFunc, hybridsj_state, hybridsj_params, BrentFunc, src_params, m1_params);

  int iter = 0;
  do {
    if (iter < m1_params.source_maxiter) {
      ierr = HybridsjIterate(HybridsjFunc, hybridsj_state, hybridsj_params, BrentFunc, src_params, m1_params);
      iter++;
    }

    // the solver is stuck!
    if (ierr == HYBRIDSJ_ENOPROGJ || ierr == HYBRIDSJ_EBADFUNC ||
        iter >= m1_params.source_maxiter) {
      if (m1_params.closure_type != Eddington) {
        // Eddington closure
        auto signal = source_update(BrentFunc, HybridsjFunc, cdt, alp, g_dd, g_uu, n_d,
                                    n_u, gamma_ud, u_d, u_u, v_d, v_u, proj_ud, W, Eold,
                                    Fold_d, Estar, Fstar_d, eta, kabs, kscat, chi, Enew,
                                    Fnew_d, m1_params, Eddington);
        if (signal == SrcOk) {
          return SrcEddington;
        } else {
          return signal;
        }
      }
    } else {
      // solver has failed
      return SrcFail;
    }
    ierr = HybridsjTestDelta(hybridsj_params.dx, hybridsj_params.x,
                             m1_params.source_epsabs, m1_params.source_epsrel);
  } while (ierr == HYBRIDSJ_CONTINUE);

  Enew = hybridsj_params.x[0];
  Fnew_d(1) = hybridsj_params.x[1];
  Fnew_d(2) = hybridsj_params.x[2];
  Fnew_d(3) = hybridsj_params.x[3];
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  Fnew_d(0) =
      -alp * n_u(1) * Fnew_d(1) - alp * n_u(2) * Fnew_d(2) - alp * n_u(3) * Fnew_d(3);

  prepare_closure(BrentFunc, hybridsj_params.x, src_params, m1_params, closure_type);
  chi = src_params.chi;

  return SrcOk;
}
}  // namespace radiationm1
#endif  // RADIATION_M1_SOURCES_HPP