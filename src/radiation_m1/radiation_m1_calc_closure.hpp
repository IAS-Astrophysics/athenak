#ifndef RADIATION_M1_CALC_CLOSURE_HPP
#define RADIATION_M1_CALC_CLOSURE_HPP
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_closure.hpp
//! \brief compute closure and inv. closure using Brent-Dekker routines

#include "radiation_m1/radiation_m1_roots_brent.hpp"
#include "radiation_m1/radiation_m1_roots_fns.hpp"

namespace radiationm1 {

// Computes the closure in the lab frame with a rootfinding procedure
KOKKOS_INLINE_FUNCTION void calc_closure(
    BrentFunctor BrentFunc, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d, const Real &w_lorentz,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &E,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d, Real &chi,
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
    const RadiationM1Params &m1_params, const RadiationM1Closure &closure_type) {
  // Special cases for which no rootfinding needed
  if (closure_type == Eddington) {
    chi = 1. / 3.;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                  m1_params);
    return;
  }
  if (closure_type == Thin) {
    chi = 1.0;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                  m1_params);
    return;
  }
  if (closure_type == Minerbo) {
    Real x_lo = 0.0;
    Real x_md = 0.5;
    Real x_hi = 1.0;
    Real root{};
    BrentState state{};

    // Initialize rootfinder
    MathSignal ierr =
        BrentInitialize(BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d, w_lorentz,
                        u_u, v_d, proj_ud, E, F_d, m1_params, m1_params.closure_type);

    // no root, most likely due to high velocities, use simple approximation
    if (ierr == LinalgEinval) {
      const Real z_ed = BrentFunc(0., g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E,
                                  F_d, m1_params, closure_type);
      const Real z_th = BrentFunc(1., g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E,
                                  F_d, m1_params, closure_type);
      if (Kokkos::abs(z_th) < Kokkos::abs(z_ed)) {
        Kokkos::printf("LinalgEinval: set chi = 1\n");
        chi = 1.0;
      } else {
        Kokkos::printf("LinalgEinval: set chi = 1/3\n");
        chi = 1. / 3.;
      }
      apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                    m1_params);
      return;
    }

    // Rootfinding
    int iter = 0;
    do {
      ++iter;
      ierr = BrentIterate(BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d, w_lorentz,
                          u_u, v_d, proj_ud, E, F_d, m1_params, m1_params.closure_type);

      // Some nans in the evaluation. This should not happen.
      if (ierr != LinalgSuccess) {
        Kokkos::printf("Unexpected error in BrentIterate.\n");
      }
      x_md = root;
      ierr = BrentTestInterval(x_lo, x_hi, m1_params.closure_epsilon, 0);
    } while (ierr == LinalgContinue && iter < m1_params.closure_maxiter);

    chi = closure_fun(x_md, closure_type);
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                    m1_params);
    if (ierr != LinalgSuccess) {
      Kokkos::printf(
          "Maximum number of iterations exceeded when computing the M1 "
          "closure\n");
    }
  }
}

}  // namespace radiationm1
#endif  // RADIATION_M1_CALC_CLOSURE_HPP
