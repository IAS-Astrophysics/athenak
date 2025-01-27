#ifndef RADIATION_M1_CALC_CLOSURE_HPP
#define RADIATION_M1_CALC_CLOSURE_HPP
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_closure.hpp
//! \brief function which finds closure using Brent-Dekker routines

#include "radiation_m1/radiation_m1.hpp"

namespace radiationm1 {

// Computes the closure in the lab frame with a rootfinding procedure
KOKKOS_INLINE_FUNCTION
void RadiationM1::calc_closure(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const Real &w_lorentz,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud,
    const Real &E, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
    Real &chi, AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
    const RadiationM1Params &params) {

  // Special cases for which no rootfinding needed
  if (params.closure_fun == Eddington) {
    chi = 1. / 3.;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                  P_dd, params);
    return;
  }
  if (params.closure_fun == Thin) {
    chi = 1.0;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                  P_dd, params);
    return;
  }
  if (params.closure_fun == Minerbo) {
    Real x_lo = 0.;
    Real x_md = 0.5;
    Real x_hi = 1.;
    Real root{};
    BrentState state{};

    // Initialize rootfinder
    int closure_maxiter = 100;
    Real closure_epsilon = 1e-6;
    Real tol = 1e-10;
    BrentSignal ierr =
        BrentInitialize(BrentFunc, x_lo, x_hi, root, state, tol, g_dd, g_uu,
                        n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, params);

    if (ierr == BRENT_EINVAL) {
      double const z_ed = BrentFunc(0., g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                                    proj_ud, E, F_d, params);
      double const z_th = BrentFunc(1., g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                                    proj_ud, E, F_d, params);
      if (Kokkos::abs(z_th) < Kokkos::abs(z_ed)) {
        chi = 1.0;
      } else {
        chi = 1. / 3.;
      }
      apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                    P_dd, params);
      return;
    }
    if (ierr != BRENT_SUCCESS) {
      printf("Unexpected error in BrentInitialize.\n");
      exit(EXIT_FAILURE);
    }

    // Rootfinding
    int iter = 0;
    do {
      ++iter;
      ierr = BrentIterate(BrentFunc, x_lo, x_hi, root, state, tol, g_dd, g_uu,
                          n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, params);

      // Some nans in the evaluation. This should not happen.
      if (ierr != BRENT_SUCCESS) {
        printf("Unexpected error in BrentIterate.\n");
        exit(EXIT_FAILURE);
      }
      x_md = root;
      ierr = BrentTestInterval(x_lo, x_hi, closure_epsilon, 0);
    } while (ierr == BRENT_CONTINUE && iter < closure_maxiter);

    chi = minerbo(x_md);

    if (ierr != BRENT_SUCCESS) {
      printf("Maximum number of iterations exceeded when computing the M1 "
             "closure\n");
    }
  }
}

} // namespace radiationm1
// namespace radiationm1
#endif // RADIATION_M1_CALC_CLOSURE_HPP
