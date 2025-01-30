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
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud,
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
    Real x_lo = 0.0;
    Real x_md = 0.5;
    Real x_hi = 1.0;
    Real root{};
    BrentState state{};

    // Initialize rootfinder
    int closure_maxiter = 64;
    Real closure_epsilon = 1e-15;
    BrentSignal ierr =
        BrentInitialize(BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d,
                        w_lorentz, u_u, v_d, proj_ud, E, F_d, params);

    // no root, most likely due to high velocities, use simple approximation
    if (ierr == BRENT_EINVAL) {
      const Real z_ed = BrentFunc(0., g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                                  proj_ud, E, F_d, params);
      const Real z_th = BrentFunc(1., g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
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

    // Rootfinding
    int iter = 0;
    do {
      ++iter;
      ierr = BrentIterate(BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d,
                          w_lorentz, u_u, v_d, proj_ud, E, F_d, params);

      // Some nans in the evaluation. This should not happen.
      if (ierr != BRENT_SUCCESS) {
        printf("Unexpected error in BrentIterate.\n");
        exit(EXIT_FAILURE);
      }
      x_md = root;
      ierr = BrentTestInterval(x_lo, x_hi, closure_epsilon, 0);
    } while (ierr == BRENT_CONTINUE && iter < closure_maxiter);

    chi = minerbo(x_md);
    printf("root = %.14e, chi = %.14e\n", x_md, chi);

    if (ierr != BRENT_SUCCESS) {
      printf("Maximum number of iterations exceeded when computing the M1 "
             "closure\n");
    }
  }
}

// compute the inverse closure
KOKKOS_INLINE_FUNCTION
void RadiationM1::calc_inv_closure(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const Real &w_lorentz,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud,
    const Real &chi, const Real &J,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d, Real &E,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
    const RadiationM1Params &params) {

  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> K_thick_dd{};
  calc_Kthick(g_dd, u_d, J, H_d, K_thick_dd);

  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> K_thin_dd{};
  calc_Kthin(g_uu, n_d, w_lorentz, u_d, proj_ud, J, H_d, K_thin_dd,
             params.rad_E_floor);

  Real x_lo = 0.0;
  Real x_md = 0.5;
  Real x_hi = 1.0;
  Real root{};
  BrentState state{};

  // Initialize rootfinder
  int inv_closure_maxiter = 64;
  Real inv_closure_epsilon = 1e-15;
  BrentSignal ierr =
      BrentInitialize(BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d,
                      w_lorentz, u_u, v_d, proj_ud, E, F_d, params);

  // no root, most likely due to truncation errors
  if (ierr == BRENT_EINVAL) {
    x_md = 3. * (1. - chi) / 2.;
    apply_inv_closure(x_md, n_u, gamma_ud, u_d, J, H_d, K_thick_dd, K_thin_dd,
                      E, F_d);
    return;
  }

  // Rootfinding
  int iter = 0;
  do {
    ++iter;
    ierr = BrentIterate(BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d,
                        w_lorentz, u_u, v_d, proj_ud, E, F_d, params);

    // Some nans in the evaluation. This should not happen.
    if (ierr != BRENT_SUCCESS) {
      printf("Unexpected error in BrentIterate.\n");
      exit(EXIT_FAILURE);
    }
    x_md = root;
    ierr = BrentTestInterval(x_lo, x_hi, inv_closure_epsilon, 0);
  } while (ierr == BRENT_CONTINUE && iter < inv_closure_maxiter);

  apply_inv_closure(x_md, n_u, gamma_ud, u_d, J, H_d, K_thick_dd, K_thin_dd, E,
                    F_d);
}
} // namespace radiationm1
#endif // RADIATION_M1_CALC_CLOSURE_HPP
