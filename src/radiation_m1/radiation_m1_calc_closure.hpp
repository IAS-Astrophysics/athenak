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
    // For a fluid at rest in the Eulerian frame, J=E and H_a=F_a,
    // independently of chi. Evaluate this limit directly, including in curved
    // coordinates. Rootfinding the squared residual near vacuum otherwise
    // amplifies cancellation in the four-tensor transformations.
    if (v_d(1) == 0.0 && v_d(2) == 0.0 && v_d(3) == 0.0) {
      Real J = E;
      auto H_d = F_d;
      apply_floor(g_uu, J, H_d, m1_params);
      const Real xi = Kokkos::fmin(1.0, Kokkos::sqrt(Kokkos::fmax(
          0.0, tensor_dot(g_uu, H_d, H_d))) / J);
      chi = closure_fun(xi, closure_type);
      apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                    P_dd, m1_params);
      return;
    }
    // Newton-Raphson rootfinder (opt-in via closure_solver = newton).
    // Seeded with the relativistic-aberration guess xi0 = |F - v E| / E.
    if (m1_params.closure_solver == ClosureNewton) {
      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hg_d{};
      for (int a = 0; a < 4; ++a) {
        Hg_d(a) = F_d(a) - v_d(a) * E;
      }
      const Real xi0 =
          (E > m1_params.rad_E_floor)
              ? Kokkos::fmin(1.0, Kokkos::sqrt(Kokkos::fmax(0.0,
                                     tensor_dot(g_uu, Hg_d, Hg_d))) / E)
              : 0.5;
      const Real xi = NewtonClosure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E,
                                    F_d, m1_params, xi0);
      chi = closure_fun(xi, closure_type);
      apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                    m1_params);
      return;
    }

    // For fixed E, F and fluid/metric fields, P is affine in the closure
    // mixing weight. Transform its two endpoints once, not at every Brent
    // trial (also called repeatedly inside the implicit photon source solve).
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> pressure{};
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> stress{};
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> h_thick{}, h_thin{};
    calc_Pthick(g_dd, g_uu, n_d, w_lorentz, v_d, E, F_d, pressure);
    assemble_rT(n_d, E, F_d, pressure, stress);
    const Real j_thick = calc_J_from_rT(stress, u_u);
    calc_H_from_rT(stress, u_u, proj_ud, h_thick);
    calc_Pthin(g_uu, E, F_d, pressure);
    assemble_rT(n_d, E, F_d, pressure, stress);
    const Real j_thin = calc_J_from_rT(stress, u_u);
    calc_H_from_rT(stress, u_u, proj_ud, h_thin);
    auto residual = [&](Real xi) {
      const Real thick = 1.5 * (1.0 - closure_fun(xi, closure_type));
      const Real thin = 1.0 - thick;
      Real j = thick * j_thick + thin * j_thin;
      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> h{};
      for (int a = 0; a < 4; ++a) h(a) = thick * h_thick(a) + thin * h_thin(a);
      // Floor the trial moments, not the endpoints: flooring is nonlinear.
      apply_floor(g_uu, j, h, m1_params);
      return j*j*xi*xi - tensor_dot(g_uu, h, h);
    };

    Real x_lo = 0.0;
    Real x_md = 0.5;
    Real x_hi = 1.0;
    Real root{};
    BrentState state{};

    // Initialize rootfinder
    MathSignal ierr =
        BrentInitialize(residual, x_lo, x_hi, root, state);

    // no root, most likely due to high velocities, use simple approximation
    if (ierr == LinalgEinval) {
      const Real z_ed = residual(0.0);
      const Real z_th = residual(1.0);
      if (Kokkos::abs(z_th) < Kokkos::abs(z_ed)) {
        // Kokkos::printf("LinalgEinval: set chi = 1\n");
        chi = 1.0;
      } else {
        // Kokkos::printf("LinalgEinval: set chi = 1/3\n");
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
      ierr = BrentIterate(residual, x_lo, x_hi, root, state);

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
