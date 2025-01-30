#ifndef RADIATION_M1_ROOTS_HPP
#define RADIATION_M1_ROOTS_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_roots_brent.hpp
//  \brief functions for Brent-Dekker rootfinder

#include "athena.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \struct BrentState
//  \brief container for the current state of the root finder
struct BrentState {
  Real a, b, c, d, e;
  Real f_a, f_b, f_c;
};

//----------------------------------------------------------------------------------------
//! \enum BrentSignal
//  \brief signal codes for the Brent-Dekker rootfinder
enum BrentSignal {
  BRENT_INVALID,
  BRENT_EINVAL,
  BRENT_SUCCESS,
  BRENT_CONTINUE,
};

//----------------------------------------------------------------------------------------
//! \class BrentFunctor
//  \brief Function to rootfind in order to determine the closure
class BrentFunctor {
public:
  KOKKOS_INLINE_FUNCTION
  Real operator()(
      Real xi, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
      const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
      const Real &w_lorentz,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud,
      const Real &E, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
      const RadiationM1Params &params) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                  minerbo(xi), P_dd, params);

    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd{};
    assemble_rT(n_d, E, F_d, P_dd, rT_dd);

    const Real J = calc_J_from_rT(rT_dd, u_u);

    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
    calc_H_from_rT(rT_dd, u_u, proj_ud, H_d);

    const Real H2 = tensor_dot(g_uu, H_d, H_d);
    return SQ(J * xi) - H2;
  }
};

class BrentFunctorInv {
public:
  KOKKOS_INLINE_FUNCTION
  Real
  operator()(Real dthick,
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
             const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
             const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_thick_dd,
             const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_thin_dd,
             const RadiationM1Params &params) {

    // Assemble stress energy tensor
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd{};
    assemble_rT_fluid(dthick, u_d, J, H_d, K_thick_dd, K_thin_dd, rT_dd);

    // Compute lab frame quantities for a given dthick
    const Real E = calc_J_from_rT(rT_dd, n_u);
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
    calc_H_from_rT(rT_dd, n_u, gamma_ud, F_d);

    // Transform back to fluid frame
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                  P_dd, params);
    assemble_rT(n_d, E, F_d, P_dd, rT_dd);
    const Real Jnew = calc_J_from_rT(rT_dd, u_u);

    return Jnew - J;
  }
};

//----------------------------------------------------------------------------------------
//! \fn BrentSignal radiationm1::BrentInitialize
//  \brief Initialize the Brent-Dekker rootfinder
template <class Functor, class... Types>
KOKKOS_INLINE_FUNCTION BrentSignal BrentInitialize(Functor &&f, Real x_lower,
                                                   Real x_upper, Real &root,
                                                   BrentState &brent_state,
                                                   Types... args) {
  if (x_lower > x_upper) {
    printf("BrentInit: %.14e must be less than %.14e\n", x_lower, x_upper);
    return BRENT_EINVAL;
  }

  root = 0.5 * (x_lower + x_upper);

  Real f_lower = f(x_lower, args...);
  Real f_upper = f(x_upper, args...);

  brent_state.a = x_lower;
  brent_state.b = x_upper;
  brent_state.c = x_upper;
  brent_state.d = x_upper - x_lower;
  brent_state.e = x_upper - x_lower;

  brent_state.f_a = f_lower;
  brent_state.f_b = f_upper;
  brent_state.f_c = f_upper;

  if ((f_lower < 0.0 && f_upper < 0.0) || (f_lower > 0.0 && f_upper > 0.0)) {
    printf("BrentInit: Function at endpoints should be of opposite signs! "
           "f_l = %.14e, f_h = %.14e\n",
           f_lower, f_upper);
    return BRENT_EINVAL;
  }
  return BRENT_SUCCESS;
}

//----------------------------------------------------------------------------------------
//! \fn BrentSignal radiationm1::BrentIterate
//  \brief Single iteration of the Brent-Dekker rootfinder
template <class Functor, class... Types>
KOKKOS_INLINE_FUNCTION BrentSignal BrentIterate(Functor &&f, Real &x_lower,
                                                Real &x_upper, Real &root,
                                                BrentState &brent_state,
                                                Types... args) {
  int ac_equal = 0;

  Real m{};
  Real a = brent_state.a;
  Real b = brent_state.b;
  Real c = brent_state.c;
  Real d = brent_state.d;
  Real e = brent_state.e;

  Real f_a = brent_state.f_a;
  Real f_b = brent_state.f_b;
  Real f_c = brent_state.f_c;

  if ((f_b < 0 && f_c < 0) || (f_b > 0 && f_c > 0)) {
    ac_equal = 1;
    c = a;
    f_c = f_a;
    d = b - a;
    e = b - a;
  }

  if (Kokkos::fabs(f_c) < Kokkos::fabs(f_b)) {
    ac_equal = 1;
    a = b;
    b = c;
    c = a;
    f_a = f_b;
    f_b = f_c;
    f_c = f_a;
  }

  Real tol = 0.5 * DBL_EPSILON * fabs(b);
  m = 0.5 * (c - b);

  if (f_b == 0) {
    root = b;
    x_lower = b;
    x_upper = b;
    return BRENT_SUCCESS;
  }

  if (Kokkos::fabs(m) <= tol) {
    root = b;

    if (b < c) {
      x_lower = b;
      x_upper = c;
    } else {
      x_lower = c;
      x_upper = b;
    }
    return BRENT_SUCCESS;
  }

  if (Kokkos::fabs(e) < tol || Kokkos::fabs(f_a) <= Kokkos::fabs(f_b)) {
    // bisection
    d = m;
    e = m;
  } else {
    // inverse cubic interpolation
    double p{}, q{}, r{};
    double s = f_b / f_a;

    if (ac_equal) {
      p = 2 * m * s;
      q = 1 - s;
    } else {
      q = f_a / f_c;
      r = f_b / f_c;
      p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
      q = (q - 1) * (r - 1) * (s - 1);
    }

    if (p > 0) {
      q = -q;
    } else {
      p = -p;
    }

    if (2 * p <
        Kokkos::min(3 * m * q - Kokkos::fabs(tol * q), Kokkos::fabs(e * q))) {
      e = d;
      d = p / q;
    } else {
      // fall back to bisection
      d = m;
      e = m;
    }
  }

  a = b;
  f_a = f_b;

  if (Kokkos::fabs(d) > tol) {
    b += d;
  } else {
    b += (m > 0 ? +tol : -tol);
  }

  f_b = f(b, args...);

  brent_state.a = a;
  brent_state.b = b;
  brent_state.c = c;
  brent_state.d = d;
  brent_state.e = e;
  brent_state.f_a = f_a;
  brent_state.f_b = f_b;
  brent_state.f_c = f_c;

  // update best estimate of root and upper and lower bounds

  root = b;

  if ((f_b < 0 && f_c < 0) || (f_b > 0 && f_c > 0)) {
    c = a;
  }

  if (b < c) {
    x_lower = b;
    x_upper = c;
  } else {
    x_lower = c;
    x_upper = b;
  }
  return BRENT_SUCCESS;
}

//----------------------------------------------------------------------------------------
//! \fn BrentSignal radiationm1::BrentTestInterval
//  \brief test convergence of interval to check if BrentIterate should continue
KOKKOS_INLINE_FUNCTION
BrentSignal BrentTestInterval(Real x_lower, Real x_upper, Real epsabs,
                              Real epsrel) {

  const Real abs_lower = Kokkos::fabs(x_lower);
  const Real abs_upper = Kokkos::fabs(x_upper);

  Real min_abs{}, tolerance{};

  if (epsrel < 0.0) {
    return BRENT_INVALID;
  }
  if (epsabs < 0.0) {
    return BRENT_INVALID;
  }
  if (x_lower > x_upper) {
    return BRENT_INVALID;
  }

  if ((x_lower > 0.0 && x_upper > 0.0) || (x_lower < 0.0 && x_upper < 0.0)) {
    min_abs = Kokkos::min(abs_lower, abs_upper);
  } else {
    min_abs = 0;
  }

  tolerance = epsabs + epsrel * min_abs;

  if (Kokkos::fabs(x_upper - x_lower) < tolerance) {
    return BRENT_SUCCESS;
  }
  return BRENT_CONTINUE;
}

} // namespace radiationm1
#endif // RADIATION_M1_ROOTS_HPP
