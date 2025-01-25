#ifndef RADIATION_M1_ROOTS_HPP
#define RADIATION_M1_ROOTS_HPP

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "radiation_m1/radiation_m1_closure.hpp"

namespace radiationm1 {

struct BrentFunctionParams {
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu;
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d;
  Real w_lorentz;
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d;
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> proj_ud;
  Real E;
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d;
  RadiationM1Params params;
};

struct BrentFunction {
  BrentFunctionParams p;
  Real M1ClosureBrentFunc(Real xi, BrentFunctionParams &p) {

    Real chi = minerbo(xi);

    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
    apply_closure(p.g_dd, p.g_uu, p.n_d, p.w_lorentz, p.u_u, p.v_d, p.proj_ud,
                  p.E, p.F_d, chi, P_dd, p.params);

    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd{};
    assemble_rT(p.n_d, p.E, p.F_d, P_dd, rT_dd);

    const Real J = calc_J_from_rT(rT_dd, p.u_u);

    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
    calc_H_from_rT(rT_dd, p.u_u, p.proj_ud, H_d);

    const Real H2 = tensor_dot(p.g_uu, H_d, H_d);
    return SQ(J * xi) - H2;
  }
};

//----------------------------------------------------------------------------------------
//! \struct BrentState
//  \brief container for the current state of the root finder
struct BrentState {
  Real a, b, c, d, e;
  Real f_a, f_b, f_c;
};

//----------------------------------------------------------------------------------------
//! \enum BrentSignal
//  \brief success/failure codes of the root finder
enum BrentSignal {
  BRENT_EINVAL,
  BRENT_SUCCESS,
};

// initialize the Brent-Dekker solver
KOKKOS_INLINE_FUNCTION
BrentSignal BrentInitialize(BrentState &brent_state, BrentFunction &brent_func,
                            Real &root, Real x_lower, Real x_upper) {

  root = 0.5 * (x_lower + x_upper);

  Real f_lower = brent_func.M1ClosureBrentFunc(x_lower, brent_func.p);
  Real f_upper = brent_func.M1ClosureBrentFunc(x_upper, brent_func.p);

  brent_state.a = x_lower;
  brent_state.b = x_upper;
  brent_state.c = x_upper;
  brent_state.d = x_upper - x_lower;
  brent_state.e = x_upper - x_lower;

  brent_state.f_a = f_lower;
  brent_state.f_b = f_upper;
  brent_state.f_c = f_upper;

  if ((f_lower < 0.0 && f_upper < 0.0) || (f_lower > 0.0 && f_upper > 0.0)) {
    printf("Endpoints should be of opposite signs!\n");
    return BRENT_EINVAL;
  }
  return BRENT_SUCCESS;
}

// iterate the Brent-Dekker solver
KOKKOS_INLINE_FUNCTION
BrentSignal BrentIterate(BrentState &brent_state, BrentFunction &brent_func,
                         Real &root, Real &x_lower, Real &x_upper) {

  int ac_equal = 0;

  Real tol{}, m{};
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

  const Real GSL_DBL_EPSILON = 1e-16;
  tol = 0.5 * GSL_DBL_EPSILON * Kokkos::fabs(b);
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
    d = m; /* use bisection */
    e = m;
  } else {
    double p{}, q{}, r{}; /* use inverse cubic interpolation */
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
      /* interpolation failed, fall back to bisection */

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

  f_b = brent_func.M1ClosureBrentFunc(b, brent_func.p);

  brent_state.a = a;
  brent_state.b = b;
  brent_state.c = c;
  brent_state.d = d;
  brent_state.e = e;
  brent_state.f_a = f_a;
  brent_state.f_b = f_b;
  brent_state.f_c = f_c;

  /* Update the best estimate of the root and bounds on each
     iteration */

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

} // namespace radiationm1
#endif // RADIATION_M1_ROOTS_HPP
