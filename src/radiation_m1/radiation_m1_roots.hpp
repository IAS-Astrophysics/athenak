#ifndef RADIATION_M1_ROOTS_HPP
#define RADIATION_M1_ROOTS_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_roots.hpp
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
//! \class BrentDekkerRoot
//  \brief find the root of a function using the Brent-Dekker method
class BrentDekkerRoot {
public:
  Real tol;

  BrentDekkerRoot() : tol(1e-14) {}
  //~BrentDekkerRoot();

  //! \brief Initialize the Brent-Dekker solver
  // \param  f        functor whose root is to be found. Need at least one arg
  // \param  x_lower  lower bound for root
  // \param  x_upper  upper bound for root
  // \param  args     additional arguments for f
  template <class Functor, class... Types>
  KOKKOS_INLINE_FUNCTION BrentSignal BrentInitialize(Functor &&f, Real x_lower,
                                                     Real x_upper, Real &root,
                                                     BrentState &brent_state,
                                                     Types... args) {
    if (x_lower > x_upper) {
      return BRENT_INVALID;
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
      printf("Endpoints should be of opposite signs!\n");
      return BRENT_EINVAL;
    }
    return BRENT_SUCCESS;
  }

  //! \brief Iterate the Brent-Dekker solver once
  // \param  f        functor whose root is to be found. Need at least one arg
  // \param  x_lower  lower bound for root
  // \param  x_upper  upper bound for root
  // \param  args     additional arguments for f
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
};

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
    return BRENT_CONTINUE;
  }
  return BRENT_CONTINUE;
}

} // namespace radiationm1
#endif // RADIATION_M1_ROOTS_HPP
