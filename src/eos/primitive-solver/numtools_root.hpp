#ifndef EOS_PRIMITIVE_SOLVER_NUMTOOLS_ROOT_HPP_
#define EOS_PRIMITIVE_SOLVER_NUMTOOLS_ROOT_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file numtools_root.hpp
//  \author Jacob Fields
//
//  \brief Declares some functions for root-finding.

#include <math.h>
#include "ps_types.hpp"

namespace NumTools {

class Root {
 public:
  /// Maximum number of iterations
  unsigned int iterations;

  Root() : iterations(30) {}

  // FalsePosition {{{

  //! \brief Find the root of a functor f using false position.
  //
  // Find the root of a generic functor taking at least one argument. The first
  // argument is assumed to be the quantity of interest. All other arguments are
  // assumed to be constant parameters for the function. The root-finding method
  // is the Anderson-Bjorck variant of false position.
  //
  // \param[in]  f  The functor to find a root for. Its root function must take at
  //                least one argument.
  // \param[in,out]  lb  The lower bound for the root.
  // \param[in,out]  ub  The upper bound for the root.
  // \param[out]  x  The location of the root.
  // \param[in]  args  Additional arguments required by f.

  template<class Functor, class ... Types>
  KOKKOS_INLINE_FUNCTION
  bool FalsePosition(Functor&& f, Real &lb, Real &ub, Real& x, Real tol,
                     Types ... args) const {
    int side = 0;
    Real ftest;
    unsigned int count = 0;
    //last_count = 0;
    // Get our initial bracket.
    Real flb = f(lb, args...);
    Real fub = f(ub, args...);
    Real xold;
    x = lb;
    // If one of the bounds is already within tolerance of the root, we have the root.
    if (fabs(flb)/lb <= tol) {
      x = lb;
      return true;
    } else if (fabs(fub)/ub <= tol) {
      x = ub;
      return true;
    }
    if (flb*fub > 0) {
      return false;
    }
    do {
      xold = x;
      // Calculate the new root position.
      x = (fub*lb - flb*ub)/(fub - flb);
      count++;
      // Calculate f at the prospective root.
      ftest = f(x,args...);
      if (fabs((x-xold)/x) <= tol) {
        return true;
      }
      // Check the sign of f. If f is on the same side as the lower bound, then we adjust
      // the lower bound. Similarly, if f is on the same side as the upper bound, we
      // adjust the upper bound. If ftest falls on the same side twice, we weight one of
      // the sides to force the new root to fall on the other side. This allows us to
      // whittle down both sides at once and get better average convergence.
      if (ftest*flb >= 0) {
        if (side == 1) {
          Real m = 1. - ftest/flb;
          fub = (m > 0) ? fub*m : 0.5*fub;
          //fub /= 2.0;
        }
        flb = ftest;
        lb = x;
        side = 1;
      } else {
        if (side == -1) {
          Real m = 1. - ftest/fub;
          flb = (m > 0) ? flb*m : 0.5*flb;
          //flb /= 2.0;
        }
        fub = ftest;
        ub = x;
        side = -1;
      }
    } while (count < iterations);
    //last_count = count;

    // Return success if we're below the tolerance, otherwise report failure.
    return fabs((x-xold)/x) <= tol;
  }

  // }}}

  // Chandrupatla {{{

  //! \brief Find the root of a functor f using Chandrupatla's method
  //
  // Find the root of a generic functor taking at least one argument. The first
  // argument is assumed to be the quantity of interest. All other arguments are
  // assumed to be constant parameters for the function. The root-finding method
  // is Chandrupatla's method, a simpler alternative to Brent's method with
  // comparable performance.
  //
  // \param[in]  f  The functor to find a root for. Its root function must take at
  //                least one argument.
  // \param[in,out]  lb  The lower bound for the root.
  // \param[in,out]  ub  The upper bound for the root.
  // \param[out]  x  The location of the root.
  // \param[in]  args  Additional arguments required by f.

  template<class Functor, class ... Types>
  KOKKOS_INLINE_FUNCTION
  bool Chandrupatla(Functor&& f, Real &lb, Real &ub, Real& x, Real tol,
                    Types ... args) const {
    unsigned int count = 0;
    //last_count = 0;
    // Get our initial bracket.
    Real flb = f(lb, args...);
    Real fub = f(ub, args...);
    x = lb;
    // If one of the bounds is already within tolerance of the root, we have the root.
    if (fabs(flb) <= tol) {
      x = lb;
      return true;
    } else if (fabs(fub) <= tol) {
      x = ub;
      return true;
    }
    // Make sure the bracket is valid
    if (flb*fub > 0) {
      return false;
    }
    Real t = 0.5;
    Real x1, x2, x3, f1, f2, f3, ftest;
    Real phi1, xi1;
    x1 = ub;
    x2 = lb;
    f1 = fub;
    f2 = flb;
    do {
      // Estimate the new root position
      x = x1 + t*(x2 - x1);
      count++;
      // Calculate f at the prospective root
      ftest = f(x, args...);
      if (fabs((x-x1)/x) <= tol) {
        break;
      }
      // Check the sign of ftest to determine the new bounds
      if (ftest*f1 >= 0) {
        x3 = x1;
        x1 = x;
        f3 = f1;
        f1 = ftest;
      } else {
        x3 = x2;
        x2 = x1;
        x1 = x;
        f3 = f2;
        f2 = f1;
        f1 = ftest;
      }
      // Check if we're in the region of validity for quadratic interpolation.
      phi1 = (f1 - f2)/(f3 - f2);
      xi1 = (x1 - x2)/(x3 - x2);
      if (1.0 - sqrt(1.0 - xi1) < phi1 && phi1 < sqrt(xi1)) {
        // Perform quadratic interpolation
        t = f1/(f3 - f2)*(f3/(f1 - f2) + (x3 - x1)/(x2 - x1)*f2/(f3 - f1));
      } else {
        // Perform bisection instead
        t = 0.5;
      }
    } while (count < iterations);
    //last_count = count;

    // Return success if we're below the tolerance, otherwise report failure.
    return fabs((x-x1)/x) <= tol;
  }

  // }}}

  // NewtonSafe {{{
  /*! \brief Find the root of a function f using a safe Newton solve.
   *
   * A safe Newton solve performs a Newton-Raphson solve, but it also brackets the
   * root using bisection to ensure that the result converges.
   *
   * \param[in]     f     The functor to find a root for. Its root function must take
   *                      at least one argument.
   * \param[in,out] lb    The lower bound for the root of f.
   * \param[in,out] ub    The upper bound for the root of f.
   * \param[out]    x     The root of f.
   * \param[in]     args  Additional arguments required by f.
   */
  template<class Functor, class ... Types>
  KOKKOS_INLINE_FUNCTION
  bool NewtonSafe(Functor&& f, Real &lb, Real &ub, Real& x, Real tol,
                  Types ... args) const {
    Real fx;
    Real dfx;
    Real xold;
    unsigned int count = 0;
    //last_count = 0;
    // We first need to ensure that the bracket is valid.
    Real fub, flb;
    f(flb, dfx, lb, args...);
    f(fub, dfx, ub, args...);
    if (flb*fub > 0) {
      return 0;
    }
    // If one of the roots is already within tolerance, then
    // we don't need to do the solve.
    if (fabs(flb) <= tol) {
      x = lb;
      return true;
    } else if (fabs(fub) <= tol) {
      x = ub;
      return true;
    }
    // Since we already had to evaluate the function at the bounds,
    // we can predict our starting position using false position.
    x = (fub*lb - flb*ub)/(fub - flb);
    do {
      xold = x;
      // Calculate f and df at point x.
      f(fx, dfx, x, args...);
      // Correct the bounds.
      if (fx*flb > 0) {
        flb = fx;
        lb = xold;
      } else if (fx*fub > 0) {
        fub = fx;
        ub = xold;
      }
      // Update the root.
      x = x - fx/dfx;
      // Check that the root is bounded properly.
      if (x > ub || x < lb) {
        // Revert to bisection if the root is not converging.
        x = 0.5*(ub + lb);
        //f(fx, dfx, x, args...);
      }
      count++;
    } while (fabs((xold-x)/x) > tol && count < iterations);
    //last_count = count;

    // Return success if we're below the tolerance, otherwise report failure.
    return fabs((x-xold)/x) <= tol;
  }
  // }}}
};

} // namespace NumTools

#endif  // EOS_PRIMITIVE_SOLVER_NUMTOOLS_ROOT_HPP_
