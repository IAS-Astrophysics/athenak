#ifndef UTILS_TOV_TOV_POLYTROPE_HPP_
#define UTILS_TOV_TOV_POLYTROPE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov_polytrope.hpp
//  \brief Fixed polytrope EOS for use with TOVStar

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tov_utils.hpp"

namespace tov {

class PolytropeEOS {
 private:
  Real kappa;
  Real gamma;

 public:
  explicit PolytropeEOS(ParameterInput* pin) {
    kappa = pin->GetReal("problem", "kappa");
    gamma = pin->GetReal("mhd", "gamma");
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    return kappa*Kokkos::pow(rho, gamma);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromE(Real e) const {
    // Note that e = rho + kappa/(gamma - 1)*rho^gamma, which doesn't usually have an
    // algebraic solution. Consequently, we choose to solve this with a Newton-Raphson
    // solve instead.
    // To ensure the solution converges, we bracket the solution from above and below by
    // recognizing that rho <= e, rho > 0
    Real lb = 0.0;
    Real ub = e;
    auto f = [&](Real rho) -> Real {
      return rho + kappa*Kokkos::pow(rho, gamma)/(gamma - 1.0) - e;
    };
    auto df = [&](Real rho) -> Real {
      return 1.0 + kappa*gamma*Kokkos::pow(rho, gamma - 1.0)/(gamma - 1.0);
    };
    Real flb = f(lb);
    Real fub = f(ub);
    Real x = (fub*lb - flb*ub)/(fub - flb);
    Real fx = f(x);
    const Real tol = 1e-15;
    while (Kokkos::fabs(fx) > e*tol) {
      Real xnew = x - fx/df(x);
      // If the guess is no good, throw it away and use bisection instead.
      if (xnew > ub || xnew < lb) {
        xnew = 0.5*(ub + lb);
      }
      fx = f(xnew);
      if (fx > 0) {
        ub = xnew;
      } else {
        lb = xnew;
      }
      x = xnew;
    }

    return x;
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    return Kokkos::pow(P/kappa, 1.0/gamma);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    return rho + kappa*Kokkos::pow(rho, gamma)/(gamma - 1.0);
  }
};

} // namespace tov

#endif // UTILS_TOV_TOV_POLYTROPE_HPP_
