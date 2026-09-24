#ifndef UTILS_TOV_TOV_PIECEWISE_POLY_HPP_
#define UTILS_TOV_TOV_PIECEWISE_POLY_HPP_
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
#include "eos/primitive-solver/piecewise_polytrope.hpp"

namespace tov {

class PiecewisePolytropeEOS: public Primitive::PiecewisePolytrope {
 public:
  explicit PiecewisePolytropeEOS(ParameterInput *pin) {
    ReadParametersFromInput("mhd", pin);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    Real nb = rho/mb;
    int p = FindPiece(nb);
    return GetColdPressure(nb, p);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromE(Real e) const {
    // Unfortunately, e(rho) cannot be inverted simply. Therefore, we use a Newton-Raphson
    // solver to get this instead.
    Real lb = 0.0;
    Real ub = e;
    auto f = [&](Real rho) -> Real {
      Real nb = rho/mb;
      int p = FindPiece(nb);
      return GetColdEnergy(nb, p) - e;
    };
    auto df = [&](Real rho) -> Real {
      Real nb = rho/mb;
      int p = FindPiece(nb);
      return 1.0 + eps_pieces[p] +
             gamma_pieces[p]*GetColdPressure(nb, p)/(rho*gamma_pieces[p] - 1.0);
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
    Real rhob = GetDensityFromColdPressure(P);
    return rhob;
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    Real nb = rho/mb;
    int p = FindPiece(nb);
    return GetColdEnergy(nb, p);
  }
};

} // namespace tov

#endif  // UTILS_TOV_TOV_PIECEWISE_POLY_HPP_
