//========================================================================================
// AthenaK astrophysical plasma code -- BSD-3-Clause (see LICENSE)
// Michel/Bondi solution, following the HSW formulation in tests/gr_bondi.cpp.
//========================================================================================
#ifndef PGEN_BONDI_PROFILE_HPP_
#define PGEN_BONDI_PROFILE_HPP_

#include <cmath>

#include "athena.hpp"

namespace bondi_profile {
struct Solution {
  Real gamma, k, rc, n, tc, c1, c2;

  void Initialize(Real gamma_in, Real k_in, Real rc_in) {
    gamma = gamma_in;
    k = k_in;
    rc = rc_in;
    n = 1.0/(gamma-1.0);
    Real uc2 = 0.5/rc;
    tc = n/(n+1.0)*uc2/(1.0-(n+3.0)*uc2);
    c1 = -pow(tc,n)*sqrt(uc2)*rc*rc;
    c2 = SQR(1.0+(n+1.0)*tc)*(1.0-1.5/rc);
  }

  KOKKOS_INLINE_FUNCTION
  Real Residual(Real t, Real r) const {
    Real ur = c1/(r*r*pow(t,n));
    return SQR(1.0+(n+1.0)*t)*(1.0-2.0/r+ur*ur)-c2;
  }

  KOKKOS_INLINE_FUNCTION
  void Sample(Real radius, Real &rho, Real &p, Real &ur, Real &ut) const {
    Real r = fmax(radius,1.0);
    Real t = tc;
    if (fabs(r-rc) > 1.0e-10*rc) {
      // Locate the residual minimum in log temperature, then select the
      // transonic branch. Log bracketing remains accurate for cold reservoirs.
      Real lo = log(1.0e-12), hi = log(100.0);
      constexpr Real f = 0.3819660112501051;
      Real a = lo+f*(hi-lo), b = hi-f*(hi-lo);
      Real fa = Residual(exp(a),r), fb = Residual(exp(b),r);
      for (int it=0; it<64; ++it) {
        if (fa < fb) {
          hi=b; b=a; fb=fa; a=lo+f*(hi-lo); fa=Residual(exp(a),r);
        } else {
          lo=a; a=b; fa=fb; b=hi-f*(hi-lo); fb=Residual(exp(b),r);
        }
      }
      Real tmin = exp(0.5*(lo+hi));
      lo = r < rc ? 1.0e-12 : tmin;
      hi = r < rc ? tmin : 100.0;
      Real flo = Residual(lo,r);
      for (int it=0; it<64; ++it) {
        t = 0.5*(lo+hi);
        Real ft = Residual(t,r);
        if ((ft > 0.0) == (flo > 0.0)) {lo=t; flo=ft;} else {hi=t;}
      }
      t = 0.5*(lo+hi);
    }
    rho = pow(t/k,n);
    p = rho*t;
    ur = c1/(r*r*pow(t,n));
    Real e = sqrt(fmax(1.0-2.0/r+ur*ur,0.0));
    // Rationalized Schwarzschild -> ingoing KS time component: regular at r=2.
    ut = (1.0+(1.0+2.0/r)*ur*ur)/(e-2.0*ur/r);
  }
};
}  // namespace bondi_profile
#endif  // PGEN_BONDI_PROFILE_HPP_
