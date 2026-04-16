#ifndef UTILS_SPHERICAL_HARM_HPP_
#define UTILS_SPHERICAL_HARM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_harm.hpp
//! \brief spin-weighted spherical harmonics

#include <cmath>
#include <iostream>
#include <list>
#include <algorithm>

#include "athena.hpp"


KOKKOS_INLINE_FUNCTION
double fac(int i) {
  double result = 1;
  if (i>0) {
    while (i>0) {
      result*=i;
      i-=1;
    }
  }
  return(result);
}

//Calculate spin-weighted spherical harmonics using Wigner-d matrix notation
// see e.g. Eq II.7, II.8 in 0709.0093
KOKKOS_INLINE_FUNCTION
void SWSphericalHarm(Real * ylmR, Real * ylmI, int l, int m, int s,
                     Real theta, Real phi) {
  Real wignerd = 0;
  int k1,k2,k;
  k1 = Kokkos::max(0, m-s);
  k2 = Kokkos::min(l+m,l-s);

  for (k = k1; k<=k2; ++k) {
    wignerd += Kokkos::pow((-1),k)*Kokkos::pow(Kokkos::cos(theta/2.0),2*l+m-s-2*k)
                *Kokkos::pow(Kokkos::sin(theta/2.0),2*k+s-m)
                /(fac(l+m-k)*fac(l-s-k)*fac(k)*fac(k+s-m));
  }
  wignerd *= Kokkos::pow((-1),s)*Kokkos::sqrt((2*l+1)/(4*M_PI))*Kokkos::sqrt(fac(l+m))
              *Kokkos::sqrt(fac(l-m))
              *Kokkos::sqrt(fac(l+s))*Kokkos::sqrt(fac(l-s));
  *ylmR = wignerd*Kokkos::cos(m*phi);
  *ylmI = wignerd*Kokkos::sin(m*phi);
}

#endif // UTILS_SPHERICAL_HARM_HPP_
