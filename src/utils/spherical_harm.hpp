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

// Calculate spherical harmonics using Wigner-d matrix notation
KOKKOS_INLINE_FUNCTION
void SphericalHarm(Real *ylmR, Real *ylmI, int l, int m, Real theta, Real phi) {
  Real wignerd = 0.0;
  int k1, k2;

  k1 = Kokkos::max(0, m);
  k2 = Kokkos::min(l + m, l);

  for (int k = k1; k <= k2; ++k) {
    wignerd += Kokkos::pow(-1.0, k)
               * Kokkos::pow(Kokkos::cos(theta/2.0), 2*l + m - 2*k)
               * Kokkos::pow(Kokkos::sin(theta/2.0), 2*k - m)
               / ( fac(l + m - k) * fac(l - k) * fac(k) * fac(k - m) );
  }

  wignerd *= Kokkos::sqrt((2*l + 1)/(4*M_PI))
             * fac(l)
             * Kokkos::sqrt(fac(l + m))
             * Kokkos::sqrt(fac(l - m));

  *ylmR = wignerd * Kokkos::cos(m*phi);
  *ylmI = wignerd * Kokkos::sin(m*phi);
}

// Calculate first derivatives of spherical harmonics using Wigner-d matrix notation
KOKKOS_INLINE_FUNCTION
void SphericalHarmFirstDerivs(Real *YlmR, Real *YlmI, Real *YlmRdth, Real *YlmIdth,
                         Real *YlmRdphi, Real *YlmIdphi, int l, int m, Real theta, Real phi) {

  SphericalHarm(YlmR, YlmI, l, m, theta, phi);

  // Phi first derivative
  *YlmRdphi = -m * (*YlmI);
  *YlmIdphi =  m * (*YlmR);

  // Theta first derivative
  // Initialization is important, since for l=m modes
  // the (m < l) block is skipped and *YlmRdth/*YlmIdth subtract
  // from garbage values
  *YlmRdth = 0.0; 
  *YlmIdth = 0.0;
  Real Y_upR, Y_upI, Y_downR, Y_downI;

  if (m < l) {
      SphericalHarm(&Y_upR, &Y_upI, l, m + 1, theta, phi);
      Real coeff = 0.5 * Kokkos::sqrt((l - m)*(l + m + 1));
      *YlmRdth = coeff * (Y_upR * Kokkos::cos(phi) + Y_upI * Kokkos::sin(phi));
      *YlmIdth = coeff * (-Y_upR * Kokkos::sin(phi) + Y_upI * Kokkos::cos(phi));
  }
  if (m > -l) {
      SphericalHarm(&Y_downR, &Y_downI, l, m-1, theta, phi);
      Real coeff = 0.5 * Kokkos::sqrt((l + m)*(l - m + 1));
      *YlmRdth -= coeff * (Y_downR * Kokkos::cos(phi) - Y_downI * Kokkos::sin(phi));
      *YlmIdth -= coeff * (Y_downR * Kokkos::sin(phi) + Y_downI * Kokkos::cos(phi));
  }
}

// Calculate second derivatives of spherical harmonics using Wigner-d matrix notation
KOKKOS_INLINE_FUNCTION
void SphericalHarmSecondDerivs(Real *YlmR, Real *YlmI, Real *YlmRdth, Real *YlmIdth,
                               Real *YlmRdphi, Real *YlmIdphi, Real *YlmRdth2, 
                               Real *YlmIdth2, Real *YlmRdphi2, Real *YlmIdphi2, 
                               Real *YlmRdthdphi, Real *YlmIdthdphi, int l, int m, Real theta, Real phi) {

  SphericalHarmFirstDerivs(YlmR, YlmI, YlmRdth, YlmIdth, YlmRdphi, YlmIdphi, l, m, theta, phi);  
  
  // Phi second derivative
  *YlmRdphi2 = -Kokkos::pow(m, 2) * (*YlmR);
  *YlmIdphi2 = -Kokkos::pow(m, 2) * (*YlmI);

  // Mixed second derivative
  *YlmRdthdphi = -m * (*YlmIdth);
  *YlmIdthdphi = m * (*YlmRdth);

  // Theta second derivative
  Real sin_theta = Kokkos::sin(theta);
  Real cot_theta = Kokkos::cos(theta) / sin_theta;

  *YlmRdth2 = -l * (l+1) * (*YlmR) - cot_theta * (*YlmRdth) + Kokkos::pow(m, 2)/Kokkos::pow(sin_theta, 2) * (*YlmR);
  *YlmIdth2 = -l * (l+1) * (*YlmI) - cot_theta * (*YlmIdth) + Kokkos::pow(m, 2)/Kokkos::pow(sin_theta, 2) * (*YlmI);        
}

#endif // UTILS_SPHERICAL_HARM_HPP_
