#include <cmath>
#include <iostream>
#include <list>
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

//Calculate spin-weighted spherical harmonics using Wigner-d matrix notation see e.g. Eq II.7, II.8 in 0709.0093
KOKKOS_INLINE_FUNCTION
std::pair<double,double> SWSphericalHarm(int l, int m, int s, Real theta, Real phi) {
  Real wignerd = 0;
  int k1,k2,k;
  k1 = std::max(0, m-s);
  k2 = std::min(l+m,l-s);

  for (k = k1; k<=k2; ++k) {
    wignerd += pow((-1),k)*pow(cos(theta/2.0),2*l+m-s-2*k)*pow(sin(theta/2.0),2*k+s-m)/(fac(l+m-k)*fac(l-s-k)*fac(k)*fac(k+s-m));
  }
  wignerd *= pow((-1),s)*sqrt((2*l+1)/(4*M_PI))*sqrt(fac(l+m))*sqrt(fac(l-m))*sqrt(fac(l+s))*sqrt(fac(l-s));
  return std::make_pair(wignerd*cos(m*phi), wignerd*sin(m*phi));
}

// theta derivative of the s=0 spherical harmonics
KOKKOS_INLINE_FUNCTION
std::pair<double,double> SphericalHarm_dtheta(int l, int m, Real theta, Real phi) {
  std::pair<double,double> value;
  if (l==m) {
    std::pair<double,double> value2 = SWSphericalHarm(l,m,0,theta,phi);
    value.first = m/tan(theta)*value2.first; 
    value.second = m/tan(theta)*value2.second; 
  } else {
    std::pair<double,double> value2 = SWSphericalHarm(l,m,0,theta,phi);
    std::pair<double,double> value3 = SWSphericalHarm(l,m+1,0,theta,phi);
    value.first = m/tan(theta)*value2.first + sqrt((l-m)*(l+m+1))*( cos(phi)*value3.first - sin(-phi)*value3.second);
    value.second = m/tan(theta)*value2.second + sqrt((l-m)*(l+m+1))*( cos(phi)*value3.second + sin(-phi)*value3.first);
  }
  return value;
}

// phi derivative of the s=0 spherical harmonics

KOKKOS_INLINE_FUNCTION
std::pair<double,double> SphericalHarm_dphi(int l, int m, Real theta, Real phi) {
  std::pair<double,double> value;
  std::pair<double,double> value2 = SWSphericalHarm(l,m,0,theta,phi);
  value.first = -m*value2.second;
  value.second = m*value2.first;
  return value;
}

KOKKOS_INLINE_FUNCTION
Real RealSphericalHarm(int l, int m, Real theta, Real phi) {
  double value;
  if (m==0) {
    value = SWSphericalHarm(l,m,0,theta,phi).first;
  } else if (m>0) {
    value = sqrt(2)*pow((-1),m)*SWSphericalHarm(l,m,0,theta,phi).first;
  } else {
    value = sqrt(2)*pow((-1),(-m))*SWSphericalHarm(l,m,0,theta,phi).second;
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real RealSphericalHarm_dtheta(int l, int m, Real theta, Real phi) {
  double value;
  if (m==0) {
    value = SphericalHarm_dtheta(l,m,theta,phi).first;
  } else if (m>0) {
    value = sqrt(2)*pow((-1),m)*SphericalHarm_dtheta(l,m,theta,phi).first;
  } else {
    value = sqrt(2)*pow((-1),(-m))*SphericalHarm_dtheta(l,m,theta,phi).second;
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real RealSphericalHarm_dphi(int l, int m, Real theta, Real phi) {
  double value;
  if (m==0) {
    value = SphericalHarm_dphi(l,m,theta,phi).first;
  } else if (m>0) {
    value = sqrt(2)*pow((-1),m)*SphericalHarm_dphi(l,m,theta,phi).first;
  } else {
    value = sqrt(2)*pow((-1),(-m))*SphericalHarm_dphi(l,m,theta,phi).second;
  }
  return value;
}
