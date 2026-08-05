#ifndef RADIATION_RADIATION_OPACITIES_HPP_
#define RADIATION_RADIATION_OPACITIES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_opacities.hpp
//! \brief implements functions for computing opacities

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OpacityFunction
//! \brief sets sigma_a, sigma_s, sigma_p in the comoving frame

KOKKOS_INLINE_FUNCTION
void OpacityFunction(// density and density scale
                     const Real dens, const Real density_scale,
                     // temperature and temperature scale
                     const Real temp, const Real temperature_scale,
                     // length scale, adiabatic index minus one, mean molecular weight
                     const Real length_scale, const Real gm1, const Real mu,
                     // power law opacities
                     const bool pow_opacity,
                     const Real rosseland_coef, const Real planck_minus_rosseland_coef,
                     // spatially and temporally constant opacities
                     const Real k_a, const Real k_s, const Real k_p,
                     // output sigma
                     Real& sigma_a, Real& sigma_s, Real& sigma_p) {
  if (pow_opacity) {  // power law opacity (accounting for diff b/w Ross & Planck)
    Real power_law = (dens*density_scale)*pow(gm1*mu/(temp*temperature_scale), 3.5);
    Real k_a_r = rosseland_coef * power_law;
    Real k_a_p = planck_minus_rosseland_coef * power_law;
    sigma_a = dens*k_a_r*density_scale*length_scale;
    sigma_p = dens*k_a_p*density_scale*length_scale;
    sigma_s = dens*k_s  *density_scale*length_scale;
  } else {  // spatially and temporally constant opacity
    sigma_a = dens*k_a*density_scale*length_scale;
    sigma_p = dens*k_p*density_scale*length_scale;
    sigma_s = dens*k_s*density_scale*length_scale;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  bool FourthPolyRoot
//  \brief Exact solution for the fourth order polynomial coef4 * x^4 + x + tconst = 0.
//  Used by the radiation/gas temperature-equilibrium updates (emission-absorption and
//  Compton) in both the intensity-based radiation module and the grey-M1 module.

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root) {
  // Calculate real root of z^3 - 4*tconst/coef4 * z - 1/coef4^2 = 0
  Real ccubic = tconst * tconst * tconst;
  Real delta1 = 0.25 - 64.0 * ccubic * coef4 / 27.0;
  if (delta1 < 0.0) {
    return false;
  }
  delta1 = sqrt(delta1);
  if (delta1 < 0.5) {
    return false;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = pow(delta1, -2.0/3.0) / 3.0;
  } else {
    zroot = pow(0.5 + delta1, 1.0/3.0) - pow(-0.5 + delta1, 1.0/3.0);
  }
  if (zroot < 0.0) {
    return false;
  }
  zroot *= pow(coef4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coef4 * rcoef);
  if (delta2 < 0.0) {
    return false;
  }
  delta2 = sqrt(delta2);
  root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return false;
  }
  return true;
}

#endif // RADIATION_RADIATION_OPACITIES_HPP_
