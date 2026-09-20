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
  if (!(Kokkos::isfinite(coef4)) || !(Kokkos::isfinite(tconst)) || coef4 < 0.0) {
    return false;
  }
  if (fabs(coef4) <= 1.0e-300) {
    root = -tconst;
    return (root >= 0.0 && Kokkos::isfinite(root));
  }

  // For coef4 >= 0, f(x)=coef4*x^4+x+tconst is monotone on x >= 0.
  // A positive root exists only when f(0) <= 0.
  if (tconst > 0.0) {
    return false;
  }

  Real lo = 0.0;
  Real hi = fmax(1.0, root);
  if (!(Kokkos::isfinite(hi)) || hi <= 0.0) {
    hi = 1.0;
  }
  bool bracketed = false;
  for (int it=0; it<128; ++it) {
    Real fhi = coef4*SQR(SQR(hi)) + hi + tconst;
    if (!(Kokkos::isfinite(fhi))) {
      return false;
    }
    if (fhi >= 0.0) {
      bracketed = true;
      break;
    }
    hi *= 2.0;
    if (!(Kokkos::isfinite(hi))) {
      return false;
    }
  }
  if (!(bracketed)) {
    return false;
  }

  Real x = fmin(fmax(root, lo), hi);
  if (x <= lo || x >= hi) {
    x = 0.5 * (lo + hi);
  }
  const Real ftol = 1.0e-13*(1.0 + fabs(tconst));
  for (int it=0; it<80; ++it) {
    const Real f = coef4*SQR(SQR(x)) + x + tconst;
    if (!(Kokkos::isfinite(f))) {
      return false;
    }
    if (fabs(f) <= ftol) {
      root = x;
      return true;
    }
    if (f > 0.0) {
      hi = x;
    } else {
      lo = x;
    }

    const Real df = 4.0*coef4*x*x*x + 1.0;
    Real xnew = x - f/df;
    if (!(Kokkos::isfinite(xnew)) || xnew <= lo || xnew >= hi) {
      xnew = 0.5*(lo + hi);
    }
    x = xnew;
  }

  root = x;
  return (root >= 0.0 && Kokkos::isfinite(root));
}

KOKKOS_INLINE_FUNCTION
bool OpacityDensityScale(const Real wdn, const Real dfloor, const Real dfloor_opacity,
                         const Real dens_trunc_max, const Real tau_truncation,
                         const Real sigmoid_residual, const Real kappa_s,
                         const Real delta_l, const Real sigma_cold,
                         const bool use_excision_density, Real &scale) {
  scale = 1.0;
  if (!(wdn > 0.0) || !(dfloor > 0.0) || !(dfloor_opacity > 0.0)) {
    return false;
  }
  if (use_excision_density) {
    scale = dfloor_opacity/wdn;
    return Kokkos::isfinite(scale);
  }
  if (!(delta_l > 0.0) || !(Kokkos::isfinite(delta_l))) {
    return false;
  }

  Real dtrunc = dfloor;
  if (kappa_s > 0.0 && tau_truncation > 0.0 && sigma_cold > 0.0) {
    dtrunc = sigma_cold*tau_truncation/(kappa_s*delta_l);
    if (!(Kokkos::isfinite(dtrunc)) || dtrunc <= 0.0) {
      return false;
    }
    dtrunc = fmin(dens_trunc_max, fmax(dfloor, dtrunc));
  }

  const Real fac_trunc = dtrunc/dfloor;
  const Real wdn_real = fmax(wdn - dfloor, dfloor_opacity);
  if (!(fac_trunc > 0.0) || !(wdn_real > 0.0) || !(Kokkos::isfinite(fac_trunc))) {
    return false;
  }

  Real wdn_opacity = wdn_real;
  if (fabs(fac_trunc - 1.0) > 1.0e-12) {
    const Real denom = log(1.0/sigmoid_residual - 1.0);
    if (!(denom > 0.0)) {
      return false;
    }
    const Real wid_trunc = 0.5*log10(fac_trunc)/denom;
    if (!(wid_trunc > 0.0) || !(Kokkos::isfinite(wid_trunc))) {
      return false;
    }
    const Real center = log10(dfloor) + 0.5*log10(fac_trunc);
    const Real fac_inv = 1.0 + exp(-(log10(wdn_real) - center)/wid_trunc);
    if (!(fac_inv > 0.0) || !(Kokkos::isfinite(fac_inv))) {
      return false;
    }
    const Real del_reduce = log10(dfloor) - log10(dfloor_opacity);
    wdn_opacity = pow(10.0, log10(wdn_real) - (1.0 - 1.0/fac_inv)*del_reduce);
  }

  scale = wdn_opacity/wdn;
  return (scale >= 0.0 && Kokkos::isfinite(scale));
}

#endif // RADIATION_RADIATION_OPACITIES_HPP_
