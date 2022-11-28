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
//! \brief sets sigma_a, sigma_s in the comoving frame

KOKKOS_INLINE_FUNCTION
void OpacityFunction(const Real dens, const Real density_scale,
                     const Real temp, const Real temperature_scale,
                     const Real length_scale,
                     const bool pow_opacity, const Real kramers_const,
                     const Real k_a, const Real k_s,
                     Real& sigma_a, Real& sigma_s) {
  if (pow_opacity) {  // Kramer's law opacity
    Real kramer = kramers_const*(dens*density_scale)*pow(temp*temperature_scale, -3.5);
    sigma_a = dens*kramer*density_scale*length_scale;
    sigma_s = dens*k_s*density_scale*length_scale;
  } else {  // spatially and temporally constant opacity
    sigma_a = dens*k_a*density_scale*length_scale;
    sigma_s = dens*k_s*density_scale*length_scale;
  }
  return;
}

#endif // RADIATION_RADIATION_OPACITIES_HPP_
