#ifndef RADIATION_M1_PHOTON_OPACITIES_HPP
#define RADIATION_M1_PHOTON_OPACITIES_HPP

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_photon_opacities.hpp
//  \brief structs for various photon opacity params

#include "athena.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \struct PhotonOpacityParams
//  \brief parameters for the photon opacities, mirrors parameters from radiation.hpp
struct PhotonOpacityParams {
  Real arad;              // radiation constant
  Real kappa_a;           // constant Rosseland mean absoprtion coefficient
  Real kappa_s;           // constant scattering coefficient
  Real kappa_p;           // Planck - Rosseland mean coefficient
  bool is_power_opacity;  // flag to enable Kramer's law opacity for kappa_a
  bool is_compton;        // flag to enable/disable compton
};

}  // namespace radiationm1
#endif  // RADIATION_M1_PHOTON_OPACITIES_HPP
