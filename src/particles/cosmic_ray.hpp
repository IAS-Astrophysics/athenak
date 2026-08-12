#ifndef PARTICLES_COSMIC_RAY_HPP_
#define PARTICLES_COSMIC_RAY_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cosmic_ray.hpp
//! \brief particle array indices for cosmic-ray particles

#include "athena.hpp"

namespace particles {
namespace cosmic_ray {

enum RealIndex {IPVX=NPARTICLE_REAL_COMMON, IPVY, IPVZ, NREAL};
constexpr int NINT = NPARTICLE_INT_COMMON;

} // namespace cosmic_ray
} // namespace particles

#endif // PARTICLES_COSMIC_RAY_HPP_
