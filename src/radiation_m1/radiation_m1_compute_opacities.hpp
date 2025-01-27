#ifndef RADIATION_M1_COMPUTE_OPACITIES_HPP
#define RADIATION_M1_COMPUTE_OPACITIES_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_compute_opacities.hpp
//  \brief compute toy opacities/Weakrates/bns_nurates

#include "athena.hpp"
#include "radiation_m1/radiation_m1.hpp"

namespace radiationm1 {

struct M1Opacities {

  Real eta_0[M1_TOTAL_NUM_SPECIES];
  Real abs_0[M1_TOTAL_NUM_SPECIES];

  Real eta_1[M1_TOTAL_NUM_SPECIES];
  Real abs_1[M1_TOTAL_NUM_SPECIES];
  Real scat_1[M1_TOTAL_NUM_SPECIES];
};
typedef struct M1Opacities M1Opacities;

KOKKOS_INLINE_FUNCTION
M1Opacities ComputeM1Opacities(const RadiationM1Params &params) {
  M1Opacities opacities{};

  switch (params.opacity_type) {
  case RadiationM1OpacityType::Toy:
    break;
  case RadiationM1OpacityType::Weakrates:
    break;
  case RadiationM1OpacityType::BnsNurates:
    break;
  default:
    break;
  }
  return opacities;
}

} // namespace radiationm1
#endif // RADIATION_M1_COMPUTE_OPACITIES_HPP
