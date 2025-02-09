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
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_calc_closure.hpp"
#include "radiation_m1_compute_opacities.hpp"
#include "radiation_m1_helpers.hpp"
#include "radiation_m1_sources.hpp"
#include "z4c/z4c.hpp"

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
Real RadiationM1::Toy_eta_0(const Real x1, const Real x2, const Real x3, const int index) {

  Real const R = 1.0;  // Sphere radius
  if (x1 * x1 + x2 * x2 + x3 * x3 <= R*R) { // Inside the sphere
      return 10.0;
  } else {
      return 0;  // Outside the sphere
  }  
}

KOKKOS_INLINE_FUNCTION
Real RadiationM1::Toy_abs_0(const Real x1, const Real x2, const Real x3, const int index) {

  Real const R = 1.0;  // Sphere radius
  if (x1 * x1 + x2 * x2 + x3 * x3 <= R*R) { // Inside the sphere
      return 10.0;
  } else {
      return 0;  // Outside the sphere
  }  
}

KOKKOS_INLINE_FUNCTION
Real RadiationM1::Toy_eta_1(const Real x1, const Real x2, const Real x3, const int index) {

  Real const R = 1.0;  // Sphere radius
  if (x1 * x1 + x2 * x2 + x3 * x3 <= R*R) { // Inside the sphere
      return 10.0;
  } else {
      return 0;  // Outside the sphere
  }  
}

KOKKOS_INLINE_FUNCTION
Real RadiationM1::Toy_abs_1(const Real x1, const Real x2, const Real x3, const int index) {

  Real const R = 1.0;  // Sphere radius
  if (x1 * x1 + x2 * x2 + x3 * x3 <= R*R) { // Inside the sphere
      return 10.0;
  } else {
      return 0;  // Outside the sphere
  }  
}

KOKKOS_INLINE_FUNCTION
Real RadiationM1::Toy_scat_1(const Real x1, const Real x2, const Real x3, const int index) {
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
M1Opacities ComputeM1Opacities(const Real x1, const Real x2, const Real x3,
                               const RadiationM1Params &params) {
  M1Opacities opacities{};

  switch (params.opacity_type) {
  case RadiationM1OpacityType::Toy:
    for (int nuidx = 0; nuidx < M1_TOTAL_NUM_SPECIES; nuidx++) {
      opacities.eta_0[nuidx] = RadiationM1::Toy_eta_0(x1, x2, x3, nuidx);
      opacities.abs_0[nuidx] = RadiationM1::Toy_abs_0(x1, x2, x3, nuidx);
      opacities.eta_1[nuidx] = RadiationM1::Toy_eta_1(x1, x2, x3, nuidx);
      opacities.abs_1[nuidx] = RadiationM1::Toy_abs_1(x1, x2, x3, nuidx);
      opacities.scat_1[nuidx] = RadiationM1::Toy_scat_1(x1, x2, x3, nuidx);
    }
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
