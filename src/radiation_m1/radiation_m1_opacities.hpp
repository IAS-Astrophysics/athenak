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

KOKKOS_INLINE_FUNCTION
void ComputeToyOpacities(const Real x, const Real y, const Real z,
                         const Real nuidx, Real &eta_0, Real &abs_0,
                         Real &eta_1, Real &abs_1, Real &scat_1) {
  eta_0 = 0;
  abs_0 = 0;
  eta_1 = 0;
  abs_1 = 0;
  scat_1 = 0;
}

}  // namespace radiationm1
#endif  // RADIATION_M1_COMPUTE_OPACITIES_HPP
