#ifndef RADIATION_M1_SETMASK_HPP
#define RADIATION_M1_SETMASK_HPP

#include "radiation_m1.hpp"

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_setmask.hpp
//  \brief function to set mask
//
//  NOTE: the radiation excision mask is now set by the RadiationM1::SetMask task
//  (implemented in radiation_m1_tasks.cpp), which derives radiation_mask from the
//  coordinate excision mask (pcoord->excision_floor) and zeroes the radiation fields
//  in excised cells. This header is retained only as a placeholder; the stub below
//  is unused.

namespace radiation_m1 {

KOKKOS_INLINE_FUNCTION
void RadiationM1SetMask() {
  // see RadiationM1::SetMask in radiation_m1_tasks.cpp
}
}
#endif //RADIATION_M1_SETMASK_HPP
