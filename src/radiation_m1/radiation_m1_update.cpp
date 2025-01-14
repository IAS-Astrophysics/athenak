//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_update.cpp
//! \brief perform semi-implicit update for M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "radiation_m1.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::RKUpdate(Driver *d, int stage) {
  // all semi-implicit updates go here
  return TaskStatus::complete;
}
}