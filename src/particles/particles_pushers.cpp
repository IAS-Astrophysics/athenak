//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_pushers.cpp
//! \brief dispatch to the particle pusher selected from the input file

#include "athena.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::Push
//! \brief Dispatch to the selected particle pusher.

TaskStatus Particles::Push(Driver *pdriver, int stage) {
  switch (pusher) {
    case ParticlesPusher::drift:
      return PushDrift(pdriver, stage);
    default:
      break;
  }

  return TaskStatus::complete;
}
} // namespace particles
