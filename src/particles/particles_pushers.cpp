//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_pushers.cpp
//! \brief dispatch to the particle pusher selected from the input file

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus ParticlePopulation::Push
//! \brief Dispatch to the selected particle pusher.

TaskStatus ParticlePopulation::Push(Driver *pdriver, int stage) {
  switch (pusher) {
    case ParticlesPusher::drift:
      return PushDrift(pdriver, stage);
    case ParticlesPusher::lagrangian_mc:
      return PushLagrangianMC(pdriver, stage);
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher has no implementation" << std::endl;
      std::exit(EXIT_FAILURE);
  }
}
} // namespace particles
