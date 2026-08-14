//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particles_newdt.cpp
//! \brief Calculate the particle contribution to the next timestep.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::NewTimeStep
//! \brief Combine the model and optional user-enrolled particle timestep limits.

TaskStatus Particles::NewTimeStep(Driver*, int) {
  Real model_dt = std::numeric_limits<float>::max();
  switch (pusher) {
    case ParticlesPusher::drift:
      model_dt = EstimateTimestepDrift();
      break;
    case ParticlesPusher::lagrangian_mc:
      // Lagrangian MC transport follows the fluid CFL limit.
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher has no timestep implementation"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  if (!(model_dt > 0.0) || !(std::isfinite(model_dt))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle model returned an invalid timestep" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real user_dt = std::numeric_limits<float>::max();
  auto *pgen = pmy_pack->pmesh->pgen.get();
  if (pgen != nullptr && pgen->user_particle_dt_func != nullptr) {
    user_dt = pgen->user_particle_dt_func(pmy_pack);
    if (!(user_dt > 0.0) || !(std::isfinite(user_dt))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User particle function returned an invalid timestep"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  dtnew = std::min(model_dt, user_dt);
  return TaskStatus::complete;
}
} // namespace particles
