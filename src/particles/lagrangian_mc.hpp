#ifndef PARTICLES_LAGRANGIAN_MC_HPP_
#define PARTICLES_LAGRANGIAN_MC_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file lagrangian_mc.hpp
//! \brief particle array indices for Lagrangian Monte Carlo particles

#include "athena.hpp"

namespace particles {
namespace lagrangian_mc {

constexpr int NREAL = NPARTICLE_REAL_COMMON;
enum IntIndex {PLASTMOVE=NPARTICLE_INT_COMMON, NINT};
enum MoveDirection {
  PMOVE_NONE=0, PMOVE_X1_LEFT, PMOVE_X1_RIGHT, PMOVE_X2_LEFT,
  PMOVE_X2_RIGHT, PMOVE_X3_LEFT, PMOVE_X3_RIGHT
};

} // namespace lagrangian_mc
} // namespace particles

#endif // PARTICLES_LAGRANGIAN_MC_HPP_
