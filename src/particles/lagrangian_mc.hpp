#ifndef PARTICLES_LAGRANGIAN_MC_HPP_
#define PARTICLES_LAGRANGIAN_MC_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file lagrangian_mc.hpp
//! \brief particle array indices for Lagrangian Monte Carlo particles

namespace particles {
namespace lagrangian_mc {

// The common particle framework requires the first three real fields to be the x1, x2,
// and x3 positions, and the first three integer fields to be the owning GID, durable tag,
// and lifecycle status.
enum RealIndex {IPX=0, IPY=1, IPZ=2, NREAL=3};
enum IntIndex {PGID=0, PTAG=1, PSTATUS=2, PLASTMOVE=3, NINT=4};
constexpr int RESTART_LAYOUT_VERSION = 1;
enum MoveDirection {
  PMOVE_NONE=0, PMOVE_X1_LEFT, PMOVE_X1_RIGHT, PMOVE_X2_LEFT,
  PMOVE_X2_RIGHT, PMOVE_X3_LEFT, PMOVE_X3_RIGHT
};

} // namespace lagrangian_mc
} // namespace particles

#endif // PARTICLES_LAGRANGIAN_MC_HPP_
