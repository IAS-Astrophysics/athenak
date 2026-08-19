#ifndef PARTICLES_COSMIC_RAY_HPP_
#define PARTICLES_COSMIC_RAY_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cosmic_ray.hpp
//! \brief particle array indices for cosmic-ray particles

namespace particles {
namespace cosmic_ray {

// The common particle framework requires the first three real fields to be the x1, x2,
// and x3 positions, and the first three integer fields to be the owning GID, durable tag,
// and lifecycle status.
enum RealIndex {IPX=0, IPY=1, IPZ=2, IPVX=3, IPVY=4, IPVZ=5, NREAL=6};
enum IntIndex {PGID=0, PTAG=1, PSTATUS=2, NINT=3};
constexpr int RESTART_LAYOUT_VERSION = 1;

} // namespace cosmic_ray
} // namespace particles

#endif // PARTICLES_COSMIC_RAY_HPP_
