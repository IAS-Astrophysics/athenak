//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file drift.cpp
//! \brief implementation of the particle drift pusher

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "particles/cosmic_ray.hpp"
#include "particles/particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::PushDrift
//! \brief Push particles using their stored velocities.

TaskStatus Particles::PushDrift(Driver*, int) {
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &pr = prtcl_rdata;
  auto dt_ = pmy_pack->pmesh->dt;

  par_for("part_update", DevExeSpace(), 0, (nprtcl_thispack - 1),
  KOKKOS_LAMBDA(const int p) {
    pr(IPX,p) += 0.5*dt_*pr(cosmic_ray::IPVX,p);

    if (multi_d) {
      pr(IPY,p) += 0.5*dt_*pr(cosmic_ray::IPVY,p);
    }

    if (three_d) {
      pr(IPZ,p) += 0.5*dt_*pr(cosmic_ray::IPVZ,p);
    }
  });

  return TaskStatus::complete;
}
} // namespace particles
