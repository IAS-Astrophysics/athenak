//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file drift.cpp
//! \brief implementation of the particle drift pusher

#include <algorithm>
#include <cmath>
#include <limits>

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
  auto &pi = prtcl_idata;
  auto dt_ = pmy_pack->pmesh->dt;

  par_for("part_update", DevExeSpace(), 0, (nprtcl_thispack - 1),
  KOKKOS_LAMBDA(const int p) {
    if (pi(PSTATUS,p) == PACTIVE) {
      pr(IPX,p) += 0.5*dt_*pr(cosmic_ray::IPVX,p);

      if (multi_d) {
        pr(IPY,p) += 0.5*dt_*pr(cosmic_ray::IPVY,p);
      }

      if (three_d) {
        pr(IPZ,p) += 0.5*dt_*pr(cosmic_ray::IPVZ,p);
      }
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn Real Particles::EstimateTimestepDrift
//! \brief Return the minimum cell-crossing limit over active particles.

Real Particles::EstimateTimestepDrift() {
  if (nprtcl_thispack == 0) return std::numeric_limits<float>::max();

  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int gids = pmy_pack->gids;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  Real dt = std::numeric_limits<float>::max();
  Kokkos::parallel_reduce(
      "particle_drift_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nprtcl_thispack),
      KOKKOS_LAMBDA(const int p, Real &min_dt) {
        if (pi(PSTATUS,p) != PACTIVE) return;

        const int m = pi(PGID,p) - gids;
        const Real vx = fabs(pr(cosmic_ray::IPVX,p));
        if (vx > 0.0) min_dt = fmin(min_dt, mbsize.d_view(m).dx1/vx);

        if (multi_d) {
          const Real vy = fabs(pr(cosmic_ray::IPVY,p));
          if (vy > 0.0) min_dt = fmin(min_dt, mbsize.d_view(m).dx2/vy);
        }
        if (three_d) {
          const Real vz = fabs(pr(cosmic_ray::IPVZ,p));
          if (vz > 0.0) min_dt = fmin(min_dt, mbsize.d_view(m).dx3/vz);
        }
      }, Kokkos::Min<Real>(dt));

  return std::min(dt, static_cast<Real>(std::numeric_limits<float>::max()));
}
} // namespace particles
