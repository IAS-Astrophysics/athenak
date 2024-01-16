//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_pushers.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::ParticlesPush
//  \brief

TaskStatus Particles::Push(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto pgid = prtcl_gid;
  auto px = prtcl_x, py = prtcl_y, pz = prtcl_z;
  auto pvx = prtcl_vx, pvy = prtcl_vy, pvz = prtcl_vz;
  auto dt_ = (pmy_pack->pmesh->dt);

  switch (pusher) {
    case ParticlesPusher::drift:

      par_for("part_update",DevExeSpace(),0,nparticles_thispack,
      KOKKOS_LAMBDA(const int p) {
        int pmb = pgid.d_view(p);
        int ip = (prtcl_x(p) - mbsize.d_view(pmb).x1min)/mbsize.d_view(pmb).dx1 + is;
        px(p) += 0.5*dt_*pvx(p);

        if (multi_d) {
          int jp = (prtcl_y(p) - mbsize.d_view(pmb).x2min)/mbsize.d_view(pmb).dx2 + js;
          py(p) += 0.5*dt_*pvy(p);
        }

        if (three_d) {
          int kp = (prtcl_z(p) - mbsize.d_view(pmb).x3min)/mbsize.d_view(pmb).dx3 + ks;
          pz(p) += 0.5*dt_*pvz(p);
        }
      });

    break;
  default:
    break;
  }

  return TaskStatus::complete;
}
} // namespace particle
