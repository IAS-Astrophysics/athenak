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
  auto &pgid = prtcl_gid;
  auto &ppos = prtcl_pos;
  auto &pvel = prtcl_vel;
  auto dt_ = (pmy_pack->pmesh->dt);

  switch (pusher) {
    case ParticlesPusher::drift:

      par_for("part_update",DevExeSpace(),0,nprtcl_thispack,
      KOKKOS_LAMBDA(const int p) {
        int &gid = pgid.d_view(p);
        int ip = (ppos(p,IPX) - mbsize.d_view(gid).x1min)/mbsize.d_view(gid).dx1 + is;
        ppos(p,IPX) += 0.5*dt_*pvel(p,IPX);

        if (multi_d) {
          int jp = (ppos(p,IPY) - mbsize.d_view(gid).x2min)/mbsize.d_view(gid).dx2 + js;
          ppos(p,IPY) += 0.5*dt_*pvel(p,IPY);
        }

        if (three_d) {
          int kp = (ppos(p,IPZ) - mbsize.d_view(gid).x3min)/mbsize.d_view(gid).dx3 + ks;
          ppos(p,IPZ) += 0.5*dt_*pvel(p,IPZ);
        }
      });

    break;
  default:
    break;
  }

  return TaskStatus::complete;
}
} // namespace particles
