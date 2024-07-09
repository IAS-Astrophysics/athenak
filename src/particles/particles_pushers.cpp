//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_pushers.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

#include <Kokkos_Random.hpp>

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::ParticlesPush
//  \brief wrapper with switch to access different particle pushers

TaskStatus Particles::Push(Driver *pdriver, int stage) {

  switch (pusher) {
    case ParticlesPusher::drift:
      PushDrift();
      break;

    case ParticlesPusher::lagrangian_mc:
      PushLagrangianMC();
      break;

    default:
      break;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::PushDrift
//! \brief push particles based on stored particle internal velocity

void Particles::PushDrift() {  
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto dt_ = pmy_pack->pmesh->dt;
  auto gids = pmy_pack->gids;

  par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
    pr(IPX,p) += 0.5*dt_*pr(IPVX,p);

    if (multi_d) {
      int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
      pr(IPY,p) += 0.5*dt_*pr(IPVY,p);
    }

    if (three_d) {
      int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
      pr(IPZ,p) += 0.5*dt_*pr(IPVZ,p);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::PushLagrangianMC
//! \brief push particles using Lagrangian Monte Carlo method (Genel+ 2013, MNRAS.435.1426G)
//         WARNING: this implementation may not work well with AMR

void Particles::PushLagrangianMC() {

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &gids = pmy_pack->gids;
  auto &mblev = pmy_pack->pmb->mb_lev;

  auto &u1_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->u1:pmy_pack->pmhd->u1;
  auto &uflxidn_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->uflxidnsaved:pmy_pack->pmhd->uflxidnsaved;
  auto &flx1_ = uflxidn_.x1f;
  auto &flx2_ = uflxidn_.x2f;
  auto &flx3_ = uflxidn_.x3f;

  auto &rand_pool64_ = rand_pool64;

  // GNW 2024-JUL-5: Warning, this may not play well with AMR
  par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {

    int m = pi(PGID,p) - gids;

    int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
    int jp = js;
    int kp = ks;

    if (multi_d) {
      jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
    }

    if (three_d) {
      kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
    }

    // get normalized fluxes based on local density
    Real mass = u1_(m,IDN,kp,jp,ip);

    // by convention, these values will be positive when there is outflow
    // with respect to the current particle's cell
    Real flx1_left = -flx1_(m,kp,jp,ip) / mass;
    Real flx1_right = flx1_(m,kp,jp,ip+1) / mass;
    Real flx2_left = (multi_d) ? -flx2_(m,kp,jp,ip) / mass : 0.;
    Real flx2_right = (multi_d) ? flx2_(m,kp,jp+1,ip) / mass : 0.;
    Real flx3_left = (three_d) ? -flx3_(m,kp,jp,ip) / mass : 0.;
    Real flx3_right = (three_d) ? flx3_(m,kp+1,jp,ip) / mass : 0.;

    flx1_left = flx1_left < 0 ? 0 : flx1_left;
    flx1_right = flx1_right < 0 ? 0 : flx1_right;
    flx2_left = flx2_left < 0 ? 0 : flx2_left;
    flx2_right = flx2_right < 0 ? 0 : flx2_right;
    flx3_left = flx3_left < 0 ? 0 : flx3_left;
    flx3_right = flx3_right < 0 ? 0 : flx3_right;

    auto rand_gen = rand_pool64_.get_state();
    Real rand = rand_gen.frand();
    rand_pool64_.free_state(rand_gen);  // free state for use by other threads

    // save refinement level of current zone
    pi(PLASTLEVEL,p) = mblev.d_view(m);

    // save parity of current zone stored as (i_isodd,j_isodd,k_isodd) * 8
    pi(PLASTMOVE,p) = 32 * (ip % 2) + 16 * (jp % 2) + 8 * (kp % 2);

    if (rand < flx1_left) {
      pr(IPX,p) -= mbsize.d_view(m).dx1;
      pi(PLASTMOVE,p) += 1;
    } else if (rand < flx1_left + flx1_right) {
      pr(IPX,p) += mbsize.d_view(m).dx1;
      pi(PLASTMOVE,p) += 2;
    } else if (multi_d && rand < flx1_left + flx1_right + flx2_left) {
      pr(IPY,p) -= mbsize.d_view(m).dx2;
      pi(PLASTMOVE,p) += 3;
    } else if (multi_d && rand < flx1_left + flx1_right + flx2_left + flx2_right) {
      pr(IPY,p) += mbsize.d_view(m).dx2;
      pi(PLASTMOVE,p) += 4;
    } else if (three_d && rand < flx1_left + flx1_right + flx2_left + flx2_right + flx3_left) {
      pr(IPZ,p) -= mbsize.d_view(m).dx3;
      pi(PLASTMOVE,p) += 5;
    } else if (three_d && rand < flx1_left + flx1_right + flx2_left + flx2_right + flx3_left + flx3_right) {
      pr(IPZ,p) += mbsize.d_view(m).dx3;
      pi(PLASTMOVE,p) += 6;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::AdjustMeshRefinement
//! \brief update locations of particles that enter meshblocks with new refinement levels

TaskStatus Particles::AdjustMeshRefinement(Driver *pdriver, int stage) {

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &gids = pmy_pack->gids;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;

  auto &uflxidn_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->uflxidnsaved:pmy_pack->pmhd->uflxidnsaved;
  auto &flx1_ = uflxidn_.x1f;
  auto &flx2_ = uflxidn_.x2f;
  auto &flx3_ = uflxidn_.x3f;

  auto &rand_pool64_ = rand_pool64;
  
  par_for("particle_meshshift",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {

    int m = pi(PGID,p) - gids;
    int level = mblev.d_view(m);

    int lastlevel = pi(PLASTLEVEL,p);
    int lastmove = pi(PLASTMOVE,p);

    // oddness of the last cell that the particle lived in
    int i_parity = lastmove / 32;
    int j_parity = (lastmove % 32) / 16;
    int k_parity = (lastmove % 16) / 8;

    // direction of last move:
    //   1 -> "left" x1 face was chosen
    //   2 -> "right" x1 face was chosen
    //   3 -> "left" x2 face was chosen
    //   4 -> "right" x2 face was chosen
    //   5 -> "left" x3 face was chosen
    //   6 -> "right" x3 face was chosen
    lastmove = lastmove % 8;

    Real dx1 = mbsize.d_view(m).dx1;
    Real dx2 = multi_d ? mbsize.d_view(m).dx2 : 0.;
    Real dx3 = three_d ? mbsize.d_view(m).dx3 : 0.;

    if (level > lastlevel) {
      // this is a higher refinement level, i.e., the zones are smaller now
 
      if (lastmove == 1) {
        // came from zone to right (dx--)
        pr(IPX,p) += dx1/2;

        pr(IPY,p) -= dx2/2;
        pr(IPZ,p) -= dx3/2;
      } else if (lastmove == 2) {
        // came from zone to left (dx++)
        pr(IPX,p) -= dx1/2;

        pr(IPY,p) -= dx2/2;
        pr(IPZ,p) -= dx3/2;
      } else if (multi_d && lastmove == 3) {
        // came from zone above (dy--)
        pr(IPY,p) += dx2/2;

        pr(IPX,p) -= dx1/2;
        pr(IPZ,p) -= dx3/2;
      } else if (multi_d && lastmove == 4) {
        // came from zone below (dy++)
        pr(IPY,p) -= dx2/2;

        pr(IPX,p) -= dx1/2;
        pr(IPZ,p) -= dx3/2;
      } else if (three_d && lastmove == 5) {
        // came from zone in front (dz--)
        pr(IPZ,p) += dx3/2;

        pr(IPX,p) -= dx1/2;
        pr(IPY,p) -= dx2/2;
      } else if (three_d && lastmove == 6) {
        // came from zone behind (dz++)
        pr(IPZ,p) -= dx3/2;

        pr(IPX,p) -= dx1/2;
        pr(IPY,p) -= dx2/2;
      }

      int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
      int jp = js;
      int kp = ks;

      if (multi_d) {
        jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
      }

      if (three_d) {
        kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
      }

      // get fluxes into the four zones that the particle could have ended up in
      Real flx1 = 0.;
      Real flx2 = 0.;
      Real flx3 = 0.;
      Real flx4 = 0.;

      if (lastmove == 1) {
        // came from zone to the right
        flx1 = -flx1_(m,kp,jp,ip+1);
        flx2 = (multi_d) ? -flx1_(m,kp,jp+1,ip+1) : 0.;
        flx3 = (three_d) ? -flx1_(m,kp+1,jp,ip+1) : 0.;
        flx4 = (multi_d && three_d) ? -flx1_(m,kp+1,jp+1,ip+1) : 0.;
      } else if (lastmove == 2) {
        // came from zone to the left
        flx1 = flx1_(m,kp,jp,ip);
        flx2 = (multi_d) ? flx1_(m,kp,jp+1,ip) : 0.;
        flx3 = (three_d) ? flx1_(m,kp+1,jp,ip) : 0.;
        flx4 = (multi_d && three_d) ? flx1_(m,kp+1,jp+1,ip) : 0.;
      } else if (lastmove == 3) {
        // came from zone above. is at least multi_d
        flx1 = -flx2_(m,kp,jp+1,ip);
        flx2 = -flx2_(m,kp,jp+1,ip+1);
        flx3 = (three_d) ? -flx2_(m,kp+1,jp+1,ip) : 0.;
        flx4 = (three_d) ? -flx2_(m,kp+1,jp+1,ip+1) : 0.;
      } else if (lastmove == 4) {
        // came from zone below. is at least multi_d
        flx1 = flx2_(m,kp,jp,ip);
        flx2 = flx2_(m,kp,jp,ip+1);
        flx3 = (three_d) ? flx2_(m,kp+1,jp,ip) : 0.;
        flx4 = (three_d) ? flx2_(m,kp+1,jp,ip+1) : 0.;
      } else if (lastmove == 5) {
        // came from zone in front. is three_d
        flx1 = -flx3_(m,kp+1,jp,ip);
        flx2 = -flx3_(m,kp+1,jp,ip+1);
        flx3 = -flx3_(m,kp+1,jp+1,ip);
        flx4 = -flx3_(m,kp+1,jp+1,ip+1);
      } else if (lastmove == 6) {
        // came from zone behind. is three_d
        flx1 = flx3_(m,kp,jp,ip);
        flx2 = flx3_(m,kp,jp,ip+1);
        flx3 = flx3_(m,kp,jp+1,ip);
        flx4 = flx3_(m,kp,jp+1,ip+1);
      }

      flx1 = (flx1 < 0) ? 0. : flx1;
      flx2 = (flx2 < 0) ? 0. : flx2;
      flx3 = (flx3 < 0) ? 0. : flx3;
      flx4 = (flx4 < 0) ? 0. : flx4;

      Real flx_total = flx1 + flx2 + flx3 + flx4;
      flx_total = (flx_total > 0) ? flx_total : 1.e-10;

      flx1 /= flx_total;
      flx2 /= flx_total;
      flx3 /= flx_total;
      flx4 /= flx_total;

      auto rand_gen = rand_pool64_.get_state();
      Real rand = rand_gen.frand();
      rand_pool64_.free_state(rand_gen);  // free state for use by other threads
      
      int target_zone = 4;
      if (rand < flx1) {
        target_zone = 1;
      } else if (rand < flx1 + flx2) {
        target_zone = 2;
      } else if (rand < flx1 + flx2 + flx3) {
        target_zone = 3;
      }

      if (lastmove == 1 || lastmove == 2) {
        if (target_zone == 2) {
          pr(IPY,p) += mbsize.d_view(m).dx2;
        } else if (target_zone == 3) {
          pr(IPZ,p) += mbsize.d_view(m).dx3;
        } else if (target_zone == 4) {
          pr(IPY,p) += mbsize.d_view(m).dx2;
          pr(IPZ,p) += mbsize.d_view(m).dx3;
        }
      } else if (lastmove == 3 || lastmove == 4) {
        if (target_zone == 2) {
          pr(IPX,p) += mbsize.d_view(m).dx1;
        } else if (target_zone == 3) {
          pr(IPZ,p) += mbsize.d_view(m).dx3;
        } else if (target_zone == 4) {
          pr(IPX,p) += mbsize.d_view(m).dx1;
          pr(IPZ,p) += mbsize.d_view(m).dx3;
        }
      } else if (lastmove == 5 || lastmove == 6) {
        if (target_zone == 2) {
          pr(IPX,p) += mbsize.d_view(m).dx1;
        } else if (target_zone == 3) {
          pr(IPY,p) += mbsize.d_view(m).dx2;
        } else if (target_zone == 4) {
          pr(IPX,p) += mbsize.d_view(m).dx1;
          pr(IPY,p) += mbsize.d_view(m).dx2;
        }
      }

    } else if (level < lastlevel) {
      // this is a lower refinement level, i.e., the zones are larger now,
      // there's nothing special to do other than to move the particle to
      // the center of the new zone

      if (i_parity) {
        pr(IPX,p) -= mbsize.d_view(m).dx1/4;
      } else {
        pr(IPX,p) += mbsize.d_view(m).dx1/4;
      }
      if (multi_d) {
        if (j_parity) {
          pr(IPY,p) -= mbsize.d_view(m).dx2/4;
        } else {
          pr(IPY,p) += mbsize.d_view(m).dx2/4;
        }
      }
      if (three_d) {
        if (k_parity) {
          pr(IPZ,p) -= mbsize.d_view(m).dx3/4;
        } else {
          pr(IPZ,p) += mbsize.d_view(m).dx3/4;
        }
      }

      if (lastmove == 1) {
        // came from zone to right (dx--)
        pr(IPX,p) -= mbsize.d_view(m).dx1/2;
      } else if (lastmove == 2) {
        // came from zone to left (dx++)
        pr(IPX,p) += mbsize.d_view(m).dx1/2;
      } else if (lastmove == 3) {
        // came from zone above (dy--)
        pr(IPY,p) -= mbsize.d_view(m).dx2/2;
      } else if (lastmove == 4) {
        // came from zone below (dy++)
        pr(IPY,p) += mbsize.d_view(m).dx2/2;
      } else if (lastmove == 5) {
        // came from zone in front (dz--)
        pr(IPZ,p) -= mbsize.d_view(m).dx3/2;
      } else if (lastmove == 6) {
        // came from zone behind (dz++)
        pr(IPZ,p) += mbsize.d_view(m).dx3/2;
      }
    }
  });

  return TaskStatus::complete;
}

} // namespace particles
