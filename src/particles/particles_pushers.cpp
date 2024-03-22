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
//  \brief

TaskStatus Particles::Push(Driver *pdriver, int stage) {
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

  // Used in lagrangian_mc pusher
  auto dtold_ = pmy_pack->pmesh->dtold;
  auto &u1_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->u1:pmy_pack->pmhd->u1;
  auto &uflx = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->uflx:pmy_pack->pmhd->uflx;
  auto &flx1_ = uflx.x1f;
  auto &flx2_ = uflx.x2f;
  auto &flx3_ = uflx.x3f;

  // TODO(GNW): Maybe move this outside and pass it in as needed? What is the
  //            overhead associated with recreating the pool each call?

  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmy_pack->gids);

  // TODO(GNW): Note that for MC tracers we need to handle mesh refinement (in AMR)
  //            more carefully.

  // TODO(GNW): Is it better to do an if statement here and capture the variables
  //            for the lambda or keep the switch?

  // TODO(GNW): Are these fluxes are the correct ones to use (RK question)?

  // TODO(GNW): The particle pusher gets called before the time integrator, but
  //            for the MC pusher we need to know the fluxes from the previous
  //            time step, which means we need to read from u1 and fluxes.
  //            One way to deal with this is just to have the MC pusher update
  //            particle positions "one step out of phase" and use dtold. An
  //            alternative is to have the MC pusher use a separate task that
  //            gets called after the time integrator.

  switch (pusher) {
    case ParticlesPusher::drift:

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

      break;

    case ParticlesPusher::lagrangian_mc:

      par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        // get random number state this thread
        auto rand_gen = rand_pool64.get_state();

        int m = pi(PGID,p) - gids;

        int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
        int jp = 0;
        int kp = 0;

        if (multi_d) {
          jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
        }

        if (three_d) {
          kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
        }

        bool particle_has_moved = false;
        Real reduced_mass = u1_(m,IDN,kp,jp,ip);

        // by convention, these values will be negative when there is outflow
        // with respect to the current particle's cell
        Real flx1_left = flx1_(m,IDN,kp,jp,ip);
        Real flx1_right = -flx1_(m,IDN,kp,jp,ip+1);
        Real flx2_left = (multi_d) ? flx2_(m,IDN,kp,jp,jp) : 0.;
        Real flx2_right = (multi_d) ? -flx2_(m,IDN,kp,jp+1,jp) : 0.;
        Real flx3_left = (three_d) ? flx3_(m,IDN,kp,kp,jp) : 0.;
        Real flx3_right = (three_d) ? -flx3_(m,IDN,kp+1,kp,jp) : 0.;

        flx1_left *= mbsize.d_view(m).dx1 * dtold_;
        flx1_right *= mbsize.d_view(m).dx1 * dtold_;

        flx2_left *= mbsize.d_view(m).dx2 * dtold_;
        flx2_right *= mbsize.d_view(m).dx2 * dtold_;

        flx3_left *= mbsize.d_view(m).dx3 * dtold_;
        flx3_right *= mbsize.d_view(m).dx3 * dtold_;

        if (flx1_left < 0) {
          if (rand_gen.frand() < fabs(flx1_left)/reduced_mass) {
            pr(IPX,p) = (ip - 1 - is) * mbsize.d_view(m).dx1 + mbsize.d_view(m).x1min;
            particle_has_moved = true;
          }
          reduced_mass += flx1_left;
        }

        if (!particle_has_moved && flx1_right < 0) {
          if (rand_gen.frand() < fabs(flx1_right)/reduced_mass) {
            pr(IPX,p) = (ip + 1 - is) * mbsize.d_view(m).dx1 + mbsize.d_view(m).x1min;
            particle_has_moved = true;
          }
          reduced_mass += flx1_right;
        }

        if (multi_d) {
          if (!particle_has_moved && flx2_left < 0) {
            if (rand_gen.frand() < fabs(flx2_left)/reduced_mass) {
              pr(IPY,p) = (jp - 1 - js) * mbsize.d_view(m).dx2 + mbsize.d_view(m).x2min;
              particle_has_moved = true;
            }
            reduced_mass += flx2_left;
          }

          if (!particle_has_moved && flx2_right < 0) {
            if (rand_gen.frand() < fabs(flx2_right)/reduced_mass) {
              pr(IPY,p) = (jp + 1 - js) * mbsize.d_view(m).dx2 + mbsize.d_view(m).x2min;
              particle_has_moved = true;
            }
            reduced_mass += flx2_right;
          }
        }

        if (three_d) {
          if (!particle_has_moved && flx3_left < 0) {
            if (rand_gen.frand() < fabs(flx3_left)/reduced_mass) {
              pr(IPZ,p) = (kp - 1 - ks) * mbsize.d_view(m).dx3 + mbsize.d_view(m).x3min;
              particle_has_moved = true;
            }
            reduced_mass += flx3_left;
          }

          if (!particle_has_moved && flx3_right < 0) {
            if (rand_gen.frand() < fabs(flx3_right)/reduced_mass) {
              pr(IPZ,p) = (kp + 1 - ks) * mbsize.d_view(m).dx3 + mbsize.d_view(m).x3min;
              particle_has_moved = true;
            }
            reduced_mass += flx3_right;
          }
        }
        rand_pool64.free_state(rand_gen);  // free state for use by other threads
      });

      break;

    default:
      break;
  }

  return TaskStatus::complete;
}
} // namespace particles
