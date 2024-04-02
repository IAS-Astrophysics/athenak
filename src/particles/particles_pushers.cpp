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
//  \brief TODO(GNW): write here

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
//! \brief TODO(GNW): write here

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
//! \brief TODO(GNW): write here

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
  auto gids = pmy_pack->gids;

  auto &u1_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->u1:pmy_pack->pmhd->u1;
  auto &uflxidn_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->uflxidnsaved:pmy_pack->pmhd->uflxidnsaved;
  auto &flx1_ = uflxidn_.x1f;
  auto &flx2_ = uflxidn_.x2f;
  auto &flx3_ = uflxidn_.x3f;

  // TODO(GNW): be careful with AMR
  // TODO(GNW): this does not work with SMR yet

  par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {

    auto rand_gen = rand_pool64.get_state();

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

    Real rand = rand_gen.frand();

    if (rand < flx1_left) {
      pr(IPX,p) -= mbsize.d_view(m).dx1;
    } else if (rand < flx1_left + flx1_right) {
      pr(IPX,p) += mbsize.d_view(m).dx1;
    } else if (multi_d && rand < flx1_left + flx1_right + flx2_left) {
      pr(IPY,p) -= mbsize.d_view(m).dx2;
    } else if (multi_d && rand < flx1_left + flx1_right + flx2_left + flx2_right) {
      pr(IPY,p) += mbsize.d_view(m).dx2;
    } else if (three_d && rand < flx1_left + flx1_right + flx2_left + flx2_right + flx3_left) {
      pr(IPZ,p) -= mbsize.d_view(m).dx3;
    } else if (three_d && rand < flx1_left + flx1_right + flx2_left + flx2_right + flx3_left + flx3_right) {
      pr(IPZ,p) += mbsize.d_view(m).dx3;
    }

    rand_pool64.free_state(rand_gen);  // free state for use by other threads
  });
}

} // namespace particles
