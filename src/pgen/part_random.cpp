//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_random.cpp
//! \brief Problem generator that initializes random particle positions and velocities.

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for random particle positions/velocities

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Random particles test requires <particles> block in input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &mbsize = pmy_mesh_->pmb_pack->pmb->mb_size;
  auto &pr = pmy_mesh_->pmb_pack->ppart->prtcl_rdata;
  auto &pi = pmy_mesh_->pmb_pack->ppart->prtcl_idata;
  auto &npart = pmy_mesh_->pmb_pack->ppart->nprtcl_thispack;
  auto gids = pmy_mesh_->pmb_pack->gids;
  auto gide = pmy_mesh_->pmb_pack->gide;

  // initialize particles
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("part_update",DevExeSpace(),0,(npart-1),
  KOKKOS_LAMBDA(const int p) {
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    // choose parent MeshBlock randomly
    int m = static_cast<int>(rand_gen.frand()*(gide - gids + 1.0));
    pi(PGID,p) = gids + m;

    Real rand = rand_gen.frand();
    pr(IPX,p) = (1. - rand)*mbsize.d_view(m).x1min + rand*mbsize.d_view(m).x1max;
    pr(IPX,p) = fmin(pr(IPX,p),mbsize.d_view(m).x1max);
    pr(IPX,p) = fmax(pr(IPX,p),mbsize.d_view(m).x1min);

    rand = rand_gen.frand();
    pr(IPY,p) = (1. - rand)*mbsize.d_view(m).x2min + rand*mbsize.d_view(m).x2max;
    pr(IPY,p) = fmin(pr(IPY,p),mbsize.d_view(m).x2max);
    pr(IPY,p) = fmax(pr(IPY,p),mbsize.d_view(m).x2min);

    rand = rand_gen.frand();
    pr(IPZ,p) = (1. - rand)*mbsize.d_view(m).x3min + rand*mbsize.d_view(m).x3max;
    pr(IPZ,p) = fmin(pr(IPZ,p),mbsize.d_view(m).x3max);
    pr(IPZ,p) = fmax(pr(IPZ,p),mbsize.d_view(m).x3min);

    pr(IPVX,p) = 2.0*(rand_gen.frand() - 0.5);
    pr(IPVY,p) = 2.0*(rand_gen.frand() - 0.5);
    pr(IPVZ,p) = 2.0*(rand_gen.frand() - 0.5);

    rand_pool64.free_state(rand_gen);  // free state for use by other threads
  });

  // set timestep (which will remain constant for entire run
  // Assumes uniform mesh (no SMR or AMR)
  // Assumes velocities normalized to one, so dt=min(dx)
  Real &dtnew_ = pmy_mesh_->pmb_pack->ppart->dtnew;
  dtnew_ = std::min(mbsize.h_view(0).dx1, mbsize.h_view(0).dx2);
  dtnew_ = std::min(dtnew_, mbsize.h_view(0).dx3);

  return;
}
