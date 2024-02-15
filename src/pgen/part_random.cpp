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
  auto &ppos = pmy_mesh_->pmb_pack->ppart->prtcl_pos;
  auto &pvel = pmy_mesh_->pmb_pack->ppart->prtcl_vel;
  auto &pgid = pmy_mesh_->pmb_pack->ppart->prtcl_gid;
  auto &npart = pmy_mesh_->pmb_pack->ppart->nprtcl_thispack;
  auto gids = pmy_mesh_->pmb_pack->gids;
  auto gide = pmy_mesh_->pmb_pack->gide;

  // initialize particles
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("part_update",DevExeSpace(),0,npart,
  KOKKOS_LAMBDA(const int p) {
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    // choose parent MeshBlock randomly
    int m = static_cast<int>(rand_gen.frand()*(gide - gids + 1.0));
    pgid(p) = gids + m;

    ppos(p,IPX) = mbsize.d_view(m).x1min +
                  rand_gen.frand()*(mbsize.d_view(m).x1max - mbsize.d_view(m).x1min);
    ppos(p,IPY) = mbsize.d_view(m).x2min +
                  rand_gen.frand()*(mbsize.d_view(m).x2max - mbsize.d_view(m).x2min);
    ppos(p,IPZ) = mbsize.d_view(m).x3min +
                  rand_gen.frand()*(mbsize.d_view(m).x3max - mbsize.d_view(m).x3min);

    pvel(p,IPX) = 2.0*(rand_gen.frand() - 0.5);
    pvel(p,IPY) = 2.0*(rand_gen.frand() - 0.5);
    pvel(p,IPZ) = 2.0*(rand_gen.frand() - 0.5);

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
