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
#include "particles/cosmic_ray.hpp"
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
  auto *population = pmbp->ppart->FindPopulation("particles");

  // capture variables for the kernel
  auto &mbsize = pmbp->pmb->mb_size;
  auto &pr = population->prtcl_rdata;
  auto &pi = population->prtcl_idata;
  auto &npart = population->nprtcl_thispack;
  auto gids = pmbp->gids;
  auto gide = pmbp->gide;

  // initialize particles
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("part_update",DevExeSpace(),0,(npart-1),
  KOKKOS_LAMBDA(const int p) {
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    // choose parent MeshBlock randomly
    int m = static_cast<int>(rand_gen.frand()*(gide - gids + 1.0));
    pi(particles::cosmic_ray::PGID,p) = gids + m;

    Real rand = rand_gen.frand();
    pr(particles::cosmic_ray::IPX,p) =
        (1. - rand)*mbsize.d_view(m).x1min + rand*mbsize.d_view(m).x1max;
    pr(particles::cosmic_ray::IPX,p) =
        fmin(pr(particles::cosmic_ray::IPX,p),mbsize.d_view(m).x1max);
    pr(particles::cosmic_ray::IPX,p) =
        fmax(pr(particles::cosmic_ray::IPX,p),mbsize.d_view(m).x1min);

    rand = rand_gen.frand();
    pr(particles::cosmic_ray::IPY,p) =
        (1. - rand)*mbsize.d_view(m).x2min + rand*mbsize.d_view(m).x2max;
    pr(particles::cosmic_ray::IPY,p) =
        fmin(pr(particles::cosmic_ray::IPY,p),mbsize.d_view(m).x2max);
    pr(particles::cosmic_ray::IPY,p) =
        fmax(pr(particles::cosmic_ray::IPY,p),mbsize.d_view(m).x2min);

    rand = rand_gen.frand();
    pr(particles::cosmic_ray::IPZ,p) =
        (1. - rand)*mbsize.d_view(m).x3min + rand*mbsize.d_view(m).x3max;
    pr(particles::cosmic_ray::IPZ,p) =
        fmin(pr(particles::cosmic_ray::IPZ,p),mbsize.d_view(m).x3max);
    pr(particles::cosmic_ray::IPZ,p) =
        fmax(pr(particles::cosmic_ray::IPZ,p),mbsize.d_view(m).x3min);

    pr(particles::cosmic_ray::IPVX,p) = 2.0*(rand_gen.frand() - 0.5);
    pr(particles::cosmic_ray::IPVY,p) = 2.0*(rand_gen.frand() - 0.5);
    pr(particles::cosmic_ray::IPVZ,p) = 2.0*(rand_gen.frand() - 0.5);

    rand_pool64.free_state(rand_gen);  // free state for use by other threads
  });

  return;
}
