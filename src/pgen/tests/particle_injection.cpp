//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_injection.cpp
//! \brief Problem generator for particle creation regression tests.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "particles/cosmic_ray.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {

int injection_cycle = 1;

Real ParticleInjectionTimestep(MeshBlockPack*) { return 0.125; }

void FillCreationBatch(MeshBlockPack *pmbp, particles::ParticleCreation *creation,
                       bool rank_zero_only) {
  auto *population = creation->GetPopulation();
  if (population->particle_type != ParticleType::cosmic_ray) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle injection test requires cosmic-ray particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int nnew = (rank_zero_only && global_variable::my_rank != 0) ? 0 : 1;
  creation->Resize(nnew);
  if (nnew == 0) return;

  auto pr = creation->prtcl_rdata;
  auto pi = creation->prtcl_idata;
  auto &mbsize = pmbp->pmb->mb_size;
  const int gid = pmbp->gids;
  par_for("particle_injection_fill", DevExeSpace(), 0, 0,
  KOKKOS_LAMBDA(const int p) {
    pi(particles::cosmic_ray::PGID,p) = gid;
    pr(particles::cosmic_ray::IPX,p) =
        0.5*(mbsize.d_view(0).x1min + mbsize.d_view(0).x1max);
    pr(particles::cosmic_ray::IPY,p) =
        0.5*(mbsize.d_view(0).x2min + mbsize.d_view(0).x2max);
    pr(particles::cosmic_ray::IPZ,p) =
        0.5*(mbsize.d_view(0).x3min + mbsize.d_view(0).x3max);
    pr(particles::cosmic_ray::IPVX,p) = 0.2;
    pr(particles::cosmic_ray::IPVY,p) = 0.0;
    pr(particles::cosmic_ray::IPVZ,p) = 0.0;
  });
}

void InitialParticleInjection(MeshBlockPack *pmbp,
                              particles::ParticleCreation *creation) {
  FillCreationBatch(pmbp, creation, true);
}

void RuntimeParticleInjection(MeshBlockPack *pmbp,
                              particles::ParticleCreation *creation) {
  if (pmbp->pmesh->ncycle == injection_cycle) {
    FillCreationBatch(pmbp, creation, false);
  }
}

} // namespace

void ProblemGenerator::ParticleInjection(ParameterInput *pin, const bool restart) {
  user_particle_dt_func = ParticleInjectionTimestep;
  user_initial_particle_injection_func = InitialParticleInjection;
  user_particle_injection_func = RuntimeParticleInjection;
  injection_cycle = pin->GetOrAddInteger("problem", "injection_cycle", 1);
  if (injection_cycle < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle injection test cycle must be positive"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle injection test requires a <particles> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!restart && pmy_mesh_->nprtcl_total != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle injection test must start without particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
