//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_smr.cpp
//! \brief Deterministic problem generator for particle SMR routing tests.

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

Real particle_x = 0.0;
Real particle_y = 0.0;
Real particle_z = 0.0;
Real particle_vx = 0.0;
Real particle_vy = 0.0;
Real particle_vz = 0.0;

Real ParticleSMRTimestep(MeshBlockPack*) { return 0.125; }

void InitialParticleInjection(MeshBlockPack *pmbp,
                              particles::ParticleCreation *creation) {
  auto *population = creation->GetPopulation();
  if (population->particle_type != ParticleType::cosmic_ray) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle SMR test requires cosmic-ray particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int owner_gid = -1;
  auto &mbsize = pmbp->pmb->mb_size;
  for (int m=0; m<pmbp->nmb_thispack; ++m) {
    auto size = mbsize.h_view(m);
    if (particle_x >= size.x1min && particle_x < size.x1max &&
        particle_y >= size.x2min && particle_y < size.x2max &&
        particle_z >= size.x3min && particle_z < size.x3max) {
      if (owner_gid >= 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Particle SMR test found more than one local owner"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      owner_gid = pmbp->gids + m;
    }
  }

  creation->Resize(owner_gid >= 0 ? 1 : 0);
  if (owner_gid < 0) return;

  auto pr = creation->prtcl_rdata;
  auto pi = creation->prtcl_idata;
  const int gid = owner_gid;
  const Real x = particle_x;
  const Real y = particle_y;
  const Real z = particle_z;
  const Real vx = particle_vx;
  const Real vy = particle_vy;
  const Real vz = particle_vz;
  par_for("particle_smr_init", DevExeSpace(), 0, 0,
  KOKKOS_LAMBDA(const int p) {
    pi(particles::cosmic_ray::PGID,p) = gid;
    pr(particles::cosmic_ray::IPX,p) = x;
    pr(particles::cosmic_ray::IPY,p) = y;
    pr(particles::cosmic_ray::IPZ,p) = z;
    pr(particles::cosmic_ray::IPVX,p) = vx;
    pr(particles::cosmic_ray::IPVY,p) = vy;
    pr(particles::cosmic_ray::IPVZ,p) = vz;
  });
}

void ParticleOwnerError(particles::ParticleOutputData *output) {
  auto pr = output->prtcl_rdata;
  auto pi = output->prtcl_idata;
  auto values = output->output_data;
  auto &mbsize = output->pmbp->pmb->mb_size;
  const int field = output->field_index;
  const int npart = output->nprtcl;
  const int gids = output->pmbp->gids;
  const int nmb = output->pmbp->nmb_thispack;
  if (npart == 0) return;
  par_for("particle_smr_owner_error", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    const int m = pi(particles::cosmic_ray::PGID,p) - gids;
    Real error = 1.0;
    if (m >= 0 && m < nmb) {
      auto size = mbsize.d_view(m);
      if (pr(particles::cosmic_ray::IPX,p) >= size.x1min &&
          pr(particles::cosmic_ray::IPX,p) < size.x1max &&
          pr(particles::cosmic_ray::IPY,p) >= size.x2min &&
          pr(particles::cosmic_ray::IPY,p) < size.x2max &&
          pr(particles::cosmic_ray::IPZ,p) >= size.x3min &&
          pr(particles::cosmic_ray::IPZ,p) < size.x3max) {
        error = 0.0;
      }
    }
    values(field,p) = error;
  });
}

void ParticleOwnerLevel(particles::ParticleOutputData *output) {
  auto pi = output->prtcl_idata;
  auto values = output->output_data;
  auto &mblev = output->pmbp->pmb->mb_lev;
  const int field = output->field_index;
  const int npart = output->nprtcl;
  const int gids = output->pmbp->gids;
  const int nmb = output->pmbp->nmb_thispack;
  const int root_level = output->pmbp->pmesh->root_level;
  if (npart == 0) return;
  par_for("particle_smr_owner_level", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    const int m = pi(particles::cosmic_ray::PGID,p) - gids;
    values(field,p) = (m >= 0 && m < nmb) ? mblev.d_view(m) - root_level : -1.0;
  });
}

void ParticleOwnerRank(particles::ParticleOutputData *output) {
  auto values = output->output_data;
  const int field = output->field_index;
  const int npart = output->nprtcl;
  const int rank = global_variable::my_rank;
  if (npart == 0) return;
  par_for("particle_smr_owner_rank", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    values(field,p) = rank;
  });
}

} // namespace

void ProblemGenerator::ParticleSMR(ParameterInput *pin, const bool restart) {
  user_particle_dt_func = ParticleSMRTimestep;
  user_initial_particle_injection_func = InitialParticleInjection;
  EnrollParticleVTKOutputVariable("owner_error", ParticleOwnerError);
  EnrollParticleVTKOutputVariable("owner_level", ParticleOwnerLevel);
  EnrollParticleVTKOutputVariable("owner_rank", ParticleOwnerRank);

  particle_x = pin->GetOrAddReal("problem", "particle_x", 0.0);
  particle_y = pin->GetOrAddReal("problem", "particle_y", 0.0);
  particle_z = pin->GetOrAddReal("problem", "particle_z", 0.0);
  particle_vx = pin->GetOrAddReal("problem", "particle_vx", 0.0);
  particle_vy = pin->GetOrAddReal("problem", "particle_vy", 0.0);
  particle_vz = pin->GetOrAddReal("problem", "particle_vz", 0.0);

  if (!pmy_mesh_->multilevel || pmy_mesh_->adaptive) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle SMR test requires static mesh refinement"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle SMR test requires a <particles> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *population = pmbp->ppart->FindPopulation("particles");
  if (population == nullptr || population->particle_type != ParticleType::cosmic_ray) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle SMR test requires cosmic-ray particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &meshsize = pmy_mesh_->mesh_size;
  if (!(particle_x >= meshsize.x1min && particle_x < meshsize.x1max &&
        particle_y >= meshsize.x2min && particle_y < meshsize.x2max &&
        particle_z >= meshsize.x3min && particle_z < meshsize.x3max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle SMR test position must lie inside the mesh"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!restart && pmy_mesh_->nprtcl_total != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle SMR test must start without particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
