//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_drift.cpp
//! \brief Deterministic problem generator for the particle drift regression test.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "particles/cosmic_ray.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {

constexpr int kDefaultParticles = 8;
int expected_particles = kDefaultParticles;
bool migration_test = false;
int frozen_tag = -1;
int delete_tag = -1;

Real ParticleDriftTimestep(MeshBlockPack*) { return 0.125; }

void ParticleRadiusSquared(particles::ParticleOutputData *output) {
  auto pr = output->prtcl_rdata;
  auto values = output->output_data;
  const int field = output->field_index;
  const int npart = output->nprtcl;
  if (npart == 0) return;
  par_for("particle_output_radius_squared", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    values(field,p) = SQR(pr(IPX,p)) + SQR(pr(IPY,p)) + SQR(pr(IPZ,p));
  });
}

KOKKOS_INLINE_FUNCTION
Real InitialX(const int id, const bool migration) {
  if (migration) {
    if (id == 3) return -0.004;
    if (id == 4) return 0.004;
    return -0.70 + 0.20*id;
  }
  return -0.35 + 0.08*id;
}

KOKKOS_INLINE_FUNCTION
Real InitialY(const int id) { return -0.20 + 0.04*id; }

KOKKOS_INLINE_FUNCTION
Real InitialZ(const int id) { return -0.10 + 0.02*id; }

KOKKOS_INLINE_FUNCTION
Real VelocityX(const int id, const bool migration) {
  if (migration) {
    if (id == 3) return 0.20;
    if (id == 4) return -0.20;
    return 0.01;
  }
  return 0.10 + 0.01*id;
}

KOKKOS_INLINE_FUNCTION
Real VelocityY(const int id) { return -0.08 + 0.005*id; }

KOKKOS_INLINE_FUNCTION
Real VelocityZ(const int id) { return 0.03 - 0.002*id; }

void DriftHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 7;
  pdata->label[0] = "max_err";
  pdata->label[1] = "npart";
  pdata->label[2] = "tag_sum";
  pdata->label[3] = "owner_err";
  pdata->label[4] = "migrated";
  pdata->label[5] = "status_err";
  pdata->label[6] = "count_err";

  auto *population = pm->pmb_pack->ppart->FindPopulation("particles");
  auto pr = population->prtcl_rdata;
  auto pi = population->prtcl_idata;
  auto &mbsize = pm->pmb_pack->pmb->mb_size;
  const int npart = population->nprtcl_thispack;
  const int gids = pm->pmb_pack->gids;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const bool migration = migration_test;
  const int frozen = frozen_tag;
  const int deleted = delete_tag;
  const Real drift_time = 0.5*pm->time;

  Real max_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_max) {
        const int id = pi(particles::cosmic_ray::PTAG,p);
        const Real move_time = (id == frozen || id == deleted) ? 0.0 : drift_time;
        const Real ex = InitialX(id, migration) + move_time*VelocityX(id, migration);
        const Real ey = InitialY(id) + move_time*VelocityY(id);
        const Real ez = InitialZ(id) + move_time*VelocityZ(id);
        Real error = fabs(pr(particles::cosmic_ray::IPX,p) - ex);
        error = fmax(error, fabs(pr(particles::cosmic_ray::IPY,p) - ey));
        error = fmax(error, fabs(pr(particles::cosmic_ray::IPZ,p) - ez));
        error = fmax(error, fabs(pr(particles::cosmic_ray::IPVX,p) -
                                 VelocityX(id, migration)));
        error = fmax(error, fabs(pr(particles::cosmic_ray::IPVY,p) - VelocityY(id)));
        error = fmax(error, fabs(pr(particles::cosmic_ray::IPVZ,p) - VelocityZ(id)));
        local_max = fmax(local_max, error);
      }, Kokkos::Max<Real>(max_error));

  Real tag_sum = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_tag_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        local_sum += static_cast<Real>(pi(particles::cosmic_ray::PTAG,p));
      }, tag_sum);

  Real owner_errors = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_owner_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const int id = pi(particles::cosmic_ray::PTAG,p);
        const Real move_time = (id == frozen || id == deleted) ? 0.0 : drift_time;
        const Real ex = InitialX(id, migration) + move_time*VelocityX(id, migration);
        const Real ey = InitialY(id) + move_time*VelocityY(id);
        const Real ez = InitialZ(id) + move_time*VelocityZ(id);
        int expected_gid = -1;
        for (int m=0; m<nmb; ++m) {
          auto size = mbsize.d_view(m);
          if (ex >= size.x1min && ex < size.x1max &&
              ey >= size.x2min && ey < size.x2max &&
              ez >= size.x3min && ez < size.x3max) {
            expected_gid = gids + m;
          }
        }
        if (pi(particles::cosmic_ray::PGID,p) != expected_gid) local_sum += 1.0;
      }, owner_errors);

  Real migrated = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_migrated", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const int id = pi(particles::cosmic_ray::PTAG,p);
        const Real ix = InitialX(id, migration);
        const Real iy = InitialY(id);
        const Real iz = InitialZ(id);
        int initial_gid = -1;
        for (int m=0; m<nmb; ++m) {
          auto size = mbsize.d_view(m);
          if (ix >= size.x1min && ix < size.x1max &&
              iy >= size.x2min && iy < size.x2max &&
              iz >= size.x3min && iz < size.x3max) {
            initial_gid = gids + m;
          }
        }
        if (pi(particles::cosmic_ray::PGID,p) != initial_gid) local_sum += 1.0;
      }, migrated);

  Real status_errors = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_status_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const int id = pi(particles::cosmic_ray::PTAG,p);
        int expected_status = PACTIVE;
        if (id == frozen) expected_status = PFROZEN;
        if (id == deleted) expected_status = PDELETE_PENDING;
        if (pi(particles::cosmic_ray::PSTATUS,p) != expected_status) local_sum += 1.0;
      }, status_errors);

  pdata->hdata[0] = max_error;
  pdata->hdata[1] = static_cast<Real>(npart);
  pdata->hdata[2] = tag_sum;
  pdata->hdata[3] = owner_errors;
  pdata->hdata[4] = migrated;
  pdata->hdata[5] = status_errors;
  const int expected_total = expected_particles - (deleted >= 0 ? 1 : 0);
  pdata->hdata[6] = static_cast<Real>(
      std::abs(pm->nprtcl_thisrank - npart) +
      std::abs(pm->nprtcl_total - expected_total));
}

} // namespace

void ProblemGenerator::ParticleDrift(ParameterInput *pin, const bool restart) {
  user_hist_func = DriftHistory;
  user_particle_dt_func = ParticleDriftTimestep;
  EnrollParticleOutputVariable("radius_squared", ParticleRadiusSquared);
  expected_particles = pin->GetOrAddInteger(
      "problem", "expected_particles", kDefaultParticles);
  migration_test = pin->GetOrAddBoolean("problem", "migration_test", false);
  frozen_tag = pin->GetOrAddInteger("problem", "frozen_tag", -1);
  delete_tag = pin->GetOrAddInteger("problem", "delete_tag", -1);
  if (expected_particles < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test expected_particles must be positive."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (frozen_tag < -1 || frozen_tag >= expected_particles) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test frozen_tag must be -1 or between 0 and "
              << (expected_particles - 1) << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (delete_tag < -1 || delete_tag >= expected_particles) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test delete_tag must be -1 or between 0 and "
              << (expected_particles - 1) << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (delete_tag >= 0 && delete_tag == frozen_tag) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test cannot freeze and delete the same tag."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test requires a <particles> block."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->nprtcl_total != expected_particles) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test expected " << expected_particles
              << " particles globally, but initialized " << pmy_mesh_->nprtcl_total << "."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto *population = pmbp->ppart->FindPopulation("particles");
  auto pr = population->prtcl_rdata;
  auto pi = population->prtcl_idata;
  auto &mbsize = pmbp->pmb->mb_size;
  const int gids = pmbp->gids;
  const int nmb = pmbp->nmb_thispack;
  const int npart = population->nprtcl_thispack;
  const bool migration = migration_test;
  const int frozen = frozen_tag;
  const int deleted = delete_tag;
  par_for("particle_drift_init", DevExeSpace(), 0, (npart - 1),
  KOKKOS_LAMBDA(const int p) {
    const int id = pi(particles::cosmic_ray::PTAG,p);
    const Real x = InitialX(id, migration);
    const Real y = InitialY(id);
    const Real z = InitialZ(id);
    int owner_gid = gids;
    for (int m=0; m<nmb; ++m) {
      auto size = mbsize.d_view(m);
      if (x >= size.x1min && x < size.x1max &&
          y >= size.x2min && y < size.x2max &&
          z >= size.x3min && z < size.x3max) {
        owner_gid = gids + m;
      }
    }
    pi(particles::cosmic_ray::PGID,p) = owner_gid;
    pi(particles::cosmic_ray::PSTATUS,p) = PACTIVE;
    if (id == frozen) pi(particles::cosmic_ray::PSTATUS,p) = PFROZEN;
    if (id == deleted) pi(particles::cosmic_ray::PSTATUS,p) = PDELETE_PENDING;
    pr(particles::cosmic_ray::IPX,p) = x;
    pr(particles::cosmic_ray::IPY,p) = y;
    pr(particles::cosmic_ray::IPZ,p) = z;
    pr(particles::cosmic_ray::IPVX,p) = VelocityX(id, migration);
    pr(particles::cosmic_ray::IPVY,p) = VelocityY(id);
    pr(particles::cosmic_ray::IPVZ,p) = VelocityZ(id);
  });
}
