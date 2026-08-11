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
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {

constexpr int kExpectedParticles = 8;
bool migration_test = false;

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
  pdata->nhist = 5;
  pdata->label[0] = "max_err";
  pdata->label[1] = "npart";
  pdata->label[2] = "tag_sum";
  pdata->label[3] = "owner_err";
  pdata->label[4] = "migrated";

  auto &particles = pm->pmb_pack->ppart;
  auto pr = particles->prtcl_rdata;
  auto pi = particles->prtcl_idata;
  auto &mbsize = pm->pmb_pack->pmb->mb_size;
  const int npart = particles->nprtcl_thispack;
  const int gids = pm->pmb_pack->gids;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const bool migration = migration_test;
  const Real drift_time = 0.5*pm->time;

  Real max_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_max) {
        const int id = pi(PTAG,p);
        const Real ex = InitialX(id, migration) + drift_time*VelocityX(id, migration);
        const Real ey = InitialY(id) + drift_time*VelocityY(id);
        const Real ez = InitialZ(id) + drift_time*VelocityZ(id);
        Real error = fabs(pr(IPX,p) - ex);
        error = fmax(error, fabs(pr(IPY,p) - ey));
        error = fmax(error, fabs(pr(IPZ,p) - ez));
        local_max = fmax(local_max, error);
      }, Kokkos::Max<Real>(max_error));

  Real tag_sum = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_tag_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        local_sum += static_cast<Real>(pi(PTAG,p));
      }, tag_sum);

  Real owner_errors = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_owner_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const int id = pi(PTAG,p);
        const Real ex = InitialX(id, migration) + drift_time*VelocityX(id, migration);
        const Real ey = InitialY(id) + drift_time*VelocityY(id);
        const Real ez = InitialZ(id) + drift_time*VelocityZ(id);
        int expected_gid = -1;
        for (int m=0; m<nmb; ++m) {
          auto size = mbsize.d_view(m);
          if (ex >= size.x1min && ex < size.x1max &&
              ey >= size.x2min && ey < size.x2max &&
              ez >= size.x3min && ez < size.x3max) {
            expected_gid = gids + m;
          }
        }
        if (pi(PGID,p) != expected_gid) local_sum += 1.0;
      }, owner_errors);

  Real migrated = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_migrated", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const int id = pi(PTAG,p);
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
        if (pi(PGID,p) != initial_gid) local_sum += 1.0;
      }, migrated);

  pdata->hdata[0] = max_error;
  pdata->hdata[1] = static_cast<Real>(npart);
  pdata->hdata[2] = tag_sum;
  pdata->hdata[3] = owner_errors;
  pdata->hdata[4] = migrated;
}

} // namespace

void ProblemGenerator::ParticleDrift(ParameterInput *pin, const bool restart) {
  user_hist_func = DriftHistory;
  migration_test = pin->GetOrAddBoolean("problem", "migration_test", false);
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test requires a <particles> block."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->nprtcl_total != kExpectedParticles) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test expected " << kExpectedParticles
              << " particles globally, but initialized " << pmy_mesh_->nprtcl_total << "."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto pr = pmbp->ppart->prtcl_rdata;
  auto pi = pmbp->ppart->prtcl_idata;
  auto &mbsize = pmbp->pmb->mb_size;
  const int gids = pmbp->gids;
  const int nmb = pmbp->nmb_thispack;
  const int npart = pmbp->ppart->nprtcl_thispack;
  const bool migration = migration_test;
  par_for("particle_drift_init", DevExeSpace(), 0, (npart - 1),
  KOKKOS_LAMBDA(const int p) {
    const int id = pi(PTAG,p);
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
    pi(PGID,p) = owner_gid;
    pr(IPX,p) = x;
    pr(IPY,p) = y;
    pr(IPZ,p) = z;
    pr(IPVX,p) = VelocityX(id, migration);
    pr(IPVY,p) = VelocityY(id);
    pr(IPVZ,p) = VelocityZ(id);
  });

  pmbp->ppart->dtnew = 0.125;
}
