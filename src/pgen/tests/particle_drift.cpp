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

KOKKOS_INLINE_FUNCTION
Real InitialX(const int id) { return -0.35 + 0.08*id; }

KOKKOS_INLINE_FUNCTION
Real InitialY(const int id) { return -0.20 + 0.04*id; }

KOKKOS_INLINE_FUNCTION
Real InitialZ(const int id) { return -0.10 + 0.02*id; }

KOKKOS_INLINE_FUNCTION
Real VelocityX(const int id) { return 0.10 + 0.01*id; }

KOKKOS_INLINE_FUNCTION
Real VelocityY(const int id) { return -0.08 + 0.005*id; }

KOKKOS_INLINE_FUNCTION
Real VelocityZ(const int id) { return 0.03 - 0.002*id; }

void DriftHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 3;
  pdata->label[0] = "max_err";
  pdata->label[1] = "npart";
  pdata->label[2] = "tag_sum";

  auto &particles = pm->pmb_pack->ppart;
  auto pr = particles->prtcl_rdata;
  auto pi = particles->prtcl_idata;
  const int npart = particles->nprtcl_thispack;
  const Real drift_time = 0.5*pm->time;

  Real max_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_drift_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_max) {
        const int id = pi(PTAG,p);
        const Real ex = InitialX(id) + drift_time*VelocityX(id);
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

  pdata->hdata[0] = max_error;
  pdata->hdata[1] = static_cast<Real>(npart);
  pdata->hdata[2] = tag_sum;
}

} // namespace

void ProblemGenerator::ParticleDrift(ParameterInput *pin, const bool restart) {
  user_hist_func = DriftHistory;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test requires a <particles> block."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->ppart->nprtcl_thispack != kExpectedParticles) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle drift test expected " << kExpectedParticles
              << " particles, but initialized " << pmbp->ppart->nprtcl_thispack << "."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto pr = pmbp->ppart->prtcl_rdata;
  auto pi = pmbp->ppart->prtcl_idata;
  const int gid = pmbp->gids;
  par_for("particle_drift_init", DevExeSpace(), 0, (kExpectedParticles - 1),
  KOKKOS_LAMBDA(const int p) {
    pi(PGID,p) = gid;
    pr(IPX,p) = InitialX(p);
    pr(IPY,p) = InitialY(p);
    pr(IPZ,p) = InitialZ(p);
    pr(IPVX,p) = VelocityX(p);
    pr(IPVY,p) = VelocityY(p);
    pr(IPVZ,p) = VelocityZ(p);
  });

  pmbp->ppart->dtnew = 0.125;
}
