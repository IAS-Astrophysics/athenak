//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_lagrangian_mc.cpp
//! \brief Deterministic problem generator for the Lagrangian MC particle test.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {

constexpr int kExpectedParticles = 8;
constexpr int kRandomSeed = 5;

KOKKOS_INLINE_FUNCTION
int CellForTag(const int tag) {
  if (tag == 1) return 3;
  if (tag == 3) return 1;
  return tag;
}

KOKKOS_INLINE_FUNCTION
bool ExpectedMove(const int tag) {
  // With seed 5, the per-tag SplitMix64 draws select these tags when the correct RK2
  // outgoing probabilities are 0.48 in even cells and 0.36 in odd cells.
  return tag == 0 || tag == 1 || tag == 2 || tag == 6;
}

KOKKOS_INLINE_FUNCTION
Real InitialDensity(const int i) { return (i % 2 == 0) ? 1.0 : 2.0; }

KOKKOS_INLINE_FUNCTION
Real FinalDensity(const int i) { return (i % 2 == 0) ? 1.24 : 1.76; }

KOKKOS_INLINE_FUNCTION
Real InitialX(const int tag) {
  return (static_cast<Real>(CellForTag(tag)) + 0.5)/kExpectedParticles;
}

void LagrangianMCHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 10;
  pdata->label[0] = "fluid_err";
  pdata->label[1] = "flux_err";
  pdata->label[2] = "pos_err";
  pdata->label[3] = "owner_err";
  pdata->label[4] = "moved";
  pdata->label[5] = "moved_tags";
  pdata->label[6] = "migrated";
  pdata->label[7] = "npart";
  pdata->label[8] = "tag_sum";
  pdata->label[9] = "status_err";

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;
  const bool final_state = pm->ncycle > 0;
  auto u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;

  Real fluid_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_fluid_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &local_max) {
        const int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        const int i = (idx - m*nkji - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        auto size = mbsize.d_view(m);
        const Real x = CellCenterX(i-is, nx1, size.x1min, size.x1max);
        int global_i = static_cast<int>(x*kExpectedParticles);
        if (global_i == kExpectedParticles) global_i = kExpectedParticles - 1;
        const Real expected = final_state ? FinalDensity(global_i) :
                                            InitialDensity(global_i);
        local_max = fmax(local_max, fabs(u0(m,IDN,k,j,i) - expected));
      }, Kokkos::Max<Real>(fluid_error));

  auto intflx1 = pmbp->phydro->density_flux_integral.x1f;
  Real flux_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_flux_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &local_max) {
        const int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        const int i = (idx - m*nkji - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        auto size = mbsize.d_view(m);
        const Real x = CellCenterX(i-is, nx1, size.x1min, size.x1max);
        int global_i = static_cast<int>(x*kExpectedParticles);
        if (global_i == kExpectedParticles) global_i = kExpectedParticles - 1;
        Real expected = 0.0;
        if (final_state) expected = (global_i % 2 == 0) ? 0.48 : 0.72;
        local_max = fmax(local_max, fabs(intflx1(m,k,j,i+1) - expected));
      }, Kokkos::Max<Real>(flux_error));

  auto particles = pmbp->ppart;
  auto pr = particles->prtcl_rdata;
  auto pi = particles->prtcl_idata;
  const int npart = particles->nprtcl_thispack;
  const int gids = pmbp->gids;
  const int nmb = pmbp->nmb_thispack;
  constexpr Real y0 = 0.125;
  constexpr Real z0 = 0.5;
  constexpr Real dx = 1.0/kExpectedParticles;

  Real position_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_position_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_max) {
        const int tag = pi(PTAG,p);
        Real expected_x = InitialX(tag);
        if (final_state && ExpectedMove(tag)) expected_x += dx;
        Real error = fabs(pr(IPX,p) - expected_x);
        error = fmax(error, fabs(pr(IPY,p) - y0));
        error = fmax(error, fabs(pr(IPZ,p) - z0));
        local_max = fmax(local_max, error);
      }, Kokkos::Max<Real>(position_error));

  Real owner_errors = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_owner_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const int tag = pi(PTAG,p);
        Real expected_x = InitialX(tag);
        if (final_state && ExpectedMove(tag)) expected_x += dx;
        int expected_gid = -1;
        for (int m=0; m<nmb; ++m) {
          auto size = mbsize.d_view(m);
          if (expected_x >= size.x1min && expected_x < size.x1max &&
              y0 >= size.x2min && y0 < size.x2max &&
              z0 >= size.x3min && z0 < size.x3max) {
            expected_gid = gids + m;
          }
        }
        if (pi(PGID,p) != expected_gid) local_sum += 1.0;
      }, owner_errors);

  Real moved = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_moved", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        if (fabs(pr(IPX,p) - InitialX(pi(PTAG,p))) > 0.5*dx) local_sum += 1.0;
      }, moved);

  Real moved_tag_sum = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_moved_tag_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const int tag = pi(PTAG,p);
        if (fabs(pr(IPX,p) - InitialX(tag)) > 0.5*dx) {
          local_sum += static_cast<Real>(tag);
        }
      }, moved_tag_sum);

  Real migrated = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_migrated", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const Real initial_x = InitialX(pi(PTAG,p));
        int initial_gid = -1;
        for (int m=0; m<nmb; ++m) {
          auto size = mbsize.d_view(m);
          if (initial_x >= size.x1min && initial_x < size.x1max &&
              y0 >= size.x2min && y0 < size.x2max &&
              z0 >= size.x3min && z0 < size.x3max) {
            initial_gid = gids + m;
          }
        }
        if (pi(PGID,p) != initial_gid) local_sum += 1.0;
      }, migrated);

  Real tag_sum = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_tag_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        local_sum += static_cast<Real>(pi(PTAG,p));
      }, tag_sum);

  Real status_errors = 0.0;
  Kokkos::parallel_reduce(
      "particle_lmc_status_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        if (pi(PSTATUS,p) != PACTIVE) local_sum += 1.0;
      }, status_errors);

  pdata->hdata[0] = fluid_error;
  pdata->hdata[1] = flux_error;
  pdata->hdata[2] = position_error;
  pdata->hdata[3] = owner_errors;
  pdata->hdata[4] = moved;
  pdata->hdata[5] = moved_tag_sum;
  pdata->hdata[6] = migrated;
  pdata->hdata[7] = static_cast<Real>(npart);
  pdata->hdata[8] = tag_sum;
  pdata->hdata[9] = status_errors;
}

} // namespace

void ProblemGenerator::ParticleLagrangianMC(ParameterInput *pin, const bool restart) {
  user_hist_func = LagrangianMCHistory;
  if (restart) return;

  if (pin->GetInteger("particles", "random_seed") != kRandomSeed) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC test requires particles/random_seed = "
              << kRandomSeed << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC test requires <hydro> and <particles> blocks."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->nprtcl_total != kExpectedParticles) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC test expected " << kExpectedParticles
              << " particles globally, but initialized " << pmy_mesh_->nprtcl_total << "."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  auto &mbsize = pmbp->pmb->mb_size;
  auto u0 = pmbp->phydro->u0;
  par_for("particle_lmc_fluid_init", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    auto size = mbsize.d_view(m);
    const Real x = CellCenterX(i-is, indcs.nx1, size.x1min, size.x1max);
    int global_i = static_cast<int>(x*kExpectedParticles);
    if (global_i == kExpectedParticles) global_i = kExpectedParticles - 1;
    const Real density = InitialDensity(global_i);
    u0(m,IDN,k,j,i) = density;
    u0(m,IM1,k,j,i) = density;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
  });

  auto particles = pmbp->ppart;
  auto pr = particles->prtcl_rdata;
  auto pi = particles->prtcl_idata;
  const int npart = particles->nprtcl_thispack;
  const int gids = pmbp->gids;
  const int nmb = pmbp->nmb_thispack;
  par_for("particle_lmc_init", DevExeSpace(), 0, (npart - 1),
  KOKKOS_LAMBDA(const int p) {
    const int tag = pi(PTAG,p);
    const Real x = InitialX(tag);
    constexpr Real y = 0.125;
    constexpr Real z = 0.5;
    int owner_gid = -1;
    for (int m=0; m<nmb; ++m) {
      auto size = mbsize.d_view(m);
      if (x >= size.x1min && x < size.x1max &&
          y >= size.x2min && y < size.x2max &&
          z >= size.x3min && z < size.x3max) {
        owner_gid = gids + m;
      }
    }
    pi(PGID,p) = owner_gid;
    pi(PSTATUS,p) = PACTIVE;
    pr(IPX,p) = x;
    pr(IPY,p) = y;
    pr(IPZ,p) = z;
  });
}
