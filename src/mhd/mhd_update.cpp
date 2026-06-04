//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_update.cpp
//! \brief Performs explicit update of MHD conserved variables (u0) for each stage of the
//! SSP RK integrators (e.g. RK1, RK2, RK3) implemented in AthenaK, using weighted average
//! and partial time update of flux divergence. Source terms are added in the
//! MHDSrcTerms() function.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

namespace mhd {
namespace {

bool FluxSymmetryDebugEnabled() {
  static bool enabled = (std::getenv("ATHENA_SYM_DEBUG") != nullptr) ||
                        (std::getenv("ATHENA_FLUX_DEBUG") != nullptr);
  return enabled;
}

Real EnvReal(const char *name, Real default_value) {
  const char *value = std::getenv(name);
  return (value == nullptr) ? default_value : static_cast<Real>(std::atof(value));
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  void MHD::Update
//  \brief Explicit RK update including flux divergence terms

TaskStatus MHD::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nv1 = nmhd + nscalars - 1;
  auto u0_ = u0;
  auto u1_ = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator used
  // Vector inner loop used for good performance on cpus
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("mhd_update",DevExeSpace(),scr_size,scr_level,0,nmb1,0,nv1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

    // compute dF1/dx1
    par_for_inner(member, is, ie, [&](const int i) {
      divf(i) = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
    });
    member.team_barrier();

    // Add dF2/dx2
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (multi_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
      });
      member.team_barrier();
    }

    // Add dF3/dx3
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
      });
      member.team_barrier();
    }

    par_for_inner(member, is, ie, [&](const int i) {
      u0_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - beta_dt*divf(i);
    });
  });
  SymmetryDebugProbe("MHD_RKUpdate", pdriver, stage);
  SymmetryRKDebugProbe("MHD_RKUpdateDiv", pdriver, stage, beta_dt);
  return TaskStatus::complete;
}

void MHD::SymmetryRKDebugProbe(const char *label, Driver *pdriver, int stage, Real beta_dt) {
  if (!FluxSymmetryDebugEnabled()) return;
  if (pdriver == nullptr) return;

  Mesh *pm = pmy_pack->pmesh;
  const int cycle = pm->ncycle;
  if (!(cycle < 3 || cycle == 80 || cycle == 160)) return;

  auto &indcs = pm->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &gid = pmy_pack->pmb->mb_gid;
  auto &lev = pmy_pack->pmb->mb_lev;
  const bool multi_d = pm->multi_d;
  const bool three_d = pm->three_d;
  const Real x_target = EnvReal("ATHENA_SYM_X_TARGET", 20.0);
  const Real z_target = EnvReal("ATHENA_SYM_Z_TARGET", 0.0);

  for (int side = 0; side < 2; ++side) {
    const bool below = (side == 0);
    int best_m = -1;
    int best_i = -1;
    int best_j = -1;
    int best_k = -1;
    Real best_abs_y = std::numeric_limits<Real>::max();
    Real best_x = 0.0;
    Real best_y = 0.0;
    Real best_z = 0.0;

    for (int m = 0; m < pmy_pack->nmb_thispack; ++m) {
      const auto &mb = size.h_view(m);
      const bool x_inside = (x_target >= mb.x1min) && (x_target < mb.x1max);
      const bool z_inside = (z_target >= mb.x3min) && (z_target < mb.x3max);
      if (!(x_inside && z_inside)) continue;

      int i = indcs.is + static_cast<int>(std::floor((x_target - mb.x1min)/mb.dx1));
      i = std::max(indcs.is, std::min(indcs.ie, i));
      int k = indcs.ks + static_cast<int>(std::floor((z_target - mb.x3min)/mb.dx3));
      k = std::max(indcs.ks, std::min(indcs.ke, k));

      for (int jj = indcs.js; jj <= indcs.je; ++jj) {
        const Real y = mb.x2min + (static_cast<Real>(jj - indcs.js) + 0.5)*mb.dx2;
        if (((below && y < 0.0) || (!below && y > 0.0)) &&
            (std::abs(y) < best_abs_y)) {
          best_abs_y = std::abs(y);
          best_m = m;
          best_i = i;
          best_j = jj;
          best_k = k;
          best_x = mb.x1min + (static_cast<Real>(i - indcs.is) + 0.5)*mb.dx1;
          best_y = y;
          best_z = mb.x3min + (static_cast<Real>(k - indcs.ks) + 0.5)*mb.dx3;
        }
      }
    }
    if (best_m < 0) continue;

    DvceArray1D<Real> diag("rk-symmetry-debug", 7);
    auto u0_ = u0;
    auto u1_ = u1;
    auto flx1 = uflx.x1f;
    auto flx2 = uflx.x2f;
    auto flx3 = uflx.x3f;
    auto &mbsize = pmy_pack->pmb->mb_size;
    const int m = best_m;
    const int i = best_i;
    const int j = best_j;
    const int k = best_k;
    Kokkos::parallel_for("mhd_rk_symmetry_debug",
                         Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
                         KOKKOS_LAMBDA(const int) {
      const Real dfx1 = (flx1(m,IDN,k,j,i+1) - flx1(m,IDN,k,j,i))/mbsize.d_view(m).dx1;
      const Real dfx2 = multi_d ?
          (flx2(m,IDN,k,j+1,i) - flx2(m,IDN,k,j,i))/mbsize.d_view(m).dx2 : 0.0;
      const Real dfx3 = three_d ?
          (flx3(m,IDN,k+1,j,i) - flx3(m,IDN,k,j,i))/mbsize.d_view(m).dx3 : 0.0;
      diag(0) = dfx1;
      diag(1) = dfx2;
      diag(2) = dfx3;
      diag(3) = dfx1 + dfx2 + dfx3;
      diag(4) = u1_(m,IDN,k,j,i);
      diag(5) = u0_(m,IDN,k,j,i);
      diag(6) = -beta_dt*(dfx1 + dfx2 + dfx3);
    });
    Kokkos::fence();
    auto hdiag = Kokkos::create_mirror_view(diag);
    Kokkos::deep_copy(hdiag, diag);

    std::cout << std::setprecision(17)
              << "RKDBG label=" << label
              << " rank=" << global_variable::my_rank
              << " gid=" << gid.h_view(m)
              << " level=" << lev.h_view(m)
              << " side=" << (below ? "below" : "above")
              << " cycle=" << cycle
              << " time=" << pm->time
              << " stage=" << stage
              << " i=" << i
              << " j=" << j
              << " k=" << k
              << " x=" << best_x
              << " y=" << best_y
              << " z=" << best_z
              << " dF1dx1=" << hdiag(0)
              << " dF2dx2=" << hdiag(1)
              << " dF3dx3=" << hdiag(2)
              << " divf=" << hdiag(3)
              << " u_saved=" << hdiag(4)
              << " u_new=" << hdiag(5)
              << " flux_delta=" << hdiag(6)
              << std::endl;
  }
}
} // namespace mhd
