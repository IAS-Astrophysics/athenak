//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrangian_mc.cpp
//! \brief implementation of the Lagrangian Monte Carlo particle pusher

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "particles/lagrangian_mc.hpp"
#include "particles/particles.hpp"

namespace particles {
namespace {

KOKKOS_INLINE_FUNCTION
std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

KOKKOS_INLINE_FUNCTION
Real LagrangianMCRandom(const std::uint64_t seed, const int tag, const int cycle) {
  // Key draws by durable state so particle reordering and MPI migration do not alter them.
  std::uint64_t key = seed;
  key ^= static_cast<std::uint64_t>(tag) * 0xd2b74407b1ce6e93ULL;
  key ^= static_cast<std::uint64_t>(cycle) * 0x9e3779b97f4a7c15ULL;
  const std::uint64_t bits = SplitMix64(key);
#if SINGLE_PRECISION_ENABLED
  return static_cast<Real>(bits >> 40) * (1.0f/16777216.0f);
#else
  return static_cast<Real>(bits >> 11) * (1.0/9007199254740992.0);
#endif
}

} // namespace
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::PushLagrangianMC
//! \brief Validate the outgoing mass budget before moving Lagrangian MC particles.

TaskStatus Particles::PushLagrangianMC(Driver*, int) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmkji = pmy_pack->nmb_thispack*nkji;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const Real invalid_probability = std::numeric_limits<Real>::max();
  auto start_u = pmy_pack->phydro->u1;
  auto intflx1 = pmy_pack->phydro->density_flux_integral.x1f;
  auto intflx2 = pmy_pack->phydro->density_flux_integral.x2f;
  auto intflx3 = pmy_pack->phydro->density_flux_integral.x3f;
  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  MaxLoc::value_type max_probability;

  // A one-hop Monte Carlo move requires total outgoing mass to be no larger than the
  // cell's beginning-of-step mass.
  Kokkos::parallel_reduce(
      "particle_lmc_check_flux_probability",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &local_max) {
        const int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        const int i = (idx - m*nkji - k*nji - j*nx1) + is;
        j += js;
        k += ks;

        const Real start_density = start_u(m,IDN,k,j,i);
        const Real x1l = -intflx1(m,k,j,i);
        const Real x1r =  intflx1(m,k,j,i+1);
        const Real x2l = multi_d ? -intflx2(m,k,j,i) : 0.0;
        const Real x2r = multi_d ?  intflx2(m,k,j+1,i) : 0.0;
        const Real x3l = three_d ? -intflx3(m,k,j,i) : 0.0;
        const Real x3r = three_d ?  intflx3(m,k+1,j,i) : 0.0;
        Real probability = 0.0;
        if (start_density < 0.0 || !Kokkos::isfinite(start_density) ||
            !Kokkos::isfinite(x1l) || !Kokkos::isfinite(x1r) ||
            !Kokkos::isfinite(x2l) || !Kokkos::isfinite(x2r) ||
            !Kokkos::isfinite(x3l) || !Kokkos::isfinite(x3r)) {
          probability = invalid_probability;
        } else {
          const Real outgoing = fmax(x1l, 0.0) + fmax(x1r, 0.0) +
                                fmax(x2l, 0.0) + fmax(x2r, 0.0) +
                                fmax(x3l, 0.0) + fmax(x3r, 0.0);
          if (start_density > 0.0) {
            probability = outgoing/start_density;
          } else if (outgoing > 0.0) {
            probability = invalid_probability;
          }
        }
        if (probability > local_max.val) {
          local_max.val = probability;
          local_max.loc = idx;
        }
      }, MaxLoc(max_probability));

  const Real tolerance = 64.0*std::numeric_limits<Real>::epsilon();
  if (!(max_probability.val <= 1.0 + tolerance)) {
    const int m = max_probability.loc/nkji;
    int k = (max_probability.loc - m*nkji)/nji;
    int j = (max_probability.loc - m*nkji - k*nji)/nx1;
    const int i = (max_probability.loc - m*nkji - k*nji - j*nx1) + is;
    j += js;
    k += ks;
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid Lagrangian MC outgoing probability "
              << std::setprecision(17) << max_probability.val << " in gid="
              << pmy_pack->gids + m << " cell (k,j,i)=(" << k << "," << j << ","
              << i << "). Probability must be finite and must not exceed one; "
              << "the beginning-of-step density must be non-negative." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int gids = pmy_pack->gids;
  const int npart = nprtcl_thispack;
  const int cycle = pmy_pack->pmesh->ncycle;
  const std::uint64_t seed = lmc_random_seed;

  par_for("particle_lmc_move", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    if (pi(PSTATUS,p) != PACTIVE) return;

    const int m = pi(PGID,p) - gids;
    auto size = mbsize.d_view(m);
    const int i = static_cast<int>((pr(IPX,p) - size.x1min)/size.dx1) + is;
    const int j = multi_d ?
        static_cast<int>((pr(IPY,p) - size.x2min)/size.dx2) + js : js;
    const int k = three_d ?
        static_cast<int>((pr(IPZ,p) - size.x3min)/size.dx3) + ks : ks;
    const Real start_density = start_u(m,IDN,k,j,i);

    pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_NONE;
    if (start_density == 0.0) return;

    const Real x1l = fmax(-intflx1(m,k,j,i), 0.0)/start_density;
    const Real x1r = fmax( intflx1(m,k,j,i+1), 0.0)/start_density;
    const Real x2l = multi_d ? fmax(-intflx2(m,k,j,i), 0.0)/start_density : 0.0;
    const Real x2r = multi_d ? fmax( intflx2(m,k,j+1,i), 0.0)/start_density : 0.0;
    const Real x3l = three_d ? fmax(-intflx3(m,k,j,i), 0.0)/start_density : 0.0;
    const Real x3r = three_d ? fmax( intflx3(m,k+1,j,i), 0.0)/start_density : 0.0;
    const Real draw = LagrangianMCRandom(seed, pi(PTAG,p), cycle);

    if (draw < x1l) {
      pr(IPX,p) -= size.dx1;
      pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_X1_LEFT;
    } else if (draw < x1l + x1r) {
      pr(IPX,p) += size.dx1;
      pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_X1_RIGHT;
    } else if (draw < x1l + x1r + x2l) {
      pr(IPY,p) -= size.dx2;
      pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_X2_LEFT;
    } else if (draw < x1l + x1r + x2l + x2r) {
      pr(IPY,p) += size.dx2;
      pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_X2_RIGHT;
    } else if (draw < x1l + x1r + x2l + x2r + x3l) {
      pr(IPZ,p) -= size.dx3;
      pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_X3_LEFT;
    } else if (draw < x1l + x1r + x2l + x2r + x3l + x3r) {
      pr(IPZ,p) += size.dx3;
      pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_X3_RIGHT;
    }
  });

  return TaskStatus::complete;
}
} // namespace particles
