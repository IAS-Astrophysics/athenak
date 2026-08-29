//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrangian_mc.cpp
//! \brief implementation of the Lagrangian Monte Carlo particle pusher

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "particles/lagrangian_mc.hpp"
#include "particles/particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {
namespace {

constexpr std::uint64_t kMoveDraw = 0;
constexpr std::uint64_t kRefinementDraw = 1;

KOKKOS_INLINE_FUNCTION
std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

KOKKOS_INLINE_FUNCTION
Real LagrangianMCRandom(const std::uint64_t seed, const int tag, const int cycle,
                        const std::uint64_t draw_id) {
  // Key draws by durable state so particle reordering and MPI migration do not alter them.
  std::uint64_t key = seed;
  key ^= static_cast<std::uint64_t>(tag) * 0xd2b74407b1ce6e93ULL;
  key ^= static_cast<std::uint64_t>(cycle) * 0x9e3779b97f4a7c15ULL;
  key ^= draw_id * 0x94d049bb133111ebULL;
  const std::uint64_t bits = SplitMix64(key);
#if SINGLE_PRECISION_ENABLED
  return static_cast<Real>(bits >> 40) * (1.0f/16777216.0f);
#else
  return static_cast<Real>(bits >> 11) * (1.0/9007199254740992.0);
#endif
}

} // namespace
//----------------------------------------------------------------------------------------
//! \fn TaskStatus ParticlePopulation::PushLagrangianMC
//! \brief Move Lagrangian MC particles using the fluid's accumulated mass fluxes.

TaskStatus ParticlePopulation::PushLagrangianMC(Driver*, int) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  DvceArray5D<Real> start_u;
  DvceArray4D<Real> intflx1;
  DvceArray4D<Real> intflx2;
  DvceArray4D<Real> intflx3;
  if (pmy_pack->phydro != nullptr) {
    start_u = pmy_pack->phydro->u1;
    intflx1 = pmy_pack->phydro->density_flux_integral.x1f;
    intflx2 = pmy_pack->phydro->density_flux_integral.x2f;
    intflx3 = pmy_pack->phydro->density_flux_integral.x3f;
  } else {
    start_u = pmy_pack->pmhd->u1;
    intflx1 = pmy_pack->pmhd->density_flux_integral.x1f;
    intflx2 = pmy_pack->pmhd->density_flux_integral.x2f;
    intflx3 = pmy_pack->pmhd->density_flux_integral.x3f;
  }
  if (lmc_check_flux_probabilities) {
    const int nkji = nx3*nx2*nx1;
    const int nji = nx2*nx1;
    const int nmkji = pmy_pack->nmb_thispack*nkji;
    const Real invalid_probability = std::numeric_limits<Real>::max();
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
          const int i = idx - m*nkji - k*nji - j*nx1 + is;
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
      const int i = max_probability.loc - m*nkji - k*nji - j*nx1 + is;
      j += js;
      k += ks;

      DvceArray1D<Real> bad_values("particle_lmc_invalid_flux_values", 7);
      par_for("particle_lmc_load_invalid_flux_values", DevExeSpace(), 0, 0,
      KOKKOS_LAMBDA(const int) {
        bad_values(0) = start_u(m,IDN,k,j,i);
        bad_values(1) = -intflx1(m,k,j,i);
        bad_values(2) =  intflx1(m,k,j,i+1);
        bad_values(3) = multi_d ? -intflx2(m,k,j,i) : 0.0;
        bad_values(4) = multi_d ?  intflx2(m,k,j+1,i) : 0.0;
        bad_values(5) = three_d ? -intflx3(m,k,j,i) : 0.0;
        bad_values(6) = three_d ?  intflx3(m,k+1,j,i) : 0.0;
      });
      auto bad_values_h =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), bad_values);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Invalid Lagrangian MC outgoing probability "
                << std::setprecision(17) << max_probability.val << " on rank="
                << global_variable::my_rank << " gid=" << pmy_pack->gids + m
                << " cell (k,j,i)=(" << k << "," << j << "," << i << "). "
                << "density=" << bad_values_h(0) << ", outward fluxes=("
                << bad_values_h(1) << "," << bad_values_h(2) << ","
                << bad_values_h(3) << "," << bad_values_h(4) << ","
                << bad_values_h(5) << "," << bad_values_h(6) << "). Probability "
                << "must be finite and must not exceed one; the beginning-of-step "
                << "density must be non-negative." << std::endl;
#if MPI_PARALLEL_ENABLED
      MPI_Abort(MPI_COMM_WORLD, 1);
#endif
      std::exit(EXIT_FAILURE);
    }
  }
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const int gids = pmy_pack->gids;
  const int npart = nprtcl_thispack;
  const int cycle = pmy_pack->pmesh->ncycle;
  const bool multilevel = pmy_pack->pmesh->multilevel;
  const std::uint64_t seed = lmc_random_seed;

  par_for("particle_lmc_move", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    if (pi(lagrangian_mc::PSTATUS,p) != PACTIVE) return;

    const int m = pi(lagrangian_mc::PGID,p) - gids;
    auto size = mbsize.d_view(m);
    const int i =
        static_cast<int>((pr(lagrangian_mc::IPX,p) - size.x1min)/size.dx1) + is;
    const int j = multi_d ?
        static_cast<int>((pr(lagrangian_mc::IPY,p) - size.x2min)/size.dx2) + js : js;
    const int k = three_d ?
        static_cast<int>((pr(lagrangian_mc::IPZ,p) - size.x3min)/size.dx3) + ks : ks;
    const Real start_density = start_u(m,IDN,k,j,i);
    int move = lagrangian_mc::PMOVE_NONE;
    int source_cell = lagrangian_mc::EncodeSourceCell(
        mblev.d_view(m), i-is, j-js, k-ks);

    if (start_density != 0.0) {
      const Real x1l = fmax(-intflx1(m,k,j,i), 0.0)/start_density;
      const Real x1r = fmax( intflx1(m,k,j,i+1), 0.0)/start_density;
      const Real x2l = multi_d ? fmax(-intflx2(m,k,j,i), 0.0)/start_density : 0.0;
      const Real x2r = multi_d ? fmax( intflx2(m,k,j+1,i), 0.0)/start_density : 0.0;
      const Real x3l = three_d ? fmax(-intflx3(m,k,j,i), 0.0)/start_density : 0.0;
      const Real x3r = three_d ? fmax( intflx3(m,k+1,j,i), 0.0)/start_density : 0.0;
      const Real draw = LagrangianMCRandom(
          seed, pi(lagrangian_mc::PTAG,p), cycle, kMoveDraw);

      if (draw < x1l) {
        pr(lagrangian_mc::IPX,p) -= size.dx1;
        move = lagrangian_mc::PMOVE_X1_LEFT;
      } else if (draw < x1l + x1r) {
        pr(lagrangian_mc::IPX,p) += size.dx1;
        move = lagrangian_mc::PMOVE_X1_RIGHT;
      } else if (draw < x1l + x1r + x2l) {
        pr(lagrangian_mc::IPY,p) -= size.dx2;
        move = lagrangian_mc::PMOVE_X2_LEFT;
      } else if (draw < x1l + x1r + x2l + x2r) {
        pr(lagrangian_mc::IPY,p) += size.dx2;
        move = lagrangian_mc::PMOVE_X2_RIGHT;
      } else if (draw < x1l + x1r + x2l + x2r + x3l) {
        pr(lagrangian_mc::IPZ,p) -= size.dx3;
        move = lagrangian_mc::PMOVE_X3_LEFT;
      } else if (draw < x1l + x1r + x2l + x2r + x3l + x3r) {
        pr(lagrangian_mc::IPZ,p) += size.dx3;
        move = lagrangian_mc::PMOVE_X3_RIGHT;
      }
    }
    if (multilevel && move != lagrangian_mc::PMOVE_NONE) {
      source_cell = lagrangian_mc::MarkSourceCorrectionPending(source_cell);
    }
    pi(lagrangian_mc::PLASTMOVE,p) = move;
    pi(lagrangian_mc::PSOURCECELL,p) = source_cell;
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ParticlePopulation::FinalizeLagrangianMCMove
//! \brief Correct SMR crossings and record the final minimum-radius position.

TaskStatus ParticlePopulation::FinalizeLagrangianMCMove(Driver*, int) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mblev = pmy_pack->pmb->mb_lev;
  DvceArray4D<Real> intflx1;
  DvceArray4D<Real> intflx2;
  DvceArray4D<Real> intflx3;
  if (pmy_pack->phydro != nullptr) {
    intflx1 = pmy_pack->phydro->density_flux_integral.x1f;
    intflx2 = pmy_pack->phydro->density_flux_integral.x2f;
    intflx3 = pmy_pack->phydro->density_flux_integral.x3f;
  } else {
    intflx1 = pmy_pack->pmhd->density_flux_integral.x1f;
    intflx2 = pmy_pack->pmhd->density_flux_integral.x2f;
    intflx3 = pmy_pack->pmhd->density_flux_integral.x3f;
  }
  const int gids = pmy_pack->gids;
  const int npart = nprtcl_thispack;
  const int cycle = pmy_pack->pmesh->ncycle;
  const Real event_time = pmy_pack->pmesh->time + pmy_pack->pmesh->dt;
  const std::uint64_t seed = lmc_random_seed;

  par_for("particle_lmc_finalize_move", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    int source_cell = pi(lagrangian_mc::PSOURCECELL,p);
    const int move = pi(lagrangian_mc::PLASTMOVE,p);
    if (lagrangian_mc::SourceCorrectionPending(source_cell)) {
      const int m = pi(lagrangian_mc::PGID,p) - gids;
      auto size = mbsize.d_view(m);
      const int source_level = lagrangian_mc::SourceLevel(source_cell);
      const int destination_level = mblev.d_view(m);
      const Real dx1 = size.dx1;
      const Real dx2 = multi_d ? size.dx2 : 0.0;
      const Real dx3 = three_d ? size.dx3 : 0.0;

      if (destination_level > source_level) {
        if (move == lagrangian_mc::PMOVE_X1_LEFT) {
          pr(lagrangian_mc::IPX,p) += 0.5*dx1;
          pr(lagrangian_mc::IPY,p) -= 0.5*dx2;
          pr(lagrangian_mc::IPZ,p) -= 0.5*dx3;
        } else if (move == lagrangian_mc::PMOVE_X1_RIGHT) {
          pr(lagrangian_mc::IPX,p) -= 0.5*dx1;
          pr(lagrangian_mc::IPY,p) -= 0.5*dx2;
          pr(lagrangian_mc::IPZ,p) -= 0.5*dx3;
        } else if (move == lagrangian_mc::PMOVE_X2_LEFT) {
          pr(lagrangian_mc::IPY,p) += 0.5*dx2;
          pr(lagrangian_mc::IPX,p) -= 0.5*dx1;
          pr(lagrangian_mc::IPZ,p) -= 0.5*dx3;
        } else if (move == lagrangian_mc::PMOVE_X2_RIGHT) {
          pr(lagrangian_mc::IPY,p) -= 0.5*dx2;
          pr(lagrangian_mc::IPX,p) -= 0.5*dx1;
          pr(lagrangian_mc::IPZ,p) -= 0.5*dx3;
        } else if (move == lagrangian_mc::PMOVE_X3_LEFT) {
          pr(lagrangian_mc::IPZ,p) += 0.5*dx3;
          pr(lagrangian_mc::IPX,p) -= 0.5*dx1;
          pr(lagrangian_mc::IPY,p) -= 0.5*dx2;
        } else if (move == lagrangian_mc::PMOVE_X3_RIGHT) {
          pr(lagrangian_mc::IPZ,p) -= 0.5*dx3;
          pr(lagrangian_mc::IPX,p) -= 0.5*dx1;
          pr(lagrangian_mc::IPY,p) -= 0.5*dx2;
        }

        const int i = static_cast<int>(
            (pr(lagrangian_mc::IPX,p) - size.x1min)/dx1) + is;
        const int j = multi_d ? static_cast<int>(
            (pr(lagrangian_mc::IPY,p) - size.x2min)/dx2) + js : js;
        const int k = three_d ? static_cast<int>(
            (pr(lagrangian_mc::IPZ,p) - size.x3min)/dx3) + ks : ks;
        Real weight[4] = {0.0, 0.0, 0.0, 0.0};
        if (move == lagrangian_mc::PMOVE_X1_LEFT) {
          weight[0] = -intflx1(m,k,j,i+1);
          weight[1] = multi_d ? -intflx1(m,k,j+1,i+1) : 0.0;
          weight[2] = three_d ? -intflx1(m,k+1,j,i+1) : 0.0;
          weight[3] = (multi_d && three_d) ? -intflx1(m,k+1,j+1,i+1) : 0.0;
        } else if (move == lagrangian_mc::PMOVE_X1_RIGHT) {
          weight[0] = intflx1(m,k,j,i);
          weight[1] = multi_d ? intflx1(m,k,j+1,i) : 0.0;
          weight[2] = three_d ? intflx1(m,k+1,j,i) : 0.0;
          weight[3] = (multi_d && three_d) ? intflx1(m,k+1,j+1,i) : 0.0;
        } else if (move == lagrangian_mc::PMOVE_X2_LEFT) {
          weight[0] = -intflx2(m,k,j+1,i);
          weight[1] = -intflx2(m,k,j+1,i+1);
          weight[2] = three_d ? -intflx2(m,k+1,j+1,i) : 0.0;
          weight[3] = three_d ? -intflx2(m,k+1,j+1,i+1) : 0.0;
        } else if (move == lagrangian_mc::PMOVE_X2_RIGHT) {
          weight[0] = intflx2(m,k,j,i);
          weight[1] = intflx2(m,k,j,i+1);
          weight[2] = three_d ? intflx2(m,k+1,j,i) : 0.0;
          weight[3] = three_d ? intflx2(m,k+1,j,i+1) : 0.0;
        } else if (move == lagrangian_mc::PMOVE_X3_LEFT) {
          weight[0] = -intflx3(m,k+1,j,i);
          weight[1] = -intflx3(m,k+1,j,i+1);
          weight[2] = -intflx3(m,k+1,j+1,i);
          weight[3] = -intflx3(m,k+1,j+1,i+1);
        } else if (move == lagrangian_mc::PMOVE_X3_RIGHT) {
          weight[0] = intflx3(m,k,j,i);
          weight[1] = intflx3(m,k,j,i+1);
          weight[2] = intflx3(m,k,j+1,i);
          weight[3] = intflx3(m,k,j+1,i+1);
        }

        Real total = 0.0;
        for (int n=0; n<4; ++n) {
          weight[n] = fmax(weight[n], 0.0);
          total += weight[n];
        }
        if (total <= 0.0) total = 1.0;
        for (int n=0; n<4; ++n) weight[n] /= total;

        const Real draw = LagrangianMCRandom(
            seed, pi(lagrangian_mc::PTAG,p), cycle, kRefinementDraw);
        int target = 3;
        if (draw < weight[0]) {
          target = 0;
        } else if (draw < weight[0] + weight[1]) {
          target = 1;
        } else if (draw < weight[0] + weight[1] + weight[2]) {
          target = 2;
        }

        if (move == lagrangian_mc::PMOVE_X1_LEFT ||
            move == lagrangian_mc::PMOVE_X1_RIGHT) {
          if (target == 1 || target == 3) pr(lagrangian_mc::IPY,p) += dx2;
          if (target == 2 || target == 3) pr(lagrangian_mc::IPZ,p) += dx3;
        } else if (move == lagrangian_mc::PMOVE_X2_LEFT ||
                   move == lagrangian_mc::PMOVE_X2_RIGHT) {
          if (target == 1 || target == 3) pr(lagrangian_mc::IPX,p) += dx1;
          if (target == 2 || target == 3) pr(lagrangian_mc::IPZ,p) += dx3;
        } else if (move == lagrangian_mc::PMOVE_X3_LEFT ||
                   move == lagrangian_mc::PMOVE_X3_RIGHT) {
          if (target == 1 || target == 3) pr(lagrangian_mc::IPX,p) += dx1;
          if (target == 2 || target == 3) pr(lagrangian_mc::IPY,p) += dx2;
        }
      } else if (destination_level < source_level) {
        const int parity = lagrangian_mc::SourceParity(source_cell);
        pr(lagrangian_mc::IPX,p) +=
            ((parity & lagrangian_mc::PSOURCE_X1_ODD) ? -0.25 : 0.25)*dx1;
        if (multi_d) {
          pr(lagrangian_mc::IPY,p) +=
              ((parity & lagrangian_mc::PSOURCE_X2_ODD) ? -0.25 : 0.25)*dx2;
        }
        if (three_d) {
          pr(lagrangian_mc::IPZ,p) +=
              ((parity & lagrangian_mc::PSOURCE_X3_ODD) ? -0.25 : 0.25)*dx3;
        }
        if (move == lagrangian_mc::PMOVE_X1_LEFT) {
          pr(lagrangian_mc::IPX,p) -= 0.5*dx1;
        } else if (move == lagrangian_mc::PMOVE_X1_RIGHT) {
          pr(lagrangian_mc::IPX,p) += 0.5*dx1;
        } else if (move == lagrangian_mc::PMOVE_X2_LEFT) {
          pr(lagrangian_mc::IPY,p) -= 0.5*dx2;
        } else if (move == lagrangian_mc::PMOVE_X2_RIGHT) {
          pr(lagrangian_mc::IPY,p) += 0.5*dx2;
        } else if (move == lagrangian_mc::PMOVE_X3_LEFT) {
          pr(lagrangian_mc::IPZ,p) -= 0.5*dx3;
        } else if (move == lagrangian_mc::PMOVE_X3_RIGHT) {
          pr(lagrangian_mc::IPZ,p) += 0.5*dx3;
        }
      }
      pi(lagrangian_mc::PSOURCECELL,p) =
          lagrangian_mc::MarkSourceCorrectionComplete(source_cell);
    }

    if (move == lagrangian_mc::PMOVE_NONE) return;
    const Real x = pr(lagrangian_mc::IPX,p);
    const Real y = pr(lagrangian_mc::IPY,p);
    const Real z = pr(lagrangian_mc::IPZ,p);
    const Real radius_sq = x*x + y*y + z*z;
    const Real minimum_radius_sq =
        pr(lagrangian_mc::IPXMIN,p)*pr(lagrangian_mc::IPXMIN,p) +
        pr(lagrangian_mc::IPYMIN,p)*pr(lagrangian_mc::IPYMIN,p) +
        pr(lagrangian_mc::IPZMIN,p)*pr(lagrangian_mc::IPZMIN,p);
    if (radius_sq < minimum_radius_sq) {
      pr(lagrangian_mc::IPXMIN,p) = x;
      pr(lagrangian_mc::IPYMIN,p) = y;
      pr(lagrangian_mc::IPZMIN,p) = z;
      pr(lagrangian_mc::IPTMIN,p) = event_time;
    }
  });

  return TaskStatus::complete;
}
} // namespace particles
