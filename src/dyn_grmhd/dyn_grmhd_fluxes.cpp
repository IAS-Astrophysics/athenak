//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_fluxes.cpp
//  \brief Calculate 3D fluxes for hydro

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
//#include <stdio.h>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "dyn_grmhd.hpp"
#include "dyn_grmhd_util.hpp"
#include "coordinates/adm.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "mhd/mhd.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "dyn_grmhd/rsolvers/llf_dyn_grmhd.hpp"
#include "dyn_grmhd/rsolvers/hlle_dyn_grmhd.hpp"
// include PrimitiveSolver stuff
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {
namespace {

constexpr int kX3DebugSides = 2;
constexpr int kX3DebugIndexCount = 4;
constexpr int kX3DebugCoordCount = 3;
constexpr int kX3DebugValueCount = 43;

enum X3DebugIndex {
  X3DBG_M = 0,
  X3DBG_I = 1,
  X3DBG_J = 2,
  X3DBG_K = 3,
};

enum X3DebugValue {
  X3DBG_FOUND = 0,
  X3DBG_CC_KM1_RHO,
  X3DBG_CC_KM1_PRS,
  X3DBG_CC_KM1_VX,
  X3DBG_CC_KM1_VY,
  X3DBG_CC_KM1_VZ,
  X3DBG_CC_K_RHO,
  X3DBG_CC_K_PRS,
  X3DBG_CC_K_VX,
  X3DBG_CC_K_VY,
  X3DBG_CC_K_VZ,
  X3DBG_WL_RHO,
  X3DBG_WL_PRS,
  X3DBG_WL_VX,
  X3DBG_WL_VY,
  X3DBG_WL_VZ,
  X3DBG_WR_RHO,
  X3DBG_WR_PRS,
  X3DBG_WR_VX,
  X3DBG_WR_VY,
  X3DBG_WR_VZ,
  X3DBG_BL_X,
  X3DBG_BL_Y,
  X3DBG_BL_Z,
  X3DBG_BR_X,
  X3DBG_BR_Y,
  X3DBG_BR_Z,
  X3DBG_BZ_FACE,
  X3DBG_ALPHA,
  X3DBG_BETA_X,
  X3DBG_BETA_Y,
  X3DBG_BETA_Z,
  X3DBG_GXX,
  X3DBG_GXY,
  X3DBG_GXZ,
  X3DBG_GYY,
  X3DBG_GYZ,
  X3DBG_GZZ,
  X3DBG_DETG,
  X3DBG_FLUX_RHO,
  X3DBG_E23,
  X3DBG_E13,
  X3DBG_LEVEL,
};

const char *X3DebugSideName(int side) {
  return (side == 0) ? "minus_y" : "plus_y";
}

const char *X3DebugValueName(int value) {
  switch (value) {
    case X3DBG_CC_KM1_RHO: return "cc_km1_rho";
    case X3DBG_CC_KM1_PRS: return "cc_km1_prs";
    case X3DBG_CC_KM1_VX: return "cc_km1_vx";
    case X3DBG_CC_KM1_VY: return "cc_km1_vy";
    case X3DBG_CC_KM1_VZ: return "cc_km1_vz";
    case X3DBG_CC_K_RHO: return "cc_k_rho";
    case X3DBG_CC_K_PRS: return "cc_k_prs";
    case X3DBG_CC_K_VX: return "cc_k_vx";
    case X3DBG_CC_K_VY: return "cc_k_vy";
    case X3DBG_CC_K_VZ: return "cc_k_vz";
    case X3DBG_WL_RHO: return "wl_rho";
    case X3DBG_WL_PRS: return "wl_prs";
    case X3DBG_WL_VX: return "wl_vx";
    case X3DBG_WL_VY: return "wl_vy";
    case X3DBG_WL_VZ: return "wl_vz";
    case X3DBG_WR_RHO: return "wr_rho";
    case X3DBG_WR_PRS: return "wr_prs";
    case X3DBG_WR_VX: return "wr_vx";
    case X3DBG_WR_VY: return "wr_vy";
    case X3DBG_WR_VZ: return "wr_vz";
    case X3DBG_BL_X: return "bl_x";
    case X3DBG_BL_Y: return "bl_y";
    case X3DBG_BL_Z: return "bl_z";
    case X3DBG_BR_X: return "br_x";
    case X3DBG_BR_Y: return "br_y";
    case X3DBG_BR_Z: return "br_z";
    case X3DBG_BZ_FACE: return "bz_face";
    case X3DBG_ALPHA: return "alpha";
    case X3DBG_BETA_X: return "beta_x";
    case X3DBG_BETA_Y: return "beta_y";
    case X3DBG_BETA_Z: return "beta_z";
    case X3DBG_GXX: return "gxx";
    case X3DBG_GXY: return "gxy";
    case X3DBG_GXZ: return "gxz";
    case X3DBG_GYY: return "gyy";
    case X3DBG_GYZ: return "gyz";
    case X3DBG_GZZ: return "gzz";
    case X3DBG_DETG: return "detg";
    case X3DBG_FLUX_RHO: return "flux_rho";
    case X3DBG_E23: return "e23";
    case X3DBG_E13: return "e13";
    default: return "";
  }
}

int X3DebugYParity(int value) {
  switch (value) {
    case X3DBG_CC_KM1_VY:
    case X3DBG_CC_K_VY:
    case X3DBG_WL_VY:
    case X3DBG_WR_VY:
    case X3DBG_BL_Y:
    case X3DBG_BR_Y:
    case X3DBG_BETA_Y:
    case X3DBG_GXY:
    case X3DBG_GYZ:
      return -1;
    default:
      return 1;
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalcFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes
//! Note this function is templated over RS for better performance on GPUs.

template<class EOSPolicy, class ErrorPolicy> template <DynGRMHD_RSolver rsolver_method_>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes(Driver *pdriver, int stage) {
  RegionIndcs indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;
  int ncells1 = indcs_.nx1 + 2*(indcs_.ng);

  int nhyd = pmy_pack->pmhd->nmhd;
  int nvars = pmy_pack->pmhd->nmhd + pmy_pack->pmhd->nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = pmy_pack->pmhd->recon_method;
  auto size_ = pmy_pack->pmb->mb_size;
  auto coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = pmy_pack->pmhd->w0;
  auto &b0_ = pmy_pack->pmhd->bcc0;
  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = pmy_pack->pmhd->peos->eos_data;
  auto &dyn_eos_ = eos;
  auto &use_fofc = pmy_pack->pmhd->use_fofc;
  bool extrema = false;
  if (recon_method_ == ReconstructionMethod::ppmx) {
    extrema = true;
  }
  // Short-circuit the flux calculation if everything is to be fixed.
  if (fixed_evolution) {
    return TaskStatus::complete;
  }

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 2 +
                    ScrArray2D<Real>::shmem_size(3, ncells1) * 2;
  int scr_level = scratch_level;
  auto flx1_ = pmy_pack->pmhd->uflx.x1f;
  auto &e31_ = pmy_pack->pmhd->e3x1;
  auto &e21_ = pmy_pack->pmhd->e2x1;
  auto &bx_  = pmy_pack->pmhd->b0.x1f;

  // set the loop limits for 1D/2D/3D problems
  int jl, ju, kl, ku;
  if (pmy_pack->pmesh->one_d) {
    jl = js, ju = je, kl = ks, ku = ke;
  } else if (pmy_pack->pmesh->two_d) {
    jl = js-1, ju = je+1, kl = ks, ku = ke;
  } else {
    jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
  }
  int il = is, iu = ie+1;
  if (use_fofc) { il = is-1, iu = ie+2; }

  par_for_outer("dyngrflux_x1",DevExeSpace(), scr_size, scr_level,
      0, nmb1, kl, ku, jl, ju,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> bl(member.team_scratch(scr_level), 3, ncells1);
    ScrArray2D<Real> br(member.team_scratch(scr_level), 3, ncells1);

    // Reconstruct qR[i] and qL[i+1]
    switch (recon_method_) {
      case ReconstructionMethod::dc:
        DonorCellX1(member, m, k, j, il-1, iu, w0_, wl, wr);
        DonorCellX1(member, m, k, j, il-1, iu, b0_, bl, br);
        break;
      case ReconstructionMethod::plm:
        PiecewiseLinearX1(member, m, k, j, il-1, iu, w0_, wl, wr);
        PiecewiseLinearX1(member, m, k, j, il-1, iu, b0_, bl, br);
        break;
      // JF: These higher-order reconstruction methods all need EOS_Data to calculate a
      // floor. However, it isn't used by DynGRMHD at all.
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
        PiecewiseParabolicX1(member,eos_,extrema,false, m, k, j, il-1, iu, w0_, wl, wr);
        PiecewiseParabolicX1(member,eos_,extrema,false, m, k, j, il-1, iu, b0_, bl, br);
        break;
      case ReconstructionMethod::wenoz:
        WENOZX1(member, eos_, false, m, k, j, il-1, iu, w0_, wl, wr);
        WENOZX1(member, eos_, false, m, k, j, il-1, iu, b0_, bl, br);
        break;
      default:
        break;
    }
    // Sync all threads in the team so that scratch memory is consistent
    member.team_barrier();

    // compute fluxes over [is,ie+1]
    auto &dyn_eos = dyn_eos_;
    auto &indcs = indcs_;
    auto &size = size_;
    auto &coord = coord_;
    auto &flx1 = flx1_;
    auto &bx = bx_;
    auto &e31 = e31_;
    auto &e21 = e21_;
    auto &nhyd_ = nhyd;
    auto nscal_ = nvars - nhyd;
    auto &adm_ = adm;
    //int il = is; int iu = ie+1;
    if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
      LLF_DYNGR<IVX>(member, dyn_eos, indcs, size, coord, m, k, j, il, iu,
                wl, wr, bl, br, bx, nhyd_, nscal_, adm_,
                flx1, e31, e21);
    } else if constexpr (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
      HLLE_DYNGR<IVX>(member, dyn_eos, indcs, size, coord, m, k, j, il, iu,
                wl, wr, bl, br, bx, nhyd_, nscal_, adm_,
                flx1, e31, e21);
    }
    member.team_barrier();

    // Calculate fluxes of scalars (if any)
    if (nvars > nhyd) {
      for (int n=nhyd; n<nvars; ++n) {
        par_for_inner(member, il, iu, [&](const int i) {
          if (flx1(m,IDN,k,j,i) >= 0.0) {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wl(n,i);
          } else {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wr(n,i);
          }
        });
      }
    }
    member.team_barrier();
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3
             + ScrArray2D<Real>::shmem_size(3, ncells1) * 3;
    auto flx2_ = pmy_pack->pmhd->uflx.x2f;
    auto &by_ = pmy_pack->pmhd->b0.x2f;
    auto &e12_ = pmy_pack->pmhd->e1x2;
    auto &e32_ = pmy_pack->pmhd->e3x2;

    // set the loop limits for 2D/3D problems
    if (pmy_pack->pmesh->two_d) {
      kl = ks, ku = ke;
    } else { // 3D
      kl = ks-1, ku = ke+1;
    }
    jl = js-1, ju = je+1;
    if (use_fofc) { jl = js-2, ju = je+2; }

    par_for_outer("dyngrflux_x2",DevExeSpace(), scr_size, scr_level, 0, nmb1, kl, ku,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k) {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr4(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> scr5(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> scr6(member.team_scratch(scr_level), 3, ncells1);

      for (int j=jl; j<=ju; ++j) {
        // Permute scratch arrays.
        auto wl     = scr1;
        auto wl_jp1 = scr2;
        auto wr     = scr3;
        auto bl     = scr4;
        auto bl_jp1 = scr5;
        auto br     = scr6;
        if ((j%2) == 0) {
          wl     = scr2;
          wl_jp1 = scr1;
          bl     = scr5;
          bl_jp1 = scr4;
        }

        // Reconstruct qR[j] and qL[j+1]
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            DonorCellX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            DonorCellX2(member, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            PiecewiseLinearX2(member, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          // JF: These higher-order reconstruction methods all need EOS_Data to calculate
          // a floor. However, it isn't used by DynGRMHD.
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX2(member,eos_,extrema,false, m, k, j, is-1, ie+1,
                                 w0_, wl_jp1, wr);
            PiecewiseParabolicX2(member,eos_,extrema,false, m, k, j, is-1, ie+1,
                                 b0_, bl_jp1, br);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX2(member, eos_, false, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            WENOZX2(member, eos_, false, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          default:
            break;
        }
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        // compute fluxes over [js,je+1]
        auto &dyn_eos = dyn_eos_;
        auto &indcs = indcs_;
        auto &size = size_;
        auto &coord = coord_;
        auto &flx2 = flx2_;
        auto &by   = by_;
        auto &e12  = e12_;
        auto &e32  = e32_;
        auto &nhyd_ = nhyd;
        auto nscal_ = nvars - nhyd;
        auto &adm_ = adm;
        //int il = is; int iu = ie;
        if (j>(jl)) {
          if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
            LLF_DYNGR<IVY>(member, dyn_eos, indcs, size, coord, m, k, j, is-1, ie+1,
                      wl, wr, bl, br, by, nhyd_, nscal_, adm_, flx2, e12, e32);
          } else if constexpr (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
            HLLE_DYNGR<IVY>(member, dyn_eos, indcs, size, coord, m, k, j, is-1, ie+1,
                      wl, wr, bl, br, by, nhyd_, nscal_, adm_, flx2, e12, e32);
          }
        }
        member.team_barrier();

        // Calculate fluxes of scalars (if any)
        if (nvars > nhyd) {
          for (int n=nhyd; n<nvars; ++n) {
            par_for_inner(member, is-1, ie+1, [&](const int i) {
              if (flx2(m,IDN,k,j,i) >= 0.0) {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end of loop over j
      member.team_barrier();
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3
             + ScrArray2D<Real>::shmem_size(3, ncells1) * 3;
    auto &flx3_ = pmy_pack->pmhd->uflx.x3f;
    auto &bz_   = pmy_pack->pmhd->b0.x3f;
    auto &e23_  = pmy_pack->pmhd->e2x3;
    auto &e13_  = pmy_pack->pmhd->e1x3;
    auto &lev_  = pmy_pack->pmb->mb_lev;

    kl = ks-1, ku = ke+1;
    if (use_fofc) { kl = ks-2, ku = ke+2; }

    Mesh *pm = pmy_pack->pmesh;
    const int debug_cycle = pm->ncycle;
    const bool x3_debug_active =
        dyngr_x3_debug &&
        (dyngr_x3_debug_cycle < 0 || dyngr_x3_debug_cycle == debug_cycle) &&
        (dyngr_x3_debug_stage < 0 || dyngr_x3_debug_stage == stage);
    DvceArray2D<int> x3_debug_idx;
    DvceArray2D<Real> x3_debug_vals;
    HostArray2D<int> x3_debug_idx_h("dyngr-x3-debug-idx-h",
                                    kX3DebugSides, kX3DebugIndexCount);
    HostArray2D<Real> x3_debug_coord_h("dyngr-x3-debug-coord-h",
                                       kX3DebugSides, kX3DebugCoordCount);
    if (x3_debug_active) {
      x3_debug_idx = DvceArray2D<int>("dyngr-x3-debug-idx",
                                      kX3DebugSides, kX3DebugIndexCount);
      x3_debug_vals = DvceArray2D<Real>("dyngr-x3-debug-vals",
                                        kX3DebugSides, kX3DebugValueCount);
      for (int side = 0; side < kX3DebugSides; ++side) {
        for (int n = 0; n < kX3DebugIndexCount; ++n) {
          x3_debug_idx_h(side, n) = -1;
        }
        for (int n = 0; n < kX3DebugCoordCount; ++n) {
          x3_debug_coord_h(side, n) = 0.0;
        }
      }

      auto &gid = pmy_pack->pmb->mb_gid;
      const Real x_target = dyngr_x3_debug_x;
      const Real y_abs = std::abs(dyngr_x3_debug_y_abs);
      const Real z_face_target = dyngr_x3_debug_z_face;
      Real best_score[kX3DebugSides];
      int best_level[kX3DebugSides];
      for (int side = 0; side < kX3DebugSides; ++side) {
        best_score[side] = std::numeric_limits<Real>::max();
        best_level[side] = std::numeric_limits<int>::min();
      }
      for (int m = 0; m < pmy_pack->nmb_thispack; ++m) {
        const auto &mb = size_.h_view(m);
        const int level = lev_.h_view(m);
        const bool x_inside = (x_target >= mb.x1min) && (x_target < mb.x1max);
        const bool z_inside = (z_face_target >= mb.x3min) && (z_face_target <= mb.x3max);
        if (!(x_inside && z_inside)) continue;

        int i = is + static_cast<int>(std::floor((x_target - mb.x1min)/mb.dx1));
        i = std::max(is, std::min(ie, i));
        int kface = ks + static_cast<int>(std::llround((z_face_target - mb.x3min)/mb.dx3));
        kface = std::max(ks, std::min(ke + 1, kface));
        if (!(kface > kl && kface <= ku)) continue;
        const Real x = mb.x1min + (static_cast<Real>(i - is) + 0.5)*mb.dx1;
        const Real zface = mb.x3min + static_cast<Real>(kface - ks)*mb.dx3;

        for (int side = 0; side < kX3DebugSides; ++side) {
          const Real y_target = (side == 0) ? -y_abs : y_abs;
          const bool y_inside = (y_target >= mb.x2min) && (y_target < mb.x2max);
          if (!y_inside) continue;
          int j = js + static_cast<int>(std::floor((y_target - mb.x2min)/mb.dx2));
          j = std::max(js, std::min(je, j));
          const Real y = mb.x2min + (static_cast<Real>(j - js) + 0.5)*mb.dx2;
          const Real score = std::abs(x - x_target) + std::abs(y - y_target) +
                             std::abs(zface - z_face_target);
          if (level > best_level[side] ||
              (level == best_level[side] && score < best_score[side])) {
            best_level[side] = level;
            best_score[side] = score;
            x3_debug_idx_h(side, X3DBG_M) = m;
            x3_debug_idx_h(side, X3DBG_I) = i;
            x3_debug_idx_h(side, X3DBG_J) = j;
            x3_debug_idx_h(side, X3DBG_K) = kface;
            x3_debug_coord_h(side, 0) = x;
            x3_debug_coord_h(side, 1) = y;
            x3_debug_coord_h(side, 2) = zface;
          }
        }
      }
      Kokkos::deep_copy(x3_debug_idx, x3_debug_idx_h);
      Kokkos::deep_copy(x3_debug_vals, std::numeric_limits<Real>::quiet_NaN());
      for (int side = 0; side < kX3DebugSides; ++side) {
        const int m = x3_debug_idx_h(side, X3DBG_M);
        if (m >= 0) {
          std::cout << std::setprecision(17)
                    << "DYNGRX3TARGET rank=" << global_variable::my_rank
                    << " gid=" << gid.h_view(m)
                    << " level=" << lev_.h_view(m)
                    << " side=" << X3DebugSideName(side)
                    << " cycle=" << debug_cycle
                    << " time=" << pm->time
                    << " stage=" << stage
                    << " m=" << m
                    << " i=" << x3_debug_idx_h(side, X3DBG_I)
                    << " j=" << x3_debug_idx_h(side, X3DBG_J)
                    << " kface=" << x3_debug_idx_h(side, X3DBG_K)
                    << " x=" << x3_debug_coord_h(side, 0)
                    << " y=" << x3_debug_coord_h(side, 1)
                    << " zface=" << x3_debug_coord_h(side, 2)
                    << std::endl;
        }
      }
    }

    par_for_outer("dyngrflux_x3",DevExeSpace(), scr_size, scr_level, 0, nmb1, js-1, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr4(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> scr5(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> scr6(member.team_scratch(scr_level), 3, ncells1);

      for (int k=kl; k<=ku; ++k) {
        // Permute scratch arrays.
        auto wl     = scr1;
        auto wl_kp1 = scr2;
        auto wr     = scr3;
        auto bl     = scr4;
        auto bl_kp1 = scr5;
        auto br     = scr6;
        if ((k%2) == 0) {
          wl     = scr2;
          wl_kp1 = scr1;
          bl     = scr5;
          bl_kp1 = scr4;
        }

        // Reconstruct qR[j] and qL[j+1]
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            DonorCellX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            DonorCellX3(member, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            PiecewiseLinearX3(member, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          // JF: These higher-order reconstruction methods all need EOS_Data to calculate
          // a floor. However, it isn't used by DynGRMHD.
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX3(member,eos_,extrema,false, m, k, j, is-1, ie+1,
                                 w0_, wl_kp1, wr);
            PiecewiseParabolicX3(member,eos_,extrema,false, m, k, j, is-1, ie+1,
                                 b0_, bl_kp1, br);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX3(member, eos_, false, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            WENOZX3(member, eos_, false, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          default:
            break;
        }
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        // compute fluxes over [ks,ke+1]
        auto &dyn_eos = dyn_eos_;
        auto &indcs = indcs_;
        auto &size = size_;
        auto &coord = coord_;
        auto &flx3 = flx3_;
        auto &bz   = bz_;
        auto &e23  = e23_;
        auto &e13  = e13_;
        auto &adm_ = adm;
        auto &nhyd_ = nhyd;
        auto nscal_ = nvars - nhyd;
        //int il = is; int iu = ie;
        if (k>(kl)) {
          if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
            LLF_DYNGR<IVZ>(member, dyn_eos, indcs, size, coord, m, k, j, is-1, ie+1,
                      wl, wr, bl, br, bz, nhyd_, nscal_, adm_, flx3, e23, e13);
          } else if constexpr (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
            HLLE_DYNGR<IVZ>(member, dyn_eos, indcs, size, coord, m, k, j, is-1, ie+1,
                      wl, wr, bl, br, bz, nhyd_, nscal_, adm_, flx3, e23, e13);
          }
        }
        member.team_barrier();

        if (x3_debug_active) {
          for (int side = 0; side < kX3DebugSides; ++side) {
            const int dbg_m = x3_debug_idx(side, X3DBG_M);
            const int dbg_i = x3_debug_idx(side, X3DBG_I);
            const int dbg_j = x3_debug_idx(side, X3DBG_J);
            const int dbg_k = x3_debug_idx(side, X3DBG_K);
            if (m == dbg_m && j == dbg_j && k == dbg_k) {
              par_for_inner(member, dbg_i, dbg_i, [&](const int i) {
                Real g3d[NSPMETRIC];
                Real beta_u[3];
                Real alpha;
                adm::Face3Metric(m, k, j, i, adm_.g_dd, adm_.beta_u, adm_.alpha,
                                 g3d, beta_u, alpha);
                const Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                                  g3d[S22], g3d[S23], g3d[S33]);
                x3_debug_vals(side, X3DBG_FOUND) = 1.0;
                x3_debug_vals(side, X3DBG_CC_KM1_RHO) = w0_(m, IDN, k-1, j, i);
                x3_debug_vals(side, X3DBG_CC_KM1_PRS) = w0_(m, IPR, k-1, j, i);
                x3_debug_vals(side, X3DBG_CC_KM1_VX) = w0_(m, IVX, k-1, j, i);
                x3_debug_vals(side, X3DBG_CC_KM1_VY) = w0_(m, IVY, k-1, j, i);
                x3_debug_vals(side, X3DBG_CC_KM1_VZ) = w0_(m, IVZ, k-1, j, i);
                x3_debug_vals(side, X3DBG_CC_K_RHO) = w0_(m, IDN, k, j, i);
                x3_debug_vals(side, X3DBG_CC_K_PRS) = w0_(m, IPR, k, j, i);
                x3_debug_vals(side, X3DBG_CC_K_VX) = w0_(m, IVX, k, j, i);
                x3_debug_vals(side, X3DBG_CC_K_VY) = w0_(m, IVY, k, j, i);
                x3_debug_vals(side, X3DBG_CC_K_VZ) = w0_(m, IVZ, k, j, i);
                x3_debug_vals(side, X3DBG_WL_RHO) = wl(IDN, i);
                x3_debug_vals(side, X3DBG_WL_PRS) = wl(IPR, i);
                x3_debug_vals(side, X3DBG_WL_VX) = wl(IVX, i);
                x3_debug_vals(side, X3DBG_WL_VY) = wl(IVY, i);
                x3_debug_vals(side, X3DBG_WL_VZ) = wl(IVZ, i);
                x3_debug_vals(side, X3DBG_WR_RHO) = wr(IDN, i);
                x3_debug_vals(side, X3DBG_WR_PRS) = wr(IPR, i);
                x3_debug_vals(side, X3DBG_WR_VX) = wr(IVX, i);
                x3_debug_vals(side, X3DBG_WR_VY) = wr(IVY, i);
                x3_debug_vals(side, X3DBG_WR_VZ) = wr(IVZ, i);
                x3_debug_vals(side, X3DBG_BL_X) = bl(IBX, i);
                x3_debug_vals(side, X3DBG_BL_Y) = bl(IBY, i);
                x3_debug_vals(side, X3DBG_BL_Z) = bl(IBZ, i);
                x3_debug_vals(side, X3DBG_BR_X) = br(IBX, i);
                x3_debug_vals(side, X3DBG_BR_Y) = br(IBY, i);
                x3_debug_vals(side, X3DBG_BR_Z) = br(IBZ, i);
                x3_debug_vals(side, X3DBG_BZ_FACE) = bz(m, k, j, i);
                x3_debug_vals(side, X3DBG_ALPHA) = alpha;
                x3_debug_vals(side, X3DBG_BETA_X) = beta_u[0];
                x3_debug_vals(side, X3DBG_BETA_Y) = beta_u[1];
                x3_debug_vals(side, X3DBG_BETA_Z) = beta_u[2];
                x3_debug_vals(side, X3DBG_GXX) = g3d[S11];
                x3_debug_vals(side, X3DBG_GXY) = g3d[S12];
                x3_debug_vals(side, X3DBG_GXZ) = g3d[S13];
                x3_debug_vals(side, X3DBG_GYY) = g3d[S22];
                x3_debug_vals(side, X3DBG_GYZ) = g3d[S23];
                x3_debug_vals(side, X3DBG_GZZ) = g3d[S33];
                x3_debug_vals(side, X3DBG_DETG) = detg;
                x3_debug_vals(side, X3DBG_FLUX_RHO) = flx3(m, IDN, k, j, i);
                x3_debug_vals(side, X3DBG_E23) = e23(m, k, j, i);
                x3_debug_vals(side, X3DBG_E13) = e13(m, k, j, i);
                x3_debug_vals(side, X3DBG_LEVEL) = static_cast<Real>(lev_.d_view(m));
              });
            }
          }
        }
        member.team_barrier();

        // Calculate fluxes of scalars (if any)
        if (nvars > nhyd) {
          for (int n=nhyd; n<nvars; ++n) {
            par_for_inner(member, is-1, ie+1, [&](const int i) {
              if (flx3(m,IDN,k,j,i) >= 0.0) {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end of loop over j
      member.team_barrier();
    });

    if (x3_debug_active) {
      Kokkos::fence();
      auto x3_debug_vals_h = Kokkos::create_mirror_view(x3_debug_vals);
      Kokkos::deep_copy(x3_debug_vals_h, x3_debug_vals);
      auto &gid = pmy_pack->pmb->mb_gid;
      for (int side = 0; side < kX3DebugSides; ++side) {
        const int m = x3_debug_idx_h(side, X3DBG_M);
        if (m < 0) continue;
        std::cout << std::setprecision(17)
                  << "DYNGRX3DBG rank=" << global_variable::my_rank
                  << " gid=" << gid.h_view(m)
                  << " level=" << x3_debug_vals_h(side, X3DBG_LEVEL)
                  << " side=" << X3DebugSideName(side)
                  << " cycle=" << debug_cycle
                  << " time=" << pm->time
                  << " stage=" << stage
                  << " m=" << m
                  << " i=" << x3_debug_idx_h(side, X3DBG_I)
                  << " j=" << x3_debug_idx_h(side, X3DBG_J)
                  << " kface=" << x3_debug_idx_h(side, X3DBG_K)
                  << " x=" << x3_debug_coord_h(side, 0)
                  << " y=" << x3_debug_coord_h(side, 1)
                  << " zface=" << x3_debug_coord_h(side, 2)
                  << " found=" << x3_debug_vals_h(side, X3DBG_FOUND)
                  << " cc_km1_rho=" << x3_debug_vals_h(side, X3DBG_CC_KM1_RHO)
                  << " cc_km1_prs=" << x3_debug_vals_h(side, X3DBG_CC_KM1_PRS)
                  << " cc_km1_vx=" << x3_debug_vals_h(side, X3DBG_CC_KM1_VX)
                  << " cc_km1_vy=" << x3_debug_vals_h(side, X3DBG_CC_KM1_VY)
                  << " cc_km1_vz=" << x3_debug_vals_h(side, X3DBG_CC_KM1_VZ)
                  << " cc_k_rho=" << x3_debug_vals_h(side, X3DBG_CC_K_RHO)
                  << " cc_k_prs=" << x3_debug_vals_h(side, X3DBG_CC_K_PRS)
                  << " cc_k_vx=" << x3_debug_vals_h(side, X3DBG_CC_K_VX)
                  << " cc_k_vy=" << x3_debug_vals_h(side, X3DBG_CC_K_VY)
                  << " cc_k_vz=" << x3_debug_vals_h(side, X3DBG_CC_K_VZ)
                  << " wl_rho=" << x3_debug_vals_h(side, X3DBG_WL_RHO)
                  << " wl_prs=" << x3_debug_vals_h(side, X3DBG_WL_PRS)
                  << " wl_vx=" << x3_debug_vals_h(side, X3DBG_WL_VX)
                  << " wl_vy=" << x3_debug_vals_h(side, X3DBG_WL_VY)
                  << " wl_vz=" << x3_debug_vals_h(side, X3DBG_WL_VZ)
                  << " wr_rho=" << x3_debug_vals_h(side, X3DBG_WR_RHO)
                  << " wr_prs=" << x3_debug_vals_h(side, X3DBG_WR_PRS)
                  << " wr_vx=" << x3_debug_vals_h(side, X3DBG_WR_VX)
                  << " wr_vy=" << x3_debug_vals_h(side, X3DBG_WR_VY)
                  << " wr_vz=" << x3_debug_vals_h(side, X3DBG_WR_VZ)
                  << " bl_x=" << x3_debug_vals_h(side, X3DBG_BL_X)
                  << " bl_y=" << x3_debug_vals_h(side, X3DBG_BL_Y)
                  << " bl_z=" << x3_debug_vals_h(side, X3DBG_BL_Z)
                  << " br_x=" << x3_debug_vals_h(side, X3DBG_BR_X)
                  << " br_y=" << x3_debug_vals_h(side, X3DBG_BR_Y)
                  << " br_z=" << x3_debug_vals_h(side, X3DBG_BR_Z)
                  << " bz_face=" << x3_debug_vals_h(side, X3DBG_BZ_FACE)
                  << " alpha=" << x3_debug_vals_h(side, X3DBG_ALPHA)
                  << " beta_x=" << x3_debug_vals_h(side, X3DBG_BETA_X)
                  << " beta_y=" << x3_debug_vals_h(side, X3DBG_BETA_Y)
                  << " beta_z=" << x3_debug_vals_h(side, X3DBG_BETA_Z)
                  << " gxx=" << x3_debug_vals_h(side, X3DBG_GXX)
                  << " gxy=" << x3_debug_vals_h(side, X3DBG_GXY)
                  << " gxz=" << x3_debug_vals_h(side, X3DBG_GXZ)
                  << " gyy=" << x3_debug_vals_h(side, X3DBG_GYY)
                  << " gyz=" << x3_debug_vals_h(side, X3DBG_GYZ)
                  << " gzz=" << x3_debug_vals_h(side, X3DBG_GZZ)
                  << " detg=" << x3_debug_vals_h(side, X3DBG_DETG)
                  << " flux_rho=" << x3_debug_vals_h(side, X3DBG_FLUX_RHO)
                  << " e23=" << x3_debug_vals_h(side, X3DBG_E23)
                  << " e13=" << x3_debug_vals_h(side, X3DBG_E13)
                  << std::endl;
      }
      const int minus_m = x3_debug_idx_h(0, X3DBG_M);
      const int plus_m = x3_debug_idx_h(1, X3DBG_M);
      const bool have_pair =
          minus_m >= 0 && plus_m >= 0 &&
          x3_debug_vals_h(0, X3DBG_FOUND) == 1.0 &&
          x3_debug_vals_h(1, X3DBG_FOUND) == 1.0;
      if (have_pair) {
        for (int value = X3DBG_CC_KM1_RHO; value <= X3DBG_E13; ++value) {
          const char *name = X3DebugValueName(value);
          if (name[0] == '\0') continue;
          const Real minus_value = x3_debug_vals_h(0, value);
          const Real plus_value = x3_debug_vals_h(1, value);
          const int parity = X3DebugYParity(value);
          const Real parity_diff = minus_value - static_cast<Real>(parity)*plus_value;
          const Real abs_diff = std::abs(parity_diff);
          const Real local_scale = std::max(std::abs(minus_value), std::abs(plus_value));
          const Real local_rel =
              (local_scale > 0.0) ? abs_diff/local_scale :
              (abs_diff == 0.0 ? 0.0 : std::numeric_limits<Real>::infinity());
          std::cout << std::setprecision(17)
                    << "DYNGRX3PAIR rank=" << global_variable::my_rank
                    << " cycle=" << debug_cycle
                    << " time=" << pm->time
                    << " stage=" << stage
                    << " field=" << name
                    << " parity_y=" << parity
                    << " minus=" << minus_value
                    << " plus=" << plus_value
                    << " parity_diff=" << parity_diff
                    << " abs_diff=" << abs_diff
                    << " local_rel=" << local_rel
                    << " x=" << x3_debug_coord_h(0, 0)
                    << " minus_y=" << x3_debug_coord_h(0, 1)
                    << " plus_y=" << x3_debug_coord_h(1, 1)
                    << " zface=" << x3_debug_coord_h(0, 2)
                    << std::endl;
        }
      }
    }
  }

  // Call FOFC if necessary
  if (pmy_pack->pmhd->use_fofc || pmy_pack->pcoord->coord_data.bh_excise) {
    FOFC<rsolver_method_>(pdriver, stage);
  }

  pmy_pack->pmhd->SymmetryFluxDebugProbe("DynGRMHD_CalcFluxes", pdriver, stage);

  return TaskStatus::complete;
}

// function definitions for each template parameter
// Macro for instantiating every flux function for each Riemann solver
#define INSTANTIATE_CALC_FLUXES(EOSPolicy, ErrorPolicy) \
template \
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::\
            CalcFluxes<DynGRMHD_RSolver::llf_dyngr>(Driver *pdriver, int stage); \
template \
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::\
            CalcFluxes<DynGRMHD_RSolver::hlle_dyngr>(Driver *pdriver, int stage);

INSTANTIATE_CALC_FLUXES(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::EOSCompOSE<Primitive::NormalLogs>,
                        Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::EOSCompOSE<Primitive::NQTLogs>,
                        Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::EOSHybrid<Primitive::NormalLogs>,
                        Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::EOSHybrid<Primitive::NQTLogs>,
                        Primitive::ResetFloor)

} // namespace dyngr
