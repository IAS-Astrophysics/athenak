//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_fluxes.cpp
//  \brief Calculate 3D fluxes for hydro

#include <iostream>
//#include <stdio.h>

#include "athena.hpp"
#include "athena_tensor.hpp"
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

    kl = ks-1, ku = ke+1;
    if (use_fofc) { kl = ks-2, ku = ke+2; }

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
  }

  // Call FOFC if necessary
  if (pmy_pack->pmhd->use_fofc || pmy_pack->pcoord->coord_data.bh_excise) {
    FOFC<rsolver_method_>(pdriver, stage);
  }

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
