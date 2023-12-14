//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes.cpp
//! \brief Calculate fluxes of the conserved variables, and area-averaged electric fields
//! E = - (v X B) on cell faces for mhd.  Fluxes are stored in face-centered vector
//! 'uflx', while electric fields are stored in individual arrays: e2x1,e3x1 on x1-faces;
//! e1x2,e3x2 on x2-faces; e1x3,e2x3 on x3-faces.

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "mhd/rsolvers/advect_mhd.hpp"
#include "mhd/rsolvers/llf_mhd.hpp"
#include "mhd/rsolvers/hlle_mhd.hpp"
#include "mhd/rsolvers/hlld_mhd.hpp"
#include "mhd/rsolvers/llf_srmhd.hpp"
#include "mhd/rsolvers/hlle_srmhd.hpp"
#include "mhd/rsolvers/llf_grmhd.hpp"
#include "mhd/rsolvers/hlle_grmhd.hpp"
// #include "mhd/rsolvers/roe_mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::CalculateFlux
//! \brief Calculate fluxes of conserved variables, and face-centered area-averaged EMFs
//! for evolution of magnetic field
//! Note this function is templated over RS for better performance on GPUs.

template <MHD_RSolver rsolver_method_>
void MHD::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;
  int ncells1 = indcs_.nx1 + 2*(indcs_.ng);

  int &nmhd_ = nmhd;
  int nvars = nmhd + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  bool extrema = false;
  if (recon_method == ReconstructionMethod::ppmx) {
    extrema = true;
  }

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  auto &b0_ = bcc0;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
                     ScrArray2D<Real>::shmem_size(3, ncells1)) * 2;
  int scr_level = 0;
  auto &flx1_ = uflx.x1f;
  auto &e31_ = e3x1;
  auto &e21_ = e2x1;
  auto &bx_ = b0.x1f;

  // set the loop limits for 1D/2D/3D problems
  int jl,ju,kl,ku;
  if (pmy_pack->pmesh->one_d) {
    jl = js, ju = je, kl = ks, ku = ke;
  } else if (pmy_pack->pmesh->two_d) {
    jl = js-1, ju = je+1, kl = ks, ku = ke;
  } else {
    jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
  }
  int il = is, iu = ie+1;
  if (use_fofc) { il = is-1, iu = ie+2; }

  par_for_outer("mhd_flux1",DevExeSpace(), scr_size, scr_level, 0, nmb1, kl, ku, jl, ju,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> bl(member.team_scratch(scr_level), 3, ncells1);
    ScrArray2D<Real> br(member.team_scratch(scr_level), 3, ncells1);

    // Reconstruct qR[i] and qL[i+1], for both W and Bcc
    switch (recon_method_) {
      case ReconstructionMethod::dc:
        DonorCellX1(member, m, k, j, il-1, iu, w0_, wl, wr);
        DonorCellX1(member, m, k, j, il-1, iu, b0_, bl, br);
        break;
      case ReconstructionMethod::plm:
        PiecewiseLinearX1(member, m, k, j, il-1, iu, w0_, wl, wr);
        PiecewiseLinearX1(member, m, k, j, il-1, iu, b0_, bl, br);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
        PiecewiseParabolicX1(member,eos_,extrema,true,  m, k, j, il-1, iu, w0_, wl, wr);
        PiecewiseParabolicX1(member,eos_,extrema,false, m, k, j, il-1, iu, b0_, bl, br);
        break;
      case ReconstructionMethod::wenoz:
        WENOZX1(member, eos_, true,  m, k, j, il-1, iu, w0_, wl, wr);
        WENOZX1(member, eos_, false, m, k, j, il-1, iu, b0_, bl, br);
        break;
      default:
        break;
    }
    // Sync all threads in the team so that scratch memory is consistent
    member.team_barrier();

    // compute fluxes over [is,ie+1].  MHD RS also computes electric fields, where
    // (IBY) component of flx = E_{z} = -(v x B)_{z} = -(v1*b2 - v2*b1)
    // (IBZ) component of flx = E_{y} = -(v x B)_{y} =  (v1*b3 - v3*b1)
    // NOTE(@pdmullen): Capture variables prior to if constexpr.  Required for cuda 11.6+.
    auto eos = eos_;
    auto indcs = indcs_;
    auto size = size_;
    auto coord = coord_;
    auto bx = bx_;
    auto flx1 = flx1_;
    auto e31 = e31_;
    auto e21 = e21_;
    if constexpr (rsolver_method_ == MHD_RSolver::advect) {
      Advect(member,eos,indcs,size,coord,m,k,j,il,iu,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
      LLF(member,eos,indcs,size,coord,m,k,j,il,iu,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
      HLLE(member,eos,indcs,size,coord,m,k,j,il,iu,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
      HLLD(member,eos,indcs,size,coord,m,k,j,il,iu,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
      LLF_SR(member,eos,indcs,size,coord,m,k,j,il,iu,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
      HLLE_SR(member,eos,indcs,size,coord,m,k,j,il,iu,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
      LLF_GR(member,eos,indcs,size,coord,m,k,j,il,iu,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
      HLLE_GR(member,eos,indcs,size,coord,m,k,j,il,iu,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    }
    member.team_barrier();

    // calculate fluxes of scalars (if any)
    if (nvars > nmhd_) {
      for (int n=nmhd_; n<nvars; ++n) {
        par_for_inner(member, is, ie+1, [&](const int i) {
          if (flx1_(m,IDN,k,j,i) >= 0.0) {
            flx1_(m,n,k,j,i) = flx1_(m,IDN,k,j,i)*wl(n,i);
          } else {
            flx1_(m,n,k,j,i) = flx1_(m,IDN,k,j,i)*wr(n,i);
          }
        });
      }
    }
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
                ScrArray2D<Real>::shmem_size(3, ncells1)) * 3;
    auto &flx2_ = uflx.x2f;
    auto &by_ = b0.x2f;
    auto &e12_ = e1x2;
    auto &e32_ = e3x2;

    // set the loop limits for 2D/3D problems
    if (pmy_pack->pmesh->two_d) {
      kl = ks, ku = ke;
    } else { // 3D
      kl = ks-1, ku = ke+1;
    }
    jl = js-1, ju = je+1;
    if (use_fofc) { jl = js-2, ju = je+2; }

    par_for_outer("mhd_flux2",DevExeSpace(),scr_size,scr_level,0,nmb1, kl, ku,
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

        // Reconstruct qR[j] and qL[j+1], for both W and Bcc
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            DonorCellX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            DonorCellX2(member, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            PiecewiseLinearX2(member, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX2(member,eos_,extrema,true, m,k,j,is-1,ie+1,w0_,wl_jp1,wr);
            PiecewiseParabolicX2(member,eos_,extrema,false,m,k,j,is-1,ie+1,b0_,bl_jp1,br);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX2(member, eos_, true,  m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            WENOZX2(member, eos_, false, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [js,je+1].  MHD RS also computes electric fields, where
        // (IBY) component of flx = E_{x} = -(v x B)_{x} = -(v2*b3 - v3*b2)
        // (IBZ) component of flx = E_{z} = -(v x B)_{z} =  (v2*b1 - v1*b2)
        if (j>jl) {
          // NOTE(@pdmullen): Capture variables prior to if constexpr.
          auto eos = eos_;
          auto indcs = indcs_;
          auto size = size_;
          auto coord = coord_;
          auto by = by_;
          auto flx2 = flx2_;
          auto e12 = e12_;
          auto e32 = e32_;
          if constexpr (rsolver_method_ == MHD_RSolver::advect) {
            Advect(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e12,e32);
          } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
            LLF(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e12,e32);
          } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
            HLLE(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e12,e32);
          } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
            HLLD(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e12,e32);
          } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
            LLF_SR(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e12,e32);
          } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
            HLLE_SR(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e12,e32);
          } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
            LLF_GR(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e12,e32);
          } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
            HLLE_GR(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e12,e32);
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nmhd_) {
          for (int n=nmhd_; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i) {
              if (flx2_(m,IDN,k,j,i) >= 0.0) {
                flx2_(m,n,k,j,i) = flx2_(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx2_(m,n,k,j,i) = flx2_(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end of loop over j
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
                ScrArray2D<Real>::shmem_size(3, ncells1)) * 3;
    auto &flx3_ = uflx.x3f;
    auto &bz_ = b0.x3f;
    auto &e23_ = e2x3;
    auto &e13_ = e1x3;

    // set the loop limits
    kl = ks-1, ku = ke+1;
    if (use_fofc) { kl = ks-2, ku = ke+2; }

    par_for_outer("mhd_flux3",DevExeSpace(), scr_size, scr_level, 0, nmb1, js-1, je+1,
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

        // Reconstruct qR[k] and qL[k+1], for both W and Bcc
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            DonorCellX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            DonorCellX3(member, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            PiecewiseLinearX3(member, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX3(member,eos_,extrema,true, m,k,j,is-1,ie+1,w0_,wl_kp1,wr);
            PiecewiseParabolicX3(member,eos_,extrema,false,m,k,j,is-1,ie+1,b0_,bl_kp1,br);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX3(member, eos_, true,  m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            WENOZX3(member, eos_, false, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [ks,ke+1].  MHD RS also computes electric fields, where
        // (IBY) component of flx = E_{y} = -(v x B)_{y} = -(v3*b1 - v1*b3)
        // (IBZ) component of flx = E_{x} = -(v x B)_{x} =  (v3*b2 - v2*b3)
        if (k>kl) {
          // NOTE(@pdmullen): Capture variables prior to if constexpr.
          auto eos = eos_;
          auto indcs = indcs_;
          auto size = size_;
          auto coord = coord_;
          auto bz = bz_;
          auto flx3 = flx3_;
          auto e23 = e23_;
          auto e13 = e13_;
          if constexpr (rsolver_method_ == MHD_RSolver::advect) {
            Advect(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e23,e13);
          } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
            LLF(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e23,e13);
          } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
            HLLE(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e23,e13);
          } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
            HLLD(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e23,e13);
          } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
            LLF_SR(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e23,e13);
          } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
            HLLE_SR(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e23,e13);
          } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
            LLF_GR(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e23,e13);
          } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
            HLLE_GR(member,eos,indcs,size,coord,
                    m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e23,e13);
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nmhd_) {
          for (int n=nmhd_; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i) {
              if (flx3_(m,IDN,k,j,i) >= 0.0) {
                flx3_(m,n,k,j,i) = flx3_(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx3_(m,n,k,j,i) = flx3_(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end loop over k
    });
  }

  return;
}

// function definitions for each template parameter
template void MHD::CalculateFluxes<MHD_RSolver::advect>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlld>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf_sr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf_gr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace mhd
