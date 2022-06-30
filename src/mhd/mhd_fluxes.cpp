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
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);

  int &nmhd_ = nmhd;
  int nvars = nmhd + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  bool extrema = false;
  if (recon_method == ReconstructionMethod::ppmx) {
    extrema = true;
  }

  auto &eos = peos->eos_data;
  auto &size = pmy_pack->pmb->mb_size;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  auto &b0_ = bcc0;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
                     ScrArray2D<Real>::shmem_size(3, ncells1)) * 2;
  int scr_level = 0;
  auto flx1 = uflx.x1f;
  auto e31 = e3x1;
  auto e21 = e2x1;
  auto &bx = b0.x1f;

  // set the loop limits for 1D/2D/3D problems
  int jl,ju,kl,ku;
  if (pmy_pack->pmesh->one_d) {
    jl = js, ju = je, kl = ks, ku = ke;
  } else if (pmy_pack->pmesh->two_d) {
    jl = js-1, ju = je+1, kl = ks, ku = ke;
  } else {
    jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
  }

  par_for_outer("mhd_flux1",DevExeSpace(), scr_size, scr_level, 0, nmb1, kl, ku, jl, ju,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> bl(member.team_scratch(scr_level), 3, ncells1);
    ScrArray2D<Real> br(member.team_scratch(scr_level), 3, ncells1);

    // Reconstruct qR[i] and qL[i+1], for both W and Bcc
    switch (recon_method_) {
      case ReconstructionMethod::dc:
        DonorCellX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
        DonorCellX1(member, m, k, j, is-1, ie+1, b0_, bl, br);
        break;
      case ReconstructionMethod::plm:
        PiecewiseLinearX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
        PiecewiseLinearX1(member, m, k, j, is-1, ie+1, b0_, bl, br);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
        PiecewiseParabolicX1(member,eos,extrema, true,  m, k, j, is-1, ie+1, w0_, wl, wr);
        PiecewiseParabolicX1(member,eos,extrema, false, m, k, j, is-1, ie+1, b0_, bl, br);
        break;
      case ReconstructionMethod::wenoz:
        WENOZX1(member, eos, true,  m, k, j, is-1, ie+1, w0_, wl, wr);
        WENOZX1(member, eos, false, m, k, j, is-1, ie+1, b0_, bl, br);
        break;
      default:
        break;
    }
    // Sync all threads in the team so that scratch memory is consistent
    member.team_barrier();

    // compute fluxes over [is,ie+1].  MHD RS also computes electric fields, where
    // (IBY) component of flx = E_{z} = -(v x B)_{z} = -(v1*b2 - v2*b1)
    // (IBZ) component of flx = E_{y} = -(v x B)_{y} =  (v1*b3 - v3*b1)
    if constexpr (rsolver_method_ == MHD_RSolver::advect) {
      Advect(member,eos,indcs,size,coord,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
      LLF(member,eos,indcs,size,coord,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
      HLLE(member,eos,indcs,size,coord,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
      HLLD(member,eos,indcs,size,coord,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
      LLF_SR(member,eos,indcs,size,coord,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
      HLLE_SR(member,eos,indcs,size,coord,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
      LLF_GR(member,eos,indcs,size,coord,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
      HLLE_GR(member,eos,indcs,size,coord,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e31,e21);
    }
    member.team_barrier();

    // calculate fluxes of scalars (if any)
    if (nvars > nmhd_) {
      for (int n=nmhd_; n<nvars; ++n) {
        par_for_inner(member, is, ie+1, [&](const int i) {
          if (flx1(m,IDN,k,j,i) >= 0.0) {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wl(n,i);
          } else {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wr(n,i);
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
    auto flx2 = uflx.x2f;
    auto &by = b0.x2f;
    auto e12 = e1x2;
    auto e32 = e3x2;

    // set the loop limits for 2D/3D problems
    if (pmy_pack->pmesh->two_d) {
      kl = ks, ku = ke;
    } else { // 3D
      kl = ks-1, ku = ke+1;
    }

    par_for_outer("mhd_flux2",DevExeSpace(),scr_size,scr_level,0,nmb1, kl, ku,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k) {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr4(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> scr5(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> scr6(member.team_scratch(scr_level), 3, ncells1);

      for (int j=js-1; j<=je+1; ++j) {
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
            PiecewiseParabolicX2(member,eos,extrema,true, m,k,j,is-1,ie+1,w0_,wl_jp1,wr);
            PiecewiseParabolicX2(member,eos,extrema,false,m,k,j,is-1,ie+1,b0_,bl_jp1,br);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX2(member, eos, true,  m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            WENOZX2(member, eos, false, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [js,je+1].  MHD RS also computes electric fields, where
        // (IBY) component of flx = E_{x} = -(v x B)_{x} = -(v2*b3 - v3*b2)
        // (IBZ) component of flx = E_{z} = -(v x B)_{z} =  (v2*b1 - v1*b2)
        if (j>(js-1)) {
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
              if (flx2(m,IDN,k,j,i) >= 0.0) {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wr(n,i);
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
    auto flx3 = uflx.x3f;
    auto &bz = b0.x3f;
    auto e23 = e2x3;
    auto e13 = e1x3;

    par_for_outer("mhd_flux3",DevExeSpace(), scr_size, scr_level, 0, nmb1, js-1, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr4(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> scr5(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> scr6(member.team_scratch(scr_level), 3, ncells1);

      for (int k=ks-1; k<=ke+1; ++k) {
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
            PiecewiseParabolicX3(member,eos,extrema,true, m,k,j,is-1,ie+1,w0_,wl_kp1,wr);
            PiecewiseParabolicX3(member,eos,extrema,false,m,k,j,is-1,ie+1,b0_,bl_kp1,br);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX3(member, eos, true,  m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            WENOZX3(member, eos, false, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [ks,ke+1].  MHD RS also computes electric fields, where
        // (IBY) component of flx = E_{y} = -(v x B)_{y} = -(v3*b1 - v1*b3)
        // (IBZ) component of flx = E_{x} = -(v x B)_{x} =  (v3*b2 - v2*b3)
        if (k>(ks-1)) {
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
              if (flx3(m,IDN,k,j,i) >= 0.0) {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end loop over k
    });
  }

  // handle excision masks
  if (pmy_pack->pcoord->is_general_relativistic) {
    if (coord.bh_excise) {
      auto &fc_mask_ = pmy_pack->pcoord->fc_mask;

      auto fcorr_x1  = uflx.x1f;
      auto fcorr_x2  = uflx.x2f;
      auto fcorr_x3  = uflx.x3f;

      auto fcorr_e31 = e3x1;
      auto fcorr_e21 = e2x1;
      auto fcorr_e12 = e1x2;
      auto fcorr_e32 = e3x2;
      auto fcorr_e23 = e2x3;
      auto fcorr_e13 = e1x3;

      auto &bcc   = bcc0;
      auto &b0_x1 = b0.x1f;
      auto &b0_x2 = b0.x2f;
      auto &b0_x3 = b0.x3f;
      par_for("excise_flux",DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;

        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        Real x1f = LeftEdgeX  (i-is, indcs.nx1, x1min, x1max);
        Real x2f = LeftEdgeX  (j-js, indcs.nx2, x2min, x2max);
        Real x3f = LeftEdgeX  (k-ks, indcs.nx3, x3min, x3max);

        if (i>(is-1)) {
          if (fc_mask_.x1f(m,k,j,i)) {
            MHDPrim1D wim1;
            wim1.d  = w0_(m,IDN,k,j,i-1);
            wim1.vx = w0_(m,IVX,k,j,i-1);
            wim1.vy = w0_(m,IVY,k,j,i-1);
            wim1.vz = w0_(m,IVZ,k,j,i-1);
            wim1.e  = w0_(m,IEN,k,j,i-1);
            wim1.by = bcc(m,IBY,k,j,i-1);
            wim1.bz = bcc(m,IBZ,k,j,i-1);

            MHDPrim1D wi;
            wi.d  = w0_(m,IDN,k,j,i);
            wi.vx = w0_(m,IVX,k,j,i);
            wi.vy = w0_(m,IVY,k,j,i);
            wi.vz = w0_(m,IVZ,k,j,i);
            wi.e  = w0_(m,IEN,k,j,i);
            wi.by = bcc(m,IBY,k,j,i);
            wi.bz = bcc(m,IBZ,k,j,i);
            Real bxi = b0_x1(m,k,j,i);

            MHDCons1D flux;
            SingleStateLLF_GRMHD(wim1, wi, bxi, x1f, x2v, x3v, IVX, coord, eos, flux);

            fcorr_x1(m,IDN,k,j,i) = flux.d;
            fcorr_x1(m,IM1,k,j,i) = flux.mx;
            fcorr_x1(m,IM2,k,j,i) = flux.my;
            fcorr_x1(m,IM3,k,j,i) = flux.mz;
            fcorr_x1(m,IEN,k,j,i) = flux.e;
            fcorr_e31(m,k,j,i)    = flux.by;
            fcorr_e21(m,k,j,i)    = flux.bz;
          }
        }

        if (j>(js-1)) {
          if (fc_mask_.x2f(m,k,j,i)) {
            MHDPrim1D wjm1;
            wjm1.d  = w0_(m,IDN,k,j-1,i);
            wjm1.vx = w0_(m,IVY,k,j-1,i);
            wjm1.vy = w0_(m,IVZ,k,j-1,i);
            wjm1.vz = w0_(m,IVX,k,j-1,i);
            wjm1.e  = w0_(m,IEN,k,j-1,i);
            wjm1.by = bcc(m,IBZ,k,j-1,i);
            wjm1.bz = bcc(m,IBX,k,j-1,i);

            MHDPrim1D wj;
            wj.d  = w0_(m,IDN,k,j,i);
            wj.vx = w0_(m,IVY,k,j,i);
            wj.vy = w0_(m,IVZ,k,j,i);
            wj.vz = w0_(m,IVX,k,j,i);
            wj.e  = w0_(m,IEN,k,j,i);
            wj.by = bcc(m,IBZ,k,j,i);
            wj.bz = bcc(m,IBX,k,j,i);
            Real bxi = b0_x2(m,k,j,i);

            MHDCons1D flux;
            SingleStateLLF_GRMHD(wjm1, wj, bxi, x1v, x2f, x3v, IVY, coord, eos, flux);

            fcorr_x2(m,IDN,k,j,i) = flux.d;
            fcorr_x2(m,IM2,k,j,i) = flux.mx;
            fcorr_x2(m,IM3,k,j,i) = flux.my;
            fcorr_x2(m,IM1,k,j,i) = flux.mz;
            fcorr_x2(m,IEN,k,j,i) = flux.e;
            fcorr_e12(m,k,j,i)    = flux.by;
            fcorr_e32(m,k,j,i)    = flux.bz;
          }
        }

        if (k>(ks-1)) {
          if (fc_mask_.x3f(m,k,j,i)) {
            MHDPrim1D wkm1;
            wkm1.d  = w0_(m,IDN,k-1,j,i);
            wkm1.vx = w0_(m,IVZ,k-1,j,i);
            wkm1.vy = w0_(m,IVX,k-1,j,i);
            wkm1.vz = w0_(m,IVY,k-1,j,i);
            wkm1.e  = w0_(m,IEN,k-1,j,i);
            wkm1.by = bcc(m,IBX,k-1,j,i);
            wkm1.bz = bcc(m,IBY,k-1,j,i);

            MHDPrim1D wk;
            wk.d  = w0_(m,IDN,k,j,i);
            wk.vx = w0_(m,IVZ,k,j,i);
            wk.vy = w0_(m,IVX,k,j,i);
            wk.vz = w0_(m,IVY,k,j,i);
            wk.e  = w0_(m,IEN,k,j,i);
            wk.by = bcc(m,IBX,k,j,i);
            wk.bz = bcc(m,IBY,k,j,i);
            Real bxi = b0_x3(m,k,j,i);

            MHDCons1D flux;
            SingleStateLLF_GRMHD(wkm1, wk, bxi, x1v, x2v, x3f, IVZ, coord, eos, flux);

            fcorr_x3(m,IDN,k,j,i) = flux.d;
            fcorr_x3(m,IM3,k,j,i) = flux.mx;
            fcorr_x3(m,IM1,k,j,i) = flux.my;
            fcorr_x3(m,IM2,k,j,i) = flux.mz;
            fcorr_x3(m,IEN,k,j,i) = flux.e;
            fcorr_e23(m,k,j,i)    = flux.by;
            fcorr_e13(m,k,j,i)    = flux.bz;
          }
        }
      });
    }
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
