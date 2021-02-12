//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes.cpp
//  \brief Calculate fluxes of the conserved variables, and area-averaged EMFs, on
//   cell faces for mhd

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"
#include "reconstruct/plm.cpp"
#include "reconstruct/ppm.cpp"
#include "reconstruct/wenoz.cpp"
// include inlined Riemann solvers (double yuck...)
#include "mhd/rsolvers/advect_mhd.cpp"
#include "mhd/rsolvers/llf_mhd.cpp"
//#include "mhd/rsolvers/hlld.cpp"
//#include "mhd/rsolvers/roe_mhd.cpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::CalcFlux
//  \brief Calculate fluxes of conserved variables, and face-centered area-averaged EMFs
//  for evolution of magnetic field

TaskStatus MHD::CalcFluxes(Driver *pdriver, int stage)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);

  int &nmhd_  = nmhd;
  int nvars = nmhd + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto recon_method = recon_method_;
  auto rsolver_method = rsolver_method_;
  auto &w0_ = w0;
  auto &b0_ = bcc0;
  auto &eos = peos->eos_data;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
                     ScrArray2D<Real>::shmem_size(3, ncells1)) * 2;
  int scr_level = 0;
  auto flx1 = uflx.x1f;
  auto e3x1_ = e3x1;
  auto e2x1_ = e2x1;
  auto &bx = b0.x1f;

  // set the loop limits for 1D/2D/3D problems
  int jl,ju,kl,ku;
  if (pmy_pack->pmesh->nx2gt1) {
    if (pmy_pack->pmesh->nx3gt1) { // 3D
      jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    } else { // 2D
      jl = js-1, ju = je+1, kl = ks, ku = ke;
    }
  } else { // 1D
    jl = js, ju = je, kl = ks, ku = ke;
  }

  par_for_outer("mhd_flux1",DevExeSpace(), scr_size, scr_level, 0, nmb1, kl, ku, jl, ju,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> bl(member.team_scratch(scr_level), 3, ncells1);
      ScrArray2D<Real> br(member.team_scratch(scr_level), 3, ncells1);

      // Reconstruct qR[i] and qL[i+1], for both W and Bcc
      switch (recon_method)
      {
        case ReconstructionMethod::dc:
          DonorCellX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          DonorCellX1(member, m, k, j, is-1, ie+1, b0_, bl, br);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinearX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          PiecewiseLinearX1(member, m, k, j, is-1, ie+1, b0_, bl, br);
          break;
        case ReconstructionMethod::ppm:
          PiecewiseParabolicX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          PiecewiseParabolicX1(member, m, k, j, is-1, ie+1, b0_, bl, br);
          break;
        case ReconstructionMethod::wenoz:
          WENOZX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          WENOZX1(member, m, k, j, is-1, ie+1, b0_, bl, br);
          break;
        default:
          break;
      }
      // Sync all threads in the team so that scratch memory is consistent
      member.team_barrier();

      // compute fluxes over [is,ie+1]
      // flx1(IBY) = (v1*b2 - v2*b1) = -EMFZ
      // flx1(IBZ) = (v1*b3 - v3*b1) =  EMFY
      switch (rsolver_method)
      {
        case MHD_RSolver::advect:
          Advect(member,eos,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e3x1_,e2x1_);
          break;
        case MHD_RSolver::llf:
          LLF(member,eos,m,k,j,is,ie+1,IVX,wl,wr,bl,br,bx,flx1,e3x1_,e2x1_);
          break;
//        case MHD_RSolver::hllc:
//          HLLC(member, eos, is, ie+1, IVX, wl, wr, uflux);
//          break;
//        case MHD_RSolver::roe:
//          Roe(member, eos, is, ie+1, IVX, wl, wr, uflux);
//          break;
        default:
          break;
      }
      member.team_barrier();

      // calculate fluxes of scalars (if any)
      if (nvars > nmhd_) {
        for (int n=nmhd_; n<nvars; ++n) {
          par_for_inner(member, is, ie+1, [&](const int i)
          {
            if (flx1(m,IDN,k,j,i) >= 0.0) {
              flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wl(n,i);
            } else {
              flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wr(n,i);
            }
          });
        }
      }

    }
  );
  if (!(pmy_pack->pmesh->nx2gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // j-direction

  scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
              ScrArray2D<Real>::shmem_size(3, ncells1)) * 3;
  auto flx2 = uflx.x2f;
  auto &by = b0.x2f;
  auto e1x2_ = e1x2;
  auto e3x2_ = e3x2;

  // set the loop limits for 2D/3D problems
  if (pmy_pack->pmesh->nx3gt1) { // 3D
    kl = ks-1, ku = ke+1;
  } else { // 2D
    kl = ks, ku = ke;
  }

  par_for_outer("mhd_flux2",DevExeSpace(),scr_size,scr_level,0,nmb1, kl, ku,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k)
    {
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
        switch (recon_method)
        {
          case ReconstructionMethod::dc:
            DonorCellX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            DonorCellX2(member, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            PiecewiseLinearX2(member, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          case ReconstructionMethod::ppm:
            PiecewiseParabolicX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            PiecewiseParabolicX2(member, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            WENOZX2(member, m, k, j, is-1, ie+1, b0_, bl_jp1, br);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [js,je+1].
        // flx2(IBY) = (v2*b3 - v3*b2) = -EMFX
        // flx2(IBZ) = (v2*b1 - v1*b2) =  EMFZ
        if (j>(js-1)) {
          switch (rsolver_method)
          {
            case MHD_RSolver::advect:
              Advect(member,eos,m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e1x2_,e3x2_);
              break;
            case MHD_RSolver::llf:
              LLF(member,eos,m,k,j,is-1,ie+1,IVY,wl,wr,bl,br,by,flx2,e1x2_,e3x2_);
              break;
//            case MHD_RSolver::hllc:
//              HLLC(member, eos, is, ie, IVY, wl, wr, uf);
//              break;
//            case MHD_RSolver::roe:
//              Roe(member, eos, is, ie, IVY, wl, wr, uf);
//              break;
            default:
              break;
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nmhd_) {
          for (int n=nmhd_; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i)
            {
              if (flx2(m,IDN,k,j,i) >= 0.0) {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }

      } // end of loop over j
    }
  );
  if (!(pmy_pack->pmesh->nx3gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
              ScrArray2D<Real>::shmem_size(3, ncells1)) * 3;
  auto flx3 = uflx.x3f;
  auto &bz = b0.x3f;
  auto e2x3_ = e2x3;
  auto e1x3_ = e1x3;

  par_for_outer("mhd_flux3",DevExeSpace(), scr_size, scr_level, 0, nmb1, js-1, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
    {
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
        switch (recon_method)
        {
          case ReconstructionMethod::dc:
            DonorCellX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            DonorCellX3(member, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            PiecewiseLinearX3(member, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          case ReconstructionMethod::ppm:
            PiecewiseParabolicX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            PiecewiseParabolicX3(member, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            WENOZX3(member, m, k, j, is-1, ie+1, b0_, bl_kp1, br);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [ks,ke+1].
        // flx3(IBY) = (v3*b1 - v1*b3) = -EMFY
        // flx3(IBZ) = (v3*b2 - v2*b3) =  EMFX
        if (k>(ks-1)) {
          switch (rsolver_method)
          {
            case MHD_RSolver::advect:
              Advect(member,eos,m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e2x3_,e1x3_);
              break;
            case MHD_RSolver::llf:
              LLF(member,eos,m,k,j,is-1,ie+1,IVZ,wl,wr,bl,br,bz,flx3,e2x3_,e1x3_);
              break;
//            case MHD_RSolver::hllc:
//              HLLC(member, eos, is, ie, IVZ, wl, wr, uf);
//              break;
//            case MHD_RSolver::roe:
//              Roe(member, eos, is, ie, IVZ, wl, wr, uf);
//              break;
            default:
              break;
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nmhd_) {
          for (int n=nmhd_; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i)
            {
              if (flx3(m,IDN,k,j,i) >= 0.0) {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }

      } // end loop over k
    }
  );
  return TaskStatus::complete;
}

} // namespace mhd
