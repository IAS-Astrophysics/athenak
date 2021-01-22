//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file calculate_divflux.cpp
//  \brief Calculate divergence of the fluxes for hydro only, no mesh refinement

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro.hpp"
#include "hydro/eos/hydro_eos.hpp"
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"
#include "reconstruct/plm.cpp"
//#include "reconstruct/ppm.cpp"
// include inlined Riemann solvers (double yuck...)
#include "hydro/rsolver/advect.cpp"
#include "hydro/rsolver/llf.cpp"
//#include "hydro/rsolver/hllc.cpp"
//#include "hydro/rsolver/roe.cpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateDivFlux
//  \brief Calculate divergence of the fluxes for hydro only, no mesh refinement

TaskStatus Hydro::HydroDivFlux(Driver *pdrive, int stage)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);

  int nhydro_ = nhydro;
  int nmb = pmy_pack->nmb_thispack;
  auto recon_method = recon_method_;
  auto rsolver_method = rsolver_method_;
  auto &eos = peos->eos_data;
  auto &w0_ = w0;
  auto &divf_ = divf;
  auto &mbsize = pmy_pack->pmb->mbsize;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells1) * 3;
  int scr_level = 0;

  par_for_outer("divflux_x1", DevExeSpace(), scr_size, scr_level,0,(nmb-1), ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> uflux(member.team_scratch(scr_level), nhydro_, ncells1);

      // Reconstruction qR[i] and qL[i+1]
      switch (recon_method)
      {
        case ReconstructionMethod::dc:
          DonorCellX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinearX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
//        case ReconstructionMethod::ppm:
//          PiecewiseParabolicX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
        default:
          break;
      }
      // Sync all threads in the team so that scratch memory is consistent
      member.team_barrier();

      // compute fluxes over [is,ie+1]
      switch (rsolver_method)
      {
        case Hydro_RSolver::advect:
          Advect(member, eos, is, ie+1, IVX, wl, wr, uflux);
          break;
        case Hydro_RSolver::llf:
          LLF(member, eos, is, ie+1, IVX, wl, wr, uflux);
          break;
//        case Hydro_RSolver::hllc:
//          HLLC(member, eos, is, ie+1, IVX, wl, wr, uflux);
//          break;
//        case Hydro_RSolver::roe:
//          Roe(member, eos, is, ie+1, IVX, wl, wr, uflux);
//          break;
        default:
          break;
      }
      member.team_barrier();

      // compute dF/dx1
      for (int n=0; n<nhydro_; ++n) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf_(m,n,k,j,i) = (uflux(n,i+1) - uflux(n,i))/mbsize.dx1.d_view(m);
        });
      }
      member.team_barrier();
    }
  );
  if (!(pmy_pack->pmesh->nx2gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // j-direction

  scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells1) * 4;

  par_for_outer("divflux_x2", DevExeSpace(), scr_size, scr_level,0,(nmb-1), ks, ke,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k)
    {
      AthenaScratch2D<Real> wl_flx(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> wl_jp1(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> uflux_jm1(member.team_scratch(scr_level), nhydro_, ncells1);

      for (int j=js-1; j<=je+1; ++j) {
        // copy Wl from last iteration of j (unless this is the first time through)
        if (j>(js-1)) {
          wl_flx = wl_jp1;
        }

        // Reconstruction qR[j] and qL[j+1]
        switch (recon_method)
        {
          case ReconstructionMethod::dc:
            DonorCellX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
            break;
//          case ReconstructionMethod::ppm:
//            PiecewiseParabolicX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [js,je+1]
        if (j>(js-1)) {
          switch (rsolver_method)
          {
            case Hydro_RSolver::advect:
              Advect(member, eos, is, ie, IVY, wl_flx, wr, wl_flx);
              break;
            case Hydro_RSolver::llf:
              LLF(member, eos, is, ie, IVY, wl_flx, wr, wl_flx);
              break;
//            case Hydro_RSolver::hllc:
//              HLLC(member, eos, is, ie, IVY, wl_flx, wr, wl_flx);
//              break;
//            case Hydro_RSolver::roe:
//              Roe(member, eos, is, ie, IVY, wl_flx, wr, wl_flx);
//              break;
            default:
              break;
          }
        }
        member.team_barrier();

        // Add dF/dx2
        // Fluxes must be summed together to symmetrize round-off error in each dir
        if (j>js) {
          for (int n=0; n<nhydro_; ++n) {
            par_for_inner(member, is, ie, [&](const int i)
            {
              divf_(m,n,k,j-1,i) += (wl_flx(n,i) - uflux_jm1(n,i))/mbsize.dx2.d_view(m);
            });
          }
        }
        member.team_barrier();
  
        // copy flux for use in next iteration
        if (j>(js-1) && j<(je+1)) {
          uflux_jm1 = wl_flx;
        }
      } // end of loop over j
    }
  );
  if (!(pmy_pack->pmesh->nx3gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells1) * 4;

  par_for_outer("divflux_x3", DevExeSpace(), scr_size, scr_level,0,(nmb-1), js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
    {
      AthenaScratch2D<Real> wl_flx(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> wl_kp1(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> uflux_km1(member.team_scratch(scr_level), nhydro_, ncells1);

      for (int k=ks-1; k<=ke+1; ++k) {
        // copy Wl from last iteration of k (unless this is the first time through)
        if (k>(ks-1)) { 
          wl_flx = wl_kp1;
        }

        switch (recon_method)
        {
          case ReconstructionMethod::dc:
            DonorCellX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
            break;
//          case ReconstructionMethod::ppm:
//            PiecewiseParabolicX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [ks,ke+1]
        if (k>(ks-1)) {
          switch (rsolver_method)
          {
            case Hydro_RSolver::advect:
              Advect(member, eos, is, ie, IVZ, wl_flx, wr, wl_flx);
              break;
            case Hydro_RSolver::llf:
              LLF(member, eos, is, ie, IVZ, wl_flx, wr, wl_flx);
              break;
//            case Hydro_RSolver::hllc:
//              HLLC(member, eos, is, ie, IVZ, wl_flx, wr, wl_flx);
//              break;
//            case Hydro_RSolver::roe:
//              Roe(member, eos, is, ie, IVZ, wl_flx, wr, wl_flx);
//              break;
            default:
              break;
          }
        }
        member.team_barrier();

        // Add dF/dx3
        // Fluxes must be summed together to symmetrize round-off error in each dir
        if (k>ks) {
          for (int n=0; n<nhydro_; ++n) {
            par_for_inner(member, is, ie, [&](const int i)
            { 
              divf_(m,n,k-1,j,i) += (wl_flx(n,i) - uflux_km1(n,i))/mbsize.dx3.d_view(m);
            });
          }
        }
        member.team_barrier();

        // copy flux for use in next iteration
        if (k>(ks-1) && k<(ke+1)) {
          uflux_km1 = wl_flx;
        }
      } // end loop over k
    }
  );
  return TaskStatus::complete;
}

} // namespace hydro
