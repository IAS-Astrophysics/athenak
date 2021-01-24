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
#include "reconstruct/ppm.cpp"
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
  int ng = pmy_pack->mb_cells.ng;
  int ncells1 = pmy_pack->mb_cells.nx1 + 2*ng;

  int nhydro_ = nhydro;
  int nvars = nhydro + nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto recon_method = recon_method_;
  auto rsolver_method = rsolver_method_;
  auto &eos = peos->eos_data;
  auto &w0_ = w0;
  auto &divf_ = divf;
  auto &mbsize = pmy_pack->pmb->mbsize;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
  int scr_level = 0;

  par_for_outer("divflux_x1",DevExeSpace(), scr_size, scr_level,0,(nmb-1), ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> uflux(member.team_scratch(scr_level), nvars, ncells1);

      // Reconstruct qR[i] and qL[i+1]
      switch (recon_method)
      {
        case ReconstructionMethod::dc:
          DonorCellX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinearX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
        case ReconstructionMethod::ppm:
          PiecewiseParabolicX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
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

      // calculate fluxes of scalars
      if (nvars > nhydro_) {
        for (int n=nhydro_; n<nvars; ++n) {
          par_for_inner(member, is, ie+1, [&](const int i)
          {
            if (uflux(IDN,i) >= 0.0) {
              uflux(n,i) = uflux(IDN,i)*wl(n,i);
            } else {
              uflux(n,i) = uflux(IDN,i)*wr(n,i);
            }
          });
        }
        member.team_barrier();
      }

      // compute dF/dx1
      for (int n=0; n<nvars; ++n) {
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

  scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 4;

  par_for_outer("divflux_x2",DevExeSpace(), scr_size, scr_level, 0, (nmb-1), ks, ke,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k)
    {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr4(member.team_scratch(scr_level), nvars, ncells1);

      for (int j=js-1; j<=je+1; ++j) {
        // Permute scratch arrays.
        // Note wr/uf always the same --> Riemann solver returns flux in input wr array
        auto wl     = scr1;
        auto wl_jp1 = scr2;
        auto wr     = scr3;
        auto uf     = scr3;
        auto uf_jm1 = scr4;
        if ((j%2) == 0) {
          wl     = scr2;
          wl_jp1 = scr1;
          wr     = scr4;
          uf     = scr4;
          uf_jm1 = scr3;
        }

        // Reconstruct qR[j] and qL[j+1]
        switch (recon_method)
        {
          case ReconstructionMethod::dc:
            DonorCellX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::ppm:
            PiecewiseParabolicX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [js,je+1].  RS returns flux in input wr array
        if (j>(js-1)) {
          switch (rsolver_method)
          {
            case Hydro_RSolver::advect:
              Advect(member, eos, is, ie, IVY, wl, wr, uf);
              break;
            case Hydro_RSolver::llf:
              LLF(member, eos, is, ie, IVY, wl, wr, uf);
              break;
//            case Hydro_RSolver::hllc:
//              HLLC(member, eos, is, ie, IVY, wl, wr, uf);
//              break;
//            case Hydro_RSolver::roe:
//              Roe(member, eos, is, ie, IVY, wl, wr, uf);
//              break;
            default:
              break;
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars
        if (nvars > nhydro_) {
          for (int n=nhydro_; n<nvars; ++n) {
            par_for_inner(member, js, je+1, [&](const int j)
            {
              if (uf(IDN,j) >= 0.0) {
                uf(n,j) = uf(IDN,j)*wl(n,j);
              } else {
                uf(n,j) = uf(IDN,j)*wr(n,j);
              }
            });
          }
          member.team_barrier();
        } 

        // Add dF/dx2
        // Fluxes must be summed together to symmetrize round-off error in each dir
        if (j>js) {
          for (int n=0; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i)
            {
              divf_(m,n,k,j-1,i) += (uf(n,i) - uf_jm1(n,i))/mbsize.dx2.d_view(m);
            });
          }
          member.team_barrier();
        }
  
      } // end of loop over j
    }
  );
  if (!(pmy_pack->pmesh->nx3gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 4;

  par_for_outer("divflux_x3",DevExeSpace(), scr_size, scr_level, 0, (nmb-1), js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
    {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr4(member.team_scratch(scr_level), nvars, ncells1);

      for (int k=ks-1; k<=ke+1; ++k) {
        // Permute scratch arrays.
        // Note wr/uf always the same --> Riemann solver returns flux in input wr array
        auto wl     = scr1;
        auto wl_kp1 = scr2;
        auto wr     = scr3;
        auto uf     = scr3;
        auto uf_km1 = scr4;
        if ((k%2) == 0) {
          wl     = scr2;
          wl_kp1 = scr1;
          wr     = scr4;
          uf     = scr4;
          uf_km1 = scr3;
        }

        // Reconstruct qR[k] and qL[k+1]
        switch (recon_method)
        {
          case ReconstructionMethod::dc:
            DonorCellX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::ppm:
            PiecewiseParabolicX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [ks,ke+1].  RS returns flux in input wr array
        if (k>(ks-1)) {
          switch (rsolver_method)
          {
            case Hydro_RSolver::advect:
              Advect(member, eos, is, ie, IVZ, wl, wr, uf);
              break;
            case Hydro_RSolver::llf:
              LLF(member, eos, is, ie, IVZ, wl, wr, uf);
              break;
//            case Hydro_RSolver::hllc:
//              HLLC(member, eos, is, ie, IVZ, wl, wr, uf);
//              break;
//            case Hydro_RSolver::roe:
//              Roe(member, eos, is, ie, IVZ, wl, wr, uf);
//              break;
            default:
              break;
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars
        if (nvars > nhydro_) {
          for (int n=nhydro_; n<nvars; ++n) {
            par_for_inner(member, ks, ke+1, [&](const int k)
            {
              if (uf(IDN,k) >= 0.0) {
                uf(n,k) = uf(IDN,k)*wl(n,k);
              } else {
                uf(n,k) = uf(IDN,k)*wr(n,k);
              }
            });
          }
          member.team_barrier();
        }

        // Add dF/dx3
        // Fluxes must be summed together to symmetrize round-off error in each dir
        if (k>ks) {
          for (int n=0; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i)
            { 
              divf_(m,n,k-1,j,i) += (uf(n,i) - uf_km1(n,i))/mbsize.dx3.d_view(m);
            });
          }
          member.team_barrier();
        }

      } // end loop over k
    }
  );
  return TaskStatus::complete;
}

} // namespace hydro
