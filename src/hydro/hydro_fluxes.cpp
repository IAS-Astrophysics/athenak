//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes.cpp
//  \brief Calculate 3D fluxes for hydro

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro.hpp"
#include "eos/eos.hpp"
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"
#include "reconstruct/plm.cpp"
#include "reconstruct/ppm.cpp"
#include "reconstruct/wenoz.cpp"
// include inlined Riemann solvers (double yuck...)
#include "hydro/rsolvers/advect.cpp"
#include "hydro/rsolvers/llf.cpp"
#include "hydro/rsolvers/llf_rel.cpp"
#include "hydro/rsolvers/hllc.cpp"
#include "hydro/rsolvers/hllc_rel.cpp"
//#include "hydro/rsolvers/roe.cpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalcFluxes
//  \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes

TaskStatus Hydro::CalcFluxes(Driver *pdriver, int stage)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);
  
  int nhyd  = nhydro;
  int nvars = nhydro + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method = recon_method_;
  const auto rsolver_method = rsolver_method_;
  auto &w0_ = w0;
  auto &eos = peos->eos_data;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 2;
  int scr_level = 0;
  auto flx1 = uflx.x1f;

  par_for_outer("hflux_x1",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);

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
        case ReconstructionMethod::wenoz:
          WENOZX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
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
          Advect(member, eos, m, k, j, is, ie+1, IVX, wl, wr, flx1);
          break;
        case Hydro_RSolver::llf:
          LLF(member, eos, m, k, j, is, ie+1, IVX, wl, wr, flx1);
          break;
        case Hydro_RSolver::hllc:
          HLLC(member, eos, m, k, j, is, ie+1, IVX, wl, wr, flx1);
          break;
//        case Hydro_RSolver::roe:
//          Roe(member, eos, m, k, j, is, ie+1, IVX, wl, wr, flx1);
//          break;
        case Hydro_RSolver::llf_rel:
          LLF_rel(member, eos, m, k, j, is, ie+1, IVX, wl, wr, flx1);
          break;
        case Hydro_RSolver::hllc_rel:
          HLLC_rel(member, eos, m, k, j, is, ie+1, IVX, wl, wr, flx1);
          break;
        default:
          break;
      }
      member.team_barrier();

      // calculate fluxes of scalars (if any)
      if (nvars > nhyd) {
        for (int n=nhyd; n<nvars; ++n) {
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

  scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
  auto flx2 = uflx.x2f;

  par_for_outer("hflux_x2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k)
    {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);

      for (int j=js-1; j<=je+1; ++j) {
        // Permute scratch arrays.
        auto wl     = scr1;
        auto wl_jp1 = scr2;
        auto wr     = scr3;
        if ((j%2) == 0) {
          wl     = scr2;
          wl_jp1 = scr1;
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
          case ReconstructionMethod::wenoz:
            WENOZX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
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
              Advect(member, eos, m, k, j, is, ie, IVY, wl, wr, flx2);
              break;
            case Hydro_RSolver::llf:
              LLF(member, eos, m, k, j, is, ie, IVY, wl, wr, flx2);
              break;
            case Hydro_RSolver::hllc:
              HLLC(member, eos, m, k, j, is, ie, IVY, wl, wr, flx2);
              break;
//            case Hydro_RSolver::roe:
//              Roe(member, eos, m, k, j, is, ie, IVY, wl, wr, flx2);
//              break;
            case Hydro_RSolver::llf_rel:
              LLF_rel(member, eos, m, k, j, is, ie, IVY, wl, wr, flx2);
              break;
            case Hydro_RSolver::hllc_rel:
              HLLC_rel(member, eos, m, k, j, is, ie, IVY, wl, wr, flx2);
              break;
            default:
              break;
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nhyd) {
          for (int n=nhyd; n<nvars; ++n) {
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

  scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
  auto flx3 = uflx.x3f;

  par_for_outer("hflux_x3",DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
    {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);

      for (int k=ks-1; k<=ke+1; ++k) {
        // Permute scratch arrays.
        auto wl     = scr1;
        auto wl_kp1 = scr2;
        auto wr     = scr3;
        if ((k%2) == 0) {
          wl     = scr2;
          wl_kp1 = scr1;
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
          case ReconstructionMethod::wenoz:
            WENOZX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
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
              Advect(member, eos, m, k, j, is, ie, IVZ, wl, wr, flx3);
              break;
            case Hydro_RSolver::llf:
              LLF(member, eos, m, k, j, is, ie, IVZ, wl, wr, flx3);
              break;
            case Hydro_RSolver::hllc:
              HLLC(member, eos, m, k, j, is, ie, IVZ, wl, wr, flx3);
              break;
//            case Hydro_RSolver::roe:
//              Roe(member, eos, m, k, j, is, ie, IVZ, wl, wr, flx3);
//              break;
            case Hydro_RSolver::llf_rel:
              LLF_rel(member, eos, m, k, j, is, ie, IVZ, wl, wr, flx3);
              break;
            case Hydro_RSolver::hllc_rel:
              HLLC_rel(member, eos, m, k, j, is, ie, IVZ, wl, wr, flx3);
              break;
            default:
              break;
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nhyd) {
          for (int n=nhyd; n<nvars; ++n) {
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

} // namespace hydro
