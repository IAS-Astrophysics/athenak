//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_divflux.cpp
//  \brief

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/eos/hydro_eos.hpp"
#include "hydro.hpp"
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"
#include "reconstruct/plm.cpp"
#include "reconstruct/ppm.cpp"
// include inlined Riemann solvers (double yuck...)
#include "hydro/rsolver/advect.cpp"
#include "hydro/rsolver/llf.cpp"
#include "hydro/rsolver/hllc.cpp"
//#include "hydro/rsolver/roe.cpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroDivFlux
//  \brief Function that computes 3D array of divergence of fluxes, by reconstructing L/R
//  states, solving Riemann problem to compute fluxes, then summing into div(F) array

TaskStatus Hydro::HydroDivFlux(Driver *pdrive, int stage)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;

  // capture Hydro and EOS class variables used in kernel
  int nhyd  = nhydro;
  int nvars = nhydro + nscalars;
  int nmb = pmy_pack->nmb_thispack;
  ReconstructionMethod &recon_method = recon_method_;
  Hydro_RSolver &rsolver_method = rsolver_method_;
  EOS_Data &eos = peos->eos_data;
  AthenaArray5D<Real> &w0_ = w0;
  AthenaArray5D<Real> &divf_ = divf;
  auto &mbsize = pmy_pack->pmb->d_mbsize;

  //--------------------------------------------------------------------------------------
  // i-direction

  int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);
  const int scr_level = 0;
  size_t scr_size = AthenaScratch2D<Real>::shmem_size(nvars, ncells1) * 2;

  par_for_outer("divflux_x1", DevExeSpace(), scr_size, scr_level,0,(nmb-1),ks,ke,js,je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);

      // Reconstruction qR[i] and qL[i+1]
      AthenaArray2DSlice<Real> qi = Kokkos::subview(w0_,m,Kokkos::ALL(),k,j,Kokkos::ALL());
      switch (recon_method)
      {
        case ReconstructionMethod::dc:
          DonorCell(member, is-1, ie+1, qi, wl, wr);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinear(member, is-1, ie+1, qi, wl, wr);
          break;
        case ReconstructionMethod::ppm:
          PiecewiseParabolic(member, is-1, ie+1, qi, wl, wr);
          break;
        default:
          break;
      }
      member.team_barrier();

      // compute hydro fluxes over [is,ie+1]
      switch (rsolver_method)
      {
        case Hydro_RSolver::advect:
          Advect(member, eos, is, ie+1, IVX, wl, wr, wl);
          break;
        case Hydro_RSolver::llf:
          LLF(member, eos, is, ie+1, IVX, wl, wr, wl);
          break;
        case Hydro_RSolver::hllc:
          HLLC(member, eos, is, ie+1, IVX, wl, wr, wl);
          break;
//        case Hydro_RSolver::roe:
//          Roe(member, eos, is, ie+1, IVX, wl, wr, wl);
//          break;
        default:
          break;
      }
      member.team_barrier();

      // calculate fluxes of scalars
      if (nvars > nhyd) {
        for (int n=nhyd; n<nvars; ++n) {
          par_for_inner(member, is, ie+1, [&](const int i)
          {
            if (wl(IDN,i) >= 0.0) {
              wl(n,i) = wl(IDN,i)*wl(n,i);
            } else {
              wl(n,i) = wl(IDN,i)*wr(n,i);
            }
          });
        }
        member.team_barrier();
      }

      // compute dF/dx1
      for (int n=0; n<nvars; ++n) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf_(m,n,k,j,i) = (wl(n,i+1) - wl(n,i))/mbsize(m,6);
        });
      }
      member.team_barrier();
    }
  );
  if (!(pmy_pack->pmesh->nx2gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // j-direction

  // capture variables used in kernel
  int ncells2 = pmy_pack->mb_cells.nx2 + 2*(pmy_pack->mb_cells.ng);
  scr_size = AthenaScratch2D<Real>::shmem_size(nvars, ncells2) * 2;

  par_for_outer("divflux_x2", DevExeSpace(), scr_size, scr_level,0,(nmb-1),ks,ke,is,ie,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int i)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nvars, ncells2);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nvars, ncells2);

      // Reconstruction qR[j] and qL[j+1]
      AthenaArray2DSlice<Real> qj = Kokkos::subview(w0_,m,Kokkos::ALL(),k,Kokkos::ALL(),i);
      switch (recon_method)
      {
        case ReconstructionMethod::dc:
          DonorCell(member, js-1, je+1, qj, wl, wr);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinear(member, js-1, je+1, qj, wl, wr);
          break;
        case ReconstructionMethod::ppm:
          PiecewiseParabolic(member, js-1, je+1, qj, wl, wr);
          break;
        default:
          break;
      }
      member.team_barrier();

      // compute hydro fluxes over [js,je+1]
      switch (rsolver_method)
      {
        case Hydro_RSolver::advect:
          Advect(member, eos, js, je+1, IVY, wl, wr, wl);
          break;
        case Hydro_RSolver::llf:
          LLF(member, eos, js, je+1, IVY, wl, wr, wl);
          break;
        case Hydro_RSolver::hllc:
          HLLC(member, eos, js, je+1, IVY, wl, wr, wl);
          break;
//        case Hydro_RSolver::roe:
//          Roe(member, eos, js, je+1, IVY, wl, wr, wl);
//          break;
        default:
          break;
      }
      member.team_barrier();

      // calculate fluxes of scalars
      if (nvars > nhyd) {
        for (int n=nhyd; n<nvars; ++n) {
          par_for_inner(member, js, je+1, [&](const int j)
          {
            if (wl(IDN,j) >= 0.0) {
              wl(n,j) = wl(IDN,j)*wl(n,j);
            } else {
              wl(n,j) = wl(IDN,j)*wr(n,j);
            }
          });
        }
        member.team_barrier();
      }

      // Add dF/dx2
      // Fluxes must be summed together to symmetrize round-off error in each dir
      for (int n=0; n<nvars; ++n) {
        par_for_inner(member, js, je, [&](const int j)
        {
          divf_(m,n,k,j,i) += (wl(n,j+1) - wl(n,j))/mbsize(m,7);
        });
      }
      member.team_barrier();
    }
  );
  if (!(pmy_pack->pmesh->nx3gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  // capture variables used in kernel
  int ncells3 = pmy_pack->mb_cells.nx3 + 2*(pmy_pack->mb_cells.ng);
  scr_size = AthenaScratch2D<Real>::shmem_size(nvars, ncells3) * 2;

  par_for_outer("divflux_x3", DevExeSpace(), scr_size, scr_level,0,(nmb-1),js,je,is,ie,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j, const int i)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nvars, ncells3);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nvars, ncells3);

      // Reconstruction qR[k] and qL[k+1]
      AthenaArray2DSlice<Real> qk = Kokkos::subview(w0_,m,Kokkos::ALL(),Kokkos::ALL(),j,i);
      switch (recon_method)
      {
        case ReconstructionMethod::dc:
          DonorCell(member, ks-1, ke+1, qk, wl, wr);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinear(member, ks-1, ke+1, qk, wl, wr);
          break;
        case ReconstructionMethod::ppm:
          PiecewiseParabolic(member, ks-1, ke+1, qk, wl, wr);
          break;
        default:
          break;
      }
      member.team_barrier();

      // compute hydro fluxes over [ks,ke+1]
      switch (rsolver_method)
      {
        case Hydro_RSolver::advect:
          Advect(member, eos, ks, ke+1, IVZ, wl, wr, wl);
          break;
        case Hydro_RSolver::llf:
          LLF(member, eos, ks, ke+1, IVZ, wl, wr, wl);
          break;
        case Hydro_RSolver::hllc:
          HLLC(member, eos, ks, ke+1, IVZ, wl, wr, wl);
          break;
//        case Hydro_RSolver::roe:
//          Roe(member, eos, ks, ke+1, IVZ, wl, wr, wl);
//          break;
        default:
          break;
      }
      member.team_barrier();

      // calculate fluxes of scalars
      if (nvars > nhyd) {
        for (int n=nhyd; n<nvars; ++n) {
          par_for_inner(member, ks, ke+1, [&](const int k)
          {
            if (wl(IDN,k) >= 0.0) {
              wl(n,k) = wl(IDN,k)*wl(n,k);
            } else {
              wl(n,k) = wl(IDN,k)*wr(n,k);
            }
          });
        }
        member.team_barrier();
      }

      // Add dF/dx2
      // Add dF/dx3
      // Fluxes must be summed together to symmetrize round-off error in each dir
      for (int n=0; n<nvars; ++n) {
        par_for_inner(member, ks, ke, [&](const int k)
        { 
          divf_(m,n,k,j,i) += (wl(n,k+1) - wl(n,k))/mbsize(m,8);
        });
      }
      member.team_barrier();
    }
  );
  return TaskStatus::complete;
}
} // namespace hydro
