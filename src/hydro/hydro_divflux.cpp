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
/***
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;

  // capture Hydro and EOS class variables used in kernel
  int nhydro_ = nhydro;
  int nvars_ = nhydro + nscalars;
  ReconstructionMethod &recon_method = recon_method_;
  Hydro_RSolver &rsolver_method = rsolver_method_;
  EOS_Data &eos = pmb->phydro->peos->eos_data;
  AthenaArray4D<Real> &w0_ = w0;
  AthenaArray4D<Real> &divf_ = divf;

  //--------------------------------------------------------------------------------------
  // i-direction

  int ncells1 = pmb->mb_cells.nx1 + 2*(pmb->mb_cells.ng);
  const int scr_level = 0;
  size_t scr_size = AthenaScratch2D<Real>::shmem_size(nvars_, ncells1) * 2;
  Real &dx1 = pmb->mb_cells.dx1;

  par_for_outer("divflux_x1", pmb->exe_space, scr_size, scr_level, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int k, const int j)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nvars_, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nvars_, ncells1);

      // Reconstruction qR[i] and qL[i+1]
      AthenaArray2DSlice<Real> qi = Kokkos::subview(w0_,Kokkos::ALL(),k,j,Kokkos::ALL());
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
      if (nvars_ > nhydro_) {
        for (int n=nhydro_; n<nvars_; ++n) {
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
      for (int n=0; n<nvars_; ++n) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf_(n,k,j,i) = (wl(n,i+1) - wl(n,i))/dx1;
        });
      }
      member.team_barrier();
    }
  );
  if (!(pmesh_->nx2gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // j-direction

  // capture variables used in kernel
  int ncells2 = pmb->mb_cells.nx2 + 2*(pmb->mb_cells.ng);
  scr_size = AthenaScratch2D<Real>::shmem_size(nvars_, ncells2) * 2;
  Real &dx2 = pmb->mb_cells.dx2;

  par_for_outer("divflux_x2", pmb->exe_space, scr_size, scr_level, ks, ke, is, ie,
    KOKKOS_LAMBDA(TeamMember_t member, const int k, const int i)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nvars_, ncells2);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nvars_, ncells2);

      // Reconstruction qR[j] and qL[j+1]
      AthenaArray2DSlice<Real> qj = Kokkos::subview(w0_,Kokkos::ALL(),k,Kokkos::ALL(),i);
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
      if (nvars_ > nhydro_) {
        for (int n=nhydro_; n<nvars_; ++n) {
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
      for (int n=0; n<nvars_; ++n) {
        par_for_inner(member, js, je, [&](const int j)
        {
          divf_(n,k,j,i) += (wl(n,j+1) - wl(n,j))/dx2;
        });
      }
      member.team_barrier();
    }
  );
  if (!(pmesh_->nx3gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  // capture variables used in kernel
  int ncells3 = pmb->mb_cells.nx3 + 2*(pmb->mb_cells.ng);
  scr_size = AthenaScratch2D<Real>::shmem_size(nvars_, ncells3) * 2;
  Real &dx3 = pmb->mb_cells.dx3;

  par_for_outer("divflux_x3", pmb->exe_space, scr_size, scr_level, js, je, is, ie,
    KOKKOS_LAMBDA(TeamMember_t member, const int j, const int i)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nvars_, ncells3);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nvars_, ncells3);

      // Reconstruction qR[k] and qL[k+1]
      AthenaArray2DSlice<Real> qk = Kokkos::subview(w0_,Kokkos::ALL(),Kokkos::ALL(),j,i);
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
      if (nvars_ > nhydro_) {
        for (int n=nhydro_; n<nvars_; ++n) {
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
      for (int n=0; n<nvars_; ++n) {
        par_for_inner(member, ks, ke, [&](const int k)
        { 
          divf_(n,k,j,i) += (wl(n,k+1) - wl(n,k))/dx3;
        });
      }
      member.team_barrier();
    }
  );
****/
  return TaskStatus::complete;
}

} // namespace hydro
