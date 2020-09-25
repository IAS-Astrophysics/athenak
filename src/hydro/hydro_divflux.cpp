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
#include "hydro/eos/eos.hpp"
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"
#include "reconstruct/plm.cpp"
#include "reconstruct/ppm.cpp"
// include inlined Riemann solvers (double yuck...)
#include "hydro/rsolver/advect.cpp"
#include "hydro/rsolver/llf.cpp"
#include "hydro/rsolver/hllc.cpp"
#include "hydro/rsolver/roe.cpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateDivFlux
//  \brief Calculate divergence of the fluxes for hydro only, no mesh refinement

TaskStatus Hydro::HydroDivFlux(Driver *pdrive, int stage)
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;

  int nhydro_ = nhydro;
  auto recon_method = recon_method_;
  auto rsolver_method = rsolver_method_;
  auto &eos = pmb->phydro->peos->eos_data;
  auto &w0_ = w0;
  auto &divf_ = divf;

  //--------------------------------------------------------------------------------------
  // i-direction

  int ncells1 = pmb->mb_cells.nx1 + 2*(pmb->mb_cells.ng);
  size_t scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells1) * 3;
  int scr_level = 1;
  Real &dx1 = pmb->mb_cells.dx1;

  par_for_outer("divflux_x1", pmb->exe_space, scr_size, scr_level, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int k, const int j)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro_, ncells1);
      AthenaScratch2D<Real> uflux(member.team_scratch(scr_level), nhydro_, ncells1);

      AthenaArray2DSlice<Real> qi = Kokkos::subview(w0_,Kokkos::ALL(),k,j,Kokkos::ALL());
      // Reconstruction qR[i] and qL[i+1]
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

      // compute fluxes over [is,ie+1]
      switch (rsolver_method)
      {
        case RiemannSolver::advect:
          Advect(member, eos, is, ie+1, IVX, wl, wr, uflux);
          break;
        case RiemannSolver::llf:
          LLF(member, eos, is, ie+1, IVX, wl, wr, uflux);
          break;
        case RiemannSolver::hllc:
          HLLC(member, eos, is, ie+1, IVX, wl, wr, uflux);
          break;
        case RiemannSolver::roe:
          Roe(member, eos, is, ie+1, IVX, wl, wr, uflux);
          break;
        default:
          break;
      }

      // compute dF/dx1
      for (int n=0; n<nhydro_; ++n) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf_(n,k,j,i) = (uflux(n,i+1) - uflux(n,i))/dx1;
        });
      }
    }
  );
  if (!(pmesh_->nx2gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // j-direction

  int ncells2 = pmb->mb_cells.nx2 + 2*(pmb->mb_cells.ng);
  scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells2) * 3;
  scr_level = 1;
  Real &dx2 = pmb->mb_cells.dx2;

  par_for_outer("divflux_x2", pmb->exe_space, scr_size, scr_level, ks, ke, is, ie,
    KOKKOS_LAMBDA(TeamMember_t member, const int k, const int i)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nhydro_, ncells2);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro_, ncells2);
      AthenaScratch2D<Real> uflux(member.team_scratch(scr_level), nhydro_, ncells2);

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

      // compute fluxes over [js,je+1]
      switch (rsolver_method)
      {
        case RiemannSolver::advect:
          Advect(member, eos, js, je+1, IVY, wl, wr, uflux);
          break;
        case RiemannSolver::llf:
          LLF(member, eos, js, je+1, IVY, wl, wr, uflux);
          break;
        case RiemannSolver::hllc:
          HLLC(member, eos, js, je+1, IVY, wl, wr, uflux);
          break;
        case RiemannSolver::roe:
          Roe(member, eos, js, je+1, IVY, wl, wr, uflux);
          break;
        default:
          break;
      }

      // Add dF/dx2
      // Fluxes must be summed together to symmetrize round-off error in each dir
      for (int n=0; n<nhydro_; ++n) {
        par_for_inner(member, js, je, [&](const int j)
        {
          divf_(n,k,j,i) += (uflux(n,j+1) - uflux(n,j))/dx2;
        });
      }
    }
  );
  if (!(pmesh_->nx3gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  int ncells3 = pmb->mb_cells.nx3 + 2*(pmb->mb_cells.ng);
  scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells3) * 3;
  scr_level = 1;
  Real &dx3 = pmb->mb_cells.dx3;

  par_for_outer("divflux_x3", pmb->exe_space, scr_size, scr_level, js, je, is, ie,
    KOKKOS_LAMBDA(TeamMember_t member, const int j, const int i)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nhydro_, ncells3);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro_, ncells3);
      AthenaScratch2D<Real> uflux(member.team_scratch(scr_level), nhydro_, ncells3);

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

      // compute fluxes over [ks,ke+1]
      switch (rsolver_method)
      {
        case RiemannSolver::advect:
          Advect(member, eos, ks, ke+1, IVZ, wl, wr, uflux);
          break;
        case RiemannSolver::llf:
          LLF(member, eos, ks, ke+1, IVZ, wl, wr, uflux);
          break;
        case RiemannSolver::hllc:
          HLLC(member, eos, ks, ke+1, IVZ, wl, wr, uflux);
          break;
        case RiemannSolver::roe:
          Roe(member, eos, ks, ke+1, IVZ, wl, wr, uflux);
          break;
        default:
          break;
      }

      // Add dF/dx3
      // Fluxes must be summed together to symmetrize round-off error in each dir
      for (int n=0; n<nhydro_; ++n) {
        par_for_inner(member, ks, ke, [&](const int k)
        { 
          divf_(n,k,j,i) += (uflux(n,k+1) - uflux(n,k))/dx3;
        });
      }
    }
  );
  return TaskStatus::complete;
}

} // namespace hydro
