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
  int ncells1 = pmb->mb_cells.nx1 + 2*(pmb->mb_cells.ng);
  Real &dx1 = pmb->mb_cells.dx1;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells1) * 3;
  int scr_level = 1;

  par_for_outer("divflux_x1", pmb->exe_space, scr_size, scr_level, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int k, const int j)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> uflux(member.team_scratch(scr_level), nhydro, ncells1);

      // compute fluxes over [is,ie+1]
      // Reconstruction and Rsolver functions contain team barriers to sync all threads
      precon->ReconstructX1(member, k, j, is-1, ie+1, w0, wl, wr);
      prsolver->RSolver(member, is, ie+1, IVX, wl, wr, uflux);

      // compute dF/dx1
      for (int n=0; n<nhydro; ++n) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf(n,k,j,i) = (uflux(n,i+1) - uflux(n,i))/dx1;
        });
      }
    }
  );
  if (!(pmesh_->nx2gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // j-direction

  scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells1) * 5;
  scr_level = 1;
  Real &dx2 = pmb->mb_cells.dx2;

  par_for_outer("divflux_x2", pmb->exe_space, scr_size, scr_level, ks, ke,
    KOKKOS_LAMBDA(TeamMember_t member, const int k)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> uflux(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> wl_jp1(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> uflux_jm1(member.team_scratch(scr_level), nhydro, ncells1);

      // compute qL(js)
      // Reconstruction and Rsolver functions contain team barriers to sync all threads
      precon->ReconstructX2(member, k, js-1, is, ie, w0, wl_jp1, wr);

      for (int j=js; j<=je+1; ++j) {
        // copy Wl from last iteration of j.
        for (int n=0; n<nhydro; ++n) {
          par_for_inner(member, is, ie, [&](const int i)
          {
            wl(n,i) = wl_jp1(n,i);
          });
        }

        // compute fluxes over [js,je+1]
        precon->ReconstructX2(member, k, j, is, ie, w0, wl_jp1, wr);
        prsolver->RSolver(member, is, ie, IVY, wl, wr, uflux);

        // Add dF/dx2
        // Note fluxes are summed together first to symmetrize round-off error in each dir
        for (int n=0; n<nhydro; ++n) {
          if (j>js) {
            par_for_inner(member, is, ie, [&](const int i)
            {
              divf(n,k,j-1,i) += (uflux(n,i) - uflux_jm1(n,i))/dx2;
            });
          }

          // copy flux for use in next iteration
          if (j<(je+1)) {
            par_for_inner(member, is, ie, [&](const int i)
            {
              uflux_jm1(n,i) = uflux(n,i);
            });
          }
        }
      }
    }
  );
  if (!(pmesh_->nx3gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  scr_size = AthenaScratch2D<Real>::shmem_size(nhydro, ncells1) * 5;
  scr_level = 1;
  Real &dx3 = pmb->mb_cells.dx3;

  par_for_outer("divflux_x3", pmb->exe_space, scr_size, scr_level, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int j)
    {
      AthenaScratch2D<Real> wl(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> wr(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> uflux(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> wl_kp1(member.team_scratch(scr_level), nhydro, ncells1);
      AthenaScratch2D<Real> uflux_km1(member.team_scratch(scr_level), nhydro, ncells1);

      // compute qL(ks)
      // Reconstruction and Rsolver functions contain team barriers to sync all threads
      precon->ReconstructX3(member, ks-1, j, is, ie, w0, wl_kp1, wr);

      for (int k=ks; k<=ke+1; ++k) {
        // copy Wl from last iteration of k.
        for (int n=0; n<nhydro; ++n) {
          par_for_inner(member, is, ie, [&](const int i)
          {
            wl(n,i) = wl_kp1(n,i);
          });
        }

        // compute fluxes over [ks,ke+1]
        precon->ReconstructX3(member, k, j, is, ie, w0, wl_kp1, wr);
        prsolver->RSolver(member, is, ie, IVZ, wl, wr, uflux);

        // Add dF/dx3
        // Note fluxes are summed together first to symmetrize round-off error in each dir
        for (int n=0; n<nhydro; ++n) {
          if (k>ks) {
            par_for_inner(member, is, ie, [&](const int i)
            { 
              divf(n,k-1,j,i) += (uflux(n,i) - uflux_km1(n,i))/dx3;
            });
          }
          if (k<(ke+1)) {
            par_for_inner(member, is, ie, [&](const int i)
            { 
              uflux_km1(n,i) = uflux(n,i);
            });
          }
        }
      }
    }
  );
  return TaskStatus::complete;
}

} // namespace hydro
