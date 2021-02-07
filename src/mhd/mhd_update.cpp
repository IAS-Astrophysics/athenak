//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_update.cpp
//  \brief Updates MHD conserved variables, using weighted average and partial time
//  step appropriate for various SSP RK integrators (e.g. RK1, RK2, RK3)

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDUpdate
//  \brief Update conserved variables 

TaskStatus MHD::Update(Driver *pdriver, int stage)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);
  bool &nx2gt1 = pmy_pack->pmesh->nx2gt1;
  bool &nx3gt1 = pmy_pack->pmesh->nx3gt1;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nv1 = nmhd + nscalars - 1;
  auto u0_ = u0;
  auto u1_ = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &mbsize = pmy_pack->pmb->mbsize;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator used
  // Important to use vector inner loop for good performance on cpus
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("mhd_update",DevExeSpace(),scr_size,scr_level,0,nmb1,0,nv1,ks,ke,js,je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j)
    {
      ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

      // compute dF1/dx1
      par_for_inner(member, is, ie, [&](const int i)
      {
        divf(i) = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.dx1.d_view(m);
      });
      member.team_barrier();

      // Add dF2/dx2
      // Fluxes must be summed in pairs to symmetrize round-off error in each dir
      if (nx2gt1) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf(i) += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.dx2.d_view(m);
        });
        member.team_barrier();
      }

      // Add dF3/dx3
      // Fluxes must be summed in pairs to symmetrize round-off error in each dir
      if (nx3gt1) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          divf(i) += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.dx3.d_view(m);
        });
        member.team_barrier();
      }

      par_for_inner(member, is, ie, [&](const int i)
      {
        u0_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - beta_dt*divf(i);
      });

    }
  );

  return TaskStatus::complete;
}
} // namespace hydro
