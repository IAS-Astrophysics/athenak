//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_update.cpp
//  \brief Updates hydro conserved variables, using weighted average and partial time
//  step appropriate for various SSP RK integrators (e.g. RK1, RK2, RK3)

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroUpdate
//  \brief Update conserved variables 

TaskStatus Hydro::HydroUpdate(Driver *pdrive, int stage)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;

  Real &gam0 = pdrive->gam0[stage-1];
  Real &gam1 = pdrive->gam1[stage-1];
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nvars = nhydro + nscalars;
  auto u0_ = u0;
  auto u1_ = u1;
  auto divf_ = divf;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator used
  // Important to use vector inner loop for good performance on cpus
  int nmb = pmy_pack->nmb_thispack;
  int scr_level = 0;
  size_t scr_size = 0;
  par_for_outer("hydro_update", DevExeSpace(), scr_size, scr_level,
    0, (nmb-1), 0, (nvars-1), ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j)
    {
      par_for_inner(member, is, ie, [&](const int i)
      {
        u0_(m,n,k,j,i) =
              gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - beta_dt*divf_(m,n,k,j,i);
      });
    }
  );

  return TaskStatus::complete;
}
} // namespace hydro
