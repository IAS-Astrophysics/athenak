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
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;

  Real &gam0 = pdrive->gam0[stage-1];
  Real &gam1 = pdrive->gam1[stage-1];
  Real &beta = pdrive->beta[stage-1];

  // 4D parallel loop that updates conserved variables to intermediate step using weights
  // and fractional time step appropriate to stages of time-integrator used (see XX)

  par_for_outer("hydro_update", pmb->exe_space, 0, 1, 0, (nhydro-1), ks, ke,
    KOKKOS_LAMBDA(TeamMember_t member, const int n, const int k)
    {
      for (int j=js; j<=je; ++j) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          u0(n,k,j,i) = gam0*u0(n,k,j,i) + gam1*u1(n,k,j,i) -
                        beta*(pmesh_->dt)*divf(n,k,j,i);
        });
      }
    }
  );

  return TaskStatus::complete;
}

} // namespace hydro
