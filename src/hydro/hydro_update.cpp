//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_update.cpp
//  \brief Updates hydro conserved variables, using weighted average and partial time
//  step appropriate for various SSP RK integrators (e.g. RK1, RK2, RK3)

#include <iostream>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroUpdate
//  \brief Calculate divergence of the fluxes for hydro only, no mesh refinement

//void Hydro::HydroUpdate(AthenaArray<Real> &u0, AthenaArray<Real> &u1, 
//                         AthenaArray<Real> &divf) {
TaskStatus Hydro::HydroUpdate(Driver *pdrive, int stage)
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;

  // update all variables to intermediate step using weights and fractional time step 
  // appropriate to stage of particular integrator used (see XX)

  Real &gam0 = pdrive->gam0[stage-1];
  Real &gam1 = pdrive->gam1[stage-1];
  Real &beta = pdrive->beta[stage-1];

  for (int n=0; n<nhydro; ++n) {

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          u0(n,k,j,i) = gam0*u0(n,k,j,i) + gam1*u1(n,k,j,i) 
                        - beta*(pmesh_->dt)*divf(n,k,j,i);
        }
      }
    }

  }

  return TaskStatus::complete;
}

} // namespace hydro
