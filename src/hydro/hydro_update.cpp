//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_update.cpp
//  \brief Updates hydro conserved variables, using weighted average and partial time
//  step appropriate for various SSP RK integrators (e.g. RK1, RK2, RK3)

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroUpdate
//  \brief Calculate divergence of the fluxes for hydro only, no mesh refinement

//void Hydro::HydroUpdate(AthenaArray<Real> &u0, AthenaArray<Real> &u1, 
//                         AthenaArray<Real> &divf) {
void Hydro::HydroUpdate() {
  int is = pmy_mblock->indx.is; int ie = pmy_mblock->indx.ie;
  int js = pmy_mblock->indx.js; int je = pmy_mblock->indx.je;
  int ks = pmy_mblock->indx.ks; int ke = pmy_mblock->indx.ke;
  int nghost = pmy_mblock->indx.nghost;

  // update all variables to intermediate step using weights and fractional time step 
  // appropriate to stage of particular integrator used (see XX)

  Real gam[2];
  gam[0] = 0.0;
  gam[1] = 1.0;
  Real beta = 0.5;

  for (int n=0; n<nhydro; ++n) {

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          u0(n,k,j,i) = gam[0]*u0(n,k,j,i) + gam[1]*u1(n,k,j,i) 
                        - beta*(pmy_mblock->pmy_mesh->dt)*divf(n,k,j,i);
        }
      }
    }

  }

  return;
}

} // namespace hydro
