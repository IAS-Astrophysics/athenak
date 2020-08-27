//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file calculate_divflux.cpp
//  \brief Calculate divergence of the fluxes for hydro only, no mesh refinement

#include <iostream>

#include "athena.hpp"
#include "athena_arrays.hpp"
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

  //--------------------------------------------------------------------------------------
  // i-direction

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {

      precon->ReconstructX1(k, j, is, ie+1, w0, wl_, wr_);
      prsolver->RSolver(is, ie+1, IVX, wl_, wr_, uflux_);

      for (int n=0; n<nhydro; ++n) {
        for (int i=is; i<=ie; ++i) {
          divf(n,k,j,i) = (uflux_(n,i+1) - uflux_(n,i))/pmb->mb_cells.dx1;
        }
      }

    }
  }
  if (!(pmesh_->nx2gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // j-direction

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {

      precon->ReconstructX2(k, j, is, ie, w0, wl_, wr_);
      prsolver->RSolver(is, ie, IVY, wl_, wr_, uflux_);

      for (int n=0; n<nhydro; ++n) {
        if (j>js) {
          for (int i=is; i<=ie; ++i) {
            divf(n,k,j-1,i) += uflux_(n,i)/pmb->mb_cells.dx2;
          }
        }
        if (j<(je+1)) {
          for (int i=is; i<=ie; ++i) {
            divf(n,k,j,i) -= uflux_(n,i)/pmb->mb_cells.dx2;
          }
        }
      }

    }
  }
  if (!(pmesh_->nx3gt1)) return TaskStatus::complete;

  //--------------------------------------------------------------------------------------
  // k-direction

  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {

      precon->ReconstructX3(k, j, is, ie, w0, wl_, wr_);
      prsolver->RSolver(is, ie, IVZ, wl_, wr_, uflux_);

      for (int n=0; n<nhydro; ++n) {
        if (k>ks) {
          for (int i=is; i<=ie; ++i) {
            divf(n,k-1,j,i) += uflux_(n,i)/pmb->mb_cells.dx3;
          }
        }
        if (k<(ke+1)) {
          for (int i=is; i<=ie; ++i) {
            divf(n,k,j,i) -= uflux_(n,i)/pmb->mb_cells.dx3;
          }
        }
      }

    }
  }
  return TaskStatus::complete;
}

} // namespace hydro
