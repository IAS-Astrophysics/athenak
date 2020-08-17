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

TaskStatus Hydro::HydroDivFlux(Driver *pdrive, int stage) {
  int is = pmy_mblock->mb_cells.is; int ie = pmy_mblock->mb_cells.ie;
  int js = pmy_mblock->mb_cells.js; int je = pmy_mblock->mb_cells.je;
  int ks = pmy_mblock->mb_cells.ks; int ke = pmy_mblock->mb_cells.ke;
  int nghost = pmy_mblock->mb_cells.nghost;

  //--------------------------------------------------------------------------------------
  // i-direction

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {

      peos->ConservedToPrimitive(k, j, is-nghost, ie+nghost, u0, w_);
      precon->ReconstructX1(is-1, ie+1, w_, wl_, wr_);
      prsolver->RSolver(is, ie+1, IVX, wl_, wr_, uflux_);

      for (int n=0; n<nhydro; ++n) {
        for (int i=is; i<=ie; ++i) {
          divf(n,k,j,i) = (uflux_(n,i+1) - uflux_(n,i))/pmy_mblock->mb_cells.dx1;
        }
      }

    }
  }

/****
  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmb->pmy_mesh->f2) {
    // set the loop limits
    il = is-1, iu = ie+1, kl = ks, ku = ke;
    // TODO(felker): fix loop limits for fourth-order hydro
    //    if (MAGNETIC_FIELDS_ENABLED) {
    if (pmb->block_size.nx3 == 1) // 2D
      kl = ks, ku = ke;
    else // 3D
      kl = ks-1, ku = ke+1;
    //    }

    for (int k=kl; k<=ku; ++k) {
      // reconstruct the first row
      if (order == 1) {
        pmb->precon->DonorCellX2(k, js-1, il, iu, w, bcc, wl_, wr_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX2(k, js-1, il, iu, w, bcc, wl_, wr_);
      } else {
        pmb->precon->PiecewiseParabolicX2(k, js-1, il, iu, w, bcc, wl_, wr_);
      }
      for (int j=js; j<=je+1; ++j) {
        // reconstruct L/R states at j
        if (order == 1) {
          pmb->precon->DonorCellX2(k, j, il, iu, w, bcc, wlb_, wr_);
        } else if (order == 2) {
          pmb->precon->PiecewiseLinearX2(k, j, il, iu, w, bcc, wlb_, wr_);
        } else {
          pmb->precon->PiecewiseParabolicX2(k, j, il, iu, w, bcc, wlb_, wr_);
        }

        pmb->pcoord->CenterWidth2(k, j, il, iu, dxw_);
        RiemannSolver(k, j, il, iu, IVY, wl_, wr_, x2flux, dxw_);
      }
    }
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->f3) {
    // set the loop limits
    // TODO(felker): fix loop limits for fourth-order hydro
    //    if (MAGNETIC_FIELDS_ENABLED)
    il = is-1, iu = ie+1, jl = js-1, ju = je+1;

    for (int j=jl; j<=ju; ++j) { // this loop ordering is intentional
      // reconstruct the first row
      if (order == 1) {
        pmb->precon->DonorCellX3(ks-1, j, il, iu, w, bcc, wl_, wr_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX3(ks-1, j, il, iu, w, bcc, wl_, wr_);
      } else {
        pmb->precon->PiecewiseParabolicX3(ks-1, j, il, iu, w, bcc, wl_, wr_);
      }
      for (int k=ks; k<=ke+1; ++k) {
        // reconstruct L/R states at k
        if (order == 1) {
          pmb->precon->DonorCellX3(k, j, il, iu, w, bcc, wlb_, wr_);
        } else if (order == 2) {
          pmb->precon->PiecewiseLinearX3(k, j, il, iu, w, bcc, wlb_, wr_);
        } else {
          pmb->precon->PiecewiseParabolicX3(k, j, il, iu, w, bcc, wlb_, wr_);
        }

        pmb->pcoord->CenterWidth3(k, j, il, iu, dxw_);
        RiemannSolver(k, j, il, iu, IVZ, wl_, wr_, x3flux, dxw_);

      }
    }
  }
****/

  return TaskStatus::complete;
}

} // namespace hydro
