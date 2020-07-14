//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file calculate_divflux.cpp
//  \brief Calculate divergence of the fluxes for hydro only, no mesh refinement

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "hydro.hpp"
#include "hydro/eos/eos.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateDivFlux
//  \brief Calculate divergence of the fluxes for hydro only, no mesh refinement

void Hydro::CalculateDivFlux(AthenaCenterArray<Real> &divf) {
  int is = pmy_mblock->indx.is; int ie = pmy_mblock->indx.ie;
  int js = pmy_mblock->indx.js; int je = pmy_mblock->indx.je;
  int ks = pmy_mblock->indx.ks; int ke = pmy_mblock->indx.ke;
  int nghost = pmy_mblock->indx.nghost;

  //--------------------------------------------------------------------------------------
  // i-direction

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {

      peos->ConservedToPrimitive(k, j, is-nghost, ie+nghost, u, w_);


//      ReconstructX1(is-1, ie+1, w_, wl_, wr_);

      // reconstruct L/R states
//      if (order == 1) {
//        pmb->precon->DonorCellX1(k, j, is-1, ie+1, w, bcc, wl_, wr_);
//      } else if (order == 2) {
//        pmb->precon->PiecewiseLinearX1(k, j, is-1, ie+1, w, bcc, wl_, wr_);
//      } else {
//        pmb->precon->PiecewiseParabolicX1(k, j, is-1, ie+1, w, bcc, wl_, wr_);
//      }

//      RiemannSolver(is, ie+1, IVX, wl_, wr_, x1flux_);

//      for (int i=is; i<=ie; ++i) {
//        divf(n,k,j,i) = (x1flux_(n,i+1) - x1flux_(i))/dx;
//      }

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

  return;
}

} // namespace hydro
