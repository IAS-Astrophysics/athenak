#ifndef DYNGR_FOFC_CPP_
#define DYNGR_FOFC_CPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fofc.cpp
//! \brief Implements functions for first-order flux correction (FOFC) algorithm.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "dyngr/rsolvers/llf_dyngrhyd.cpp"
#include "dyngr/dyngr.hpp"
#include "dyngr/dyngr_util.hpp"
#include "hydro/hydro.hpp"

namespace dyngr {
//----------------------------------------------------------------------------------------
//! \fn void Hydro::FOFC
//! \brief Implements first-order flux-correction (FOFC) algorithm for Hydro.  First an
//! estimate of the updated conserved variables is made. This estimate is then used to
//! flag any cell where floors will be required during the conversion to primitives. Then
//! the fluxes on the faces of flagged cells are replaced with first-order LLF fluxes.
//! Often this is enough to prevent floors from being needed.

template<class EOSPolicy, class ErrorPolicy>
void DynGRPS<EOSPolicy, ErrorPolicy>::FOFC(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb = pmy_pack->nmb_thispack;
  auto flx1 = pmy_pack->phydro->uflx.x1f;
  auto flx2 = pmy_pack->phydro->uflx.x2f;
  auto flx3 = pmy_pack->phydro->uflx.x3f;
  auto &size = pmy_pack->pmb->mb_size;

  int &nhyd_ = pmy_pack->phydro->nhydro;
  int &nscal_ = pmy_pack->phydro->nscalars;
  auto &u0_ = pmy_pack->phydro->u0;
  auto &u1_ = pmy_pack->phydro->u1;
  auto &utest_ = pmy_pack->phydro->utest;

  // Estimate updated conserved variables and cell-centered fields
  par_for("FOFC-newu", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dtodx1 = beta_dt/size.d_view(m).dx1;
    Real dtodx2 = beta_dt/size.d_view(m).dx2;
    Real dtodx3 = beta_dt/size.d_view(m).dx3;

    // Estimate conserved variables
    for (int n=0; n < nhyd_; ++n) {
      Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
      if (multi_d) {
        divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
      }
      if (three_d) {
        divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
      }
      utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
    }
  });

  // Test whether conversion to primitives requires floors
  // Note b0 and w0 passed to function, but not used/changed.
  eos.ConsToPrim(utest_, pmy_pack->phydro->w0, is, ie, js, je, ks, ke, true);

  auto &eos_ = eos;
  auto fofc_ = pmy_pack->phydro->fofc;
  auto &w0_ = pmy_pack->phydro->w0;
  auto &adm = pmy_pack->padm->adm;

  const Real mb = eos.ps.GetEOS().GetBaryonMass();

  // Replace fluxes with first-order LLF fluxes for any cell where floors are needed
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Replace x1-flux at i
    if (fofc_(m,k,j,i)) {
      // Load left state
      Real wim1[NPRIM];
      ExtractPrimitives(w0_, wim1, eos_, nhyd_, nscal_, m, k, j, i-1);

      // Load right state
      Real wi[NPRIM];
      ExtractPrimitives(w0_, wi, eos_, nhyd_, nscal_, m, k, j, i);

      // Compute the metric terms at the face.
      Real g3d[NSPMETRIC], beta_u[3], alpha;
      Face1Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

      // Compute new 1st-order LLF flux
      Real flux[NCONS];
      SingleStateLLF_DYNGR(eos_, wim1, wi, IVX, g3d, beta_u, alpha, flux);
      
      // Store 1st-order fluxes
      InsertFluxes(flux, flx1, m, k, j, i);

      // Replace x1-flux at i+1
      // Load right state
      Real wip1[NPRIM];
      ExtractPrimitives(w0_, wip1, eos_, nhyd_, nscal_, m, k, j, i+1);

      // Compute the metric terms at the face.
      Face1Metric(m, k, j, i+1, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

      // Compute new 1st-order LLF flux
      SingleStateLLF_DYNGR(eos_, wi, wip1, IVX, g3d, beta_u, alpha, flux);

      // Store 1st-order fluxes
      InsertFluxes(flux, flx1, m, k, j, i+1);

      if (multi_d) {
        // Replace x2-flux at j
        // Load left state
        Real wjm1[NPRIM];
        ExtractPrimitives(w0_, wjm1, eos_, nhyd_, nscal_, m, k, j-1, i);

        // Load right state; because we don't need to permute the states
        // like the static hydro solver, we can just replace this with wi.
        Real *wj = wi;

        // Compute the metric terms at the face.
        Face2Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // Compute new 1st-order LLF flux
        SingleStateLLF_DYNGR(eos_, wjm1, wj, IVY, g3d, beta_u, alpha, flux);

        // Store 1st-order fluxes
        InsertFluxes(flux, flx2, m, k, j, i);

        // Replace x2-flux at j+1
        // Load right state
        Real wjp1[NPRIM];
        ExtractPrimitives(w0_, wjp1, eos_, nhyd_, nscal_, m, k, j+1, i);

        // Compute the metric terms at the face.
        Face2Metric(m, k, j+1, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // Compute new 1st-order LLF flux
        SingleStateLLF_DYNGR(eos_, wj, wjp1, IVY, g3d, beta_u, alpha, flux);

        // Store 1st-order fluxes
        InsertFluxes(flux, flx2, m, k, j+1, i);
      }

      if (three_d) {
        // Replace x3-flux at k
        // Load left state
        Real wkm1[NPRIM];
        ExtractPrimitives(w0_, wkm1, eos_, nhyd_, nscal_, m, k-1, j, i);

        // Load right state; because we don't need to permute the states
        // like the static hydro solver, we can just replace this with wi.
        Real *wk = wi;

        // Compute the metric terms at the face.
        Face3Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // Compute new 1st-order LLF flux
        SingleStateLLF_DYNGR(eos_, wkm1, wk, IVZ, g3d, beta_u, alpha, flux);

        // Store 1st-order fluxes
        InsertFluxes(flux, flx3, m, k, j, i);

        // Replace x3-flux at k+1
        // Load right state
        Real wkp1[NPRIM];
        ExtractPrimitives(w0_, wkp1, eos_, nhyd_, nscal_, m, k+1, j, i);

        // Compute the metric termas at the face.
        Face3Metric(m, k+1, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // Compute new 1st-order LLF flux
        SingleStateLLF_DYNGR(eos_, wk, wkp1, IVZ, g3d, beta_u, alpha, flux);

        // Store 1st-order fluxes
        InsertFluxes(flux, flx3, m, k+1, j, i);
      }

      // reset FOFC flag
      fofc_(m,k,j,i) = false;
    }
  });

  return;
}

} // namespace dyngr

#endif
