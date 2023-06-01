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
      //Real wrim1[NPRIM], wri[NPRIM], wrip1[NPRIM];
      //Real wli[NPRIM], wlip1[NPRIM], wlip2[NPRIM];
      Real wli[NPRIM], wri[NPRIM], wrip1[NPRIM], *wlip1;
      // Reconstruct states; note we make this call three times because each call gives
      // the right side at i-1/2 and the left side at i+1/2.
      //ExtractPrimitivesWithMinmod<IVX>(w0_, wli, wrim1, eos_, nhyd_, nscal_, m, k, j, i-1);
      //ExtractPrimitivesWithMinmod<IVX>(w0_, wlip1, wri, eos_, nhyd_, nscal_, m, k, j, i);
      //ExtractPrimitivesWithMinmod<IVX>(w0_, wlip2, wrip1, eos_, nhyd_, nscal_, m, k, j, i+1);
      ExtractPrimitives(w0_, wli, eos_, nhyd_, nscal_, m, k, j, i-1);
      ExtractPrimitives(w0_, wri, eos_, nhyd_, nscal_, m, k, j, i);
      ExtractPrimitives(w0_, wrip1, eos_, nhyd_, nscal_, m, k, j, i+1);
      wlip1 = wri;

      // Compute the metric terms at i-1/2.
      Real g3d[NSPMETRIC], beta_u[3], alpha;
      Face1Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

      // Compute new 1st-order LLF flux
      Real flux[NCONS];
      SingleStateLLF_DYNGR(eos_, wli, wri, IVX, g3d, beta_u, alpha, flux);

      // Store 1st-order fluxes at i-1/2.
      InsertFluxes(flux, flx1, m, k, j, i);

      // Compute the metric terms at i+1/2.
      Face1Metric(m, k, j, i+1, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

      // Compute new 1st-order LLF flux
      SingleStateLLF_DYNGR(eos_, wlip1, wrip1, IVX, g3d, beta_u, alpha, flux);

      // Store 1st-order fluxes at i+1/2.
      InsertFluxes(flux, flx1, m, k, j, i+1);

      if (multi_d) {
        //Real wrjm1[NPRIM], wrj[NPRIM], wrjp1[NPRIM];
        //Real wlj[NPRIM], wljp1[NPRIM], wljp2[NPRIM];
        Real wlj[NPRIM], wrjp1[NPRIM], *wrj, *wljp1;
        // Reconstruct states
        //ExtractPrimitivesWithMinmod<IVY>(w0_, wlj, wrjm1, eos_, nhyd_, nscal_, m, k, j-1, i);
        //ExtractPrimitivesWithMinmod<IVY>(w0_, wljp1, wrj, eos_, nhyd_, nscal_, m, k, j, i);
        //ExtractPrimitivesWithMinmod<IVY>(w0_, wljp2, wrjp1, eos_, nhyd_, nscal_, m, k, j+1, i);
        ExtractPrimitives(w0_, wlj, eos_, nhyd_, nscal_, m, k, j-1, i);
        ExtractPrimitives(w0_, wrjp1, eos_, nhyd_, nscal_, m, k, j+1, i);
        wrj = wri;
        wljp1 = wrj;

        // Compute the metric terms at j-1/2.
        Face2Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // Compute new 1st-order LLF flux.
        SingleStateLLF_DYNGR(eos_, wlj, wrj, IVY, g3d, beta_u, alpha, flux);

        // Store 1st-order fluxes at j-1/2.
        InsertFluxes(flux, flx2, m, k, j, i);

        // Compute the metric terms at j+1/2.
        Face2Metric(m, k, j+1, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // Compute new 1st-order LLF flux.
        SingleStateLLF_DYNGR(eos_, wljp1, wrjp1, IVY, g3d, beta_u, alpha, flux);

        // Store 1st-order fluxes at j+1/2.
        InsertFluxes(flux, flx2, m, k, j+1, i);
      }

      if (three_d) {
        //Real wrkm1[NPRIM], wrk[NPRIM], wrkp1[NPRIM];
        //Real wlk[NPRIM], wlkp1[NPRIM], wlkp2[NPRIM];
        Real wlk[NPRIM], wrkp1[NPRIM], *wrk, *wlkp1;
        // Reconstruct states
        //ExtractPrimitivesWithMinmod<IVZ>(w0_, wlk, wrkm1, eos_, nhyd_, nscal_, m, k-1, j, i);
        //ExtractPrimitivesWithMinmod<IVZ>(w0_, wlkp1, wrk, eos_, nhyd_, nscal_, m, k, j, i);
        //ExtractPrimitivesWithMinmod<IVZ>(w0_, wlkp2, wrkp1, eos_, nhyd_, nscal_, m, k+1, j, i);
        ExtractPrimitives(w0_, wlk, eos_, nhyd_, nscal_, m, k-1, j, i);
        ExtractPrimitives(w0_, wrkp1, eos_, nhyd_, nscal_, m, k+1, j, i);
        wrk = wri;
        wlkp1 = wrk;

        // Compute the metric terms at k-1/2.
        Face3Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // Compute new 1st-order LLF flux.
        SingleStateLLF_DYNGR(eos_, wlk, wrk, IVZ, g3d, beta_u, alpha, flux);

        // Store 1st-order fluxes at k-1/2.
        InsertFluxes(flux, flx3, m, k, j, i);

        // Compute the metric terms at k+1/2.
        Face3Metric(m, k+1, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // Compute new 1st-order LLF flux.
        SingleStateLLF_DYNGR(eos_, wlkp1, wrkp1, IVZ, g3d, beta_u, alpha, flux);

        // Store 1st-order fluxes at k+1/2.
        InsertFluxes(flux, flx2, m, k+1, j, i);
      }

      // reset FOFC flag
      fofc_(m,k,j,i) = false;
    }
  });

  return;
}

} // namespace dyngr

#endif
