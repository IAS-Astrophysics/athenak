//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fofc.cpp
//! \brief Implements functions for first-order flux correction (FOFC) algorithm.

#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_grmhd/rsolvers/llf_dyn_grmhd.hpp"
#include "dyn_grmhd/rsolvers/hlle_dyn_grmhd.hpp"
#include "dyn_grmhd/dyn_grmhd_util.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"

namespace dyngr {
//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::FOFC
//! \brief Implements first-order flux-correction (FOFC) algorithm for MHD.  First an
//! estimate of the updated conserved variables is made. This estimate is then used to
//! flag any cell where floors will be required during the conversion to primitives. Then
//! the fluxes on the faces of flagged cells are replaced with first-order LLF fluxes.
//! Often this is enough to prevent floors from being needed.  The FOFC infrastructure is
//! also exploited for BH excision.  If a cell is about the horizon, FOFC is automatically
//! triggered (without estimating updated conserved variables).

template<class EOSPolicy, class ErrorPolicy> template <DynGRMHD_RSolver rsolver_method_>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::FOFC(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb = pmy_pack->nmb_thispack;
  auto flx1 = pmy_pack->pmhd->uflx.x1f;
  auto flx2 = pmy_pack->pmhd->uflx.x2f;
  auto flx3 = pmy_pack->pmhd->uflx.x3f;
  auto &size = pmy_pack->pmb->mb_size;

  auto &bcc0_ = pmy_pack->pmhd->bcc0;
  auto &e3x1_ = pmy_pack->pmhd->e3x1;
  auto &e2x1_ = pmy_pack->pmhd->e2x1;
  auto &e1x2_ = pmy_pack->pmhd->e1x2;
  auto &e3x2_ = pmy_pack->pmhd->e3x2;
  auto &e2x3_ = pmy_pack->pmhd->e2x3;
  auto &e1x3_ = pmy_pack->pmhd->e1x3;

  int &nmhd_ = pmy_pack->pmhd->nmhd;
  int &nscal_ = pmy_pack->pmhd->nscalars;

  // Extract EOS species fraction limits onto device before the GPU kernel.
  DvceArray1D<Real> eos_min_Y("eos_min_Y", nscal_);
  DvceArray1D<Real> eos_max_Y("eos_max_Y", nscal_);
  if (nscal_ > 0) {
    auto h_min_Y = Kokkos::create_mirror_view(eos_min_Y);
    auto h_max_Y = Kokkos::create_mirror_view(eos_max_Y);
    for (int n = 0; n < nscal_; ++n) {
      h_min_Y(n) = eos.ps.GetEOS().GetMinimumSpeciesFraction(n);
      h_max_Y(n) = eos.ps.GetEOS().GetMaximumSpeciesFraction(n);
    }
    Kokkos::deep_copy(eos_min_Y, h_min_Y);
    Kokkos::deep_copy(eos_max_Y, h_max_Y);
  }

  if (pmy_pack->pmhd->use_fofc) {
    Real &gam0 = pdriver->gam0[stage-1];
    Real &gam1 = pdriver->gam1[stage-1];
    Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

    auto &u0_ = pmy_pack->pmhd->u0;
    auto &u1_ = pmy_pack->pmhd->u1;
    auto &utest_ = pmy_pack->pmhd->utest;
    auto &bcctest_ = pmy_pack->pmhd->bcctest;
    auto &b1_ = pmy_pack->pmhd->b1;
    auto fofc_ = pmy_pack->pmhd->fofc;
    auto fofc_scal_ = pmy_pack->pmhd->fofc_scal;

    bool &max_ = enforce_maximum;
    Real Rmax = std::numeric_limits<Real>::max();
    Real &dmp_M_ = dmp_M;

    // Index bounds
    int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
    int jadd = 0, kadd = 0;
    if (multi_d) { jl = js-1, ju = je+1, jadd = 1; }
    if (three_d) { kl = ks-1, ku = ke+1, kadd = 1; }

    // Estimate updated conserved variables and cell-centered fields
    par_for("FOFC-newu", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size.d_view(m).dx1;
      Real dtodx2 = beta_dt/size.d_view(m).dx2;
      Real dtodx3 = beta_dt/size.d_view(m).dx3;

      // Estimate conserved variables
      for (int n=0; n<nmhd_ + nscal_; ++n) {
        Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
        if (multi_d) {
          divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
        }
        if (three_d) {
          divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
        }
        utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
      }

      // Enforce maximum principle
      const int nvars = 2;
      int indcs[nvars] = {IDN, IEN};
      if (max_) {
        for (int n = 0; n < nvars; ++n) {
          Real varmax = -Rmax;
          Real varmin = Rmax;
          for (int kt = k - kadd; kt <= k + kadd; kt++) {
            for (int jt = j - jadd; jt <= j + jadd; jt++) {
              for (int it = i-1; it <= i+1; it++) {
                varmax = fmax(varmax, u1_(m,indcs[n],kt,jt,it));
                varmin = fmin(varmin, u1_(m,indcs[n],kt,jt,it));
              }
            }
          }
          if (utest_(m,indcs[n],k,j,i) > dmp_M_*varmax ||
              utest_(m,indcs[n],k,j,i) < varmin/dmp_M_) {
            fofc_(m,k,j,i) = true;
          }
        }
        // Enforce maximum principle for scalar
        if (nscal_ > 0) {
          for (int n = 0; n < nscal_; ++n) {
            Real varmax = eos_min_Y(n);
            Real varmin = eos_max_Y(n);
            for (int kt = k - kadd; kt <= k + kadd; kt++) {
              for (int jt = j - jadd; jt <= j + jadd; jt++) {
                for (int it = i-1; it <= i+1; it++) {
                  varmax = fmax(varmax, u1_(m,nmhd_+n,kt,jt,it));
                  varmin = fmin(varmin, u1_(m,nmhd_+n,kt,jt,it));
                }
              }
            }
            if (utest_(m,nmhd_+n,k,j,i) > dmp_M_*varmax ||
                utest_(m,nmhd_+n,k,j,i) < varmin/dmp_M_) {
              fofc_scal_(m,n,k,j,i) = true;
            }
          }
        }
      }

      if ( nscal_ > 0 ) {
        for (int n=0; n<nscal_; ++n) {
          if ( utest_(m,IDN,k,j,i) > 0 ) {
            Real min_Y_ = eos_min_Y(n);
            Real max_Y_ = eos_max_Y(n);
            if ( utest_(m,IDN,k,j,i) * min_Y_ > utest_(m,nmhd_+n,k,j,i) ||
                 utest_(m,IDN,k,j,i) * max_Y_ < utest_(m,nmhd_+n,k,j,i) ) {
              fofc_scal_(m,n,k,j,i) = true;
            }
          } else {
            fofc_scal_(m,n,k,j,i) = true;
          }
        }
      }

      // Estimate updated cell-centered fields
      Real b1old = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      Real b2old = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      Real b3old = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));

      bcctest_(m,IBX,k,j,i) = gam0*bcc0_(m,IBX,k,j,i) + gam1*b1old;
      bcctest_(m,IBY,k,j,i) = gam0*bcc0_(m,IBY,k,j,i) + gam1*b2old;
      bcctest_(m,IBZ,k,j,i) = gam0*bcc0_(m,IBZ,k,j,i) + gam1*b3old;

      bcctest_(m,IBY,k,j,i) += dtodx1*(e3x1_(m,k,j,i+1) - e3x1_(m,k,j,i));
      bcctest_(m,IBZ,k,j,i) -= dtodx1*(e2x1_(m,k,j,i+1) - e2x1_(m,k,j,i));
      if (multi_d) {
        bcctest_(m,IBX,k,j,i) -= dtodx2*(e3x2_(m,k,j+1,i) - e3x2_(m,k,j,i));
        bcctest_(m,IBZ,k,j,i) += dtodx2*(e1x2_(m,k,j+1,i) - e1x2_(m,k,j,i));
      }
      if (three_d) {
        bcctest_(m,IBX,k,j,i) += dtodx3*(e2x3_(m,k+1,j,i) - e2x3_(m,k,j,i));
        bcctest_(m,IBY,k,j,i) -= dtodx3*(e1x3_(m,k+1,j,i) - e1x3_(m,k,j,i));
      }
    });

    // Test whether conversion to primitives requires floors
    // Note b0 and w0 passed to function, but not used/changed.
    eos.ConsToPrim(utest_, pmy_pack->pmhd->b0, bcctest_,
                           pmy_pack->pmhd->w0, temperature,
                           il, iu, jl, ju, kl, ku, true);
  }

  auto &use_fofc_ = pmy_pack->pmhd->use_fofc;
  auto fofc_ = pmy_pack->pmhd->fofc;
  auto fofc_scal_ = pmy_pack->pmhd->fofc_scal;
  auto &eos_ = eos;
  auto &use_excise_ = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &w0_ = pmy_pack->pmhd->w0;
  auto &b0_ = pmy_pack->pmhd->b0;
  auto &adm = pmy_pack->padm->adm;

  // Index bounds
  int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-1, ju = je+1; }
  if (three_d) { kl = ks-1, ku = ke+1; }

  // Replace fluxes with first-order LLF fluxes at i,j,k faces for any cell where FOFC
  // and/or excision is used (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_) { fofc_flag = fofc_(m,k,j,i); }

    bool fofc_scalar_any = fofc_flag;
    if (use_fofc_ && nscal_ > 0 && !fofc_flag) {
      // Check if any scalar flag
      for (int n=0; n < nscal_; ++n) {
        fofc_scalar_any |= fofc_scal_(m,n,k,j,i);
      }
    }

    // Check for GR + excision
    bool fofc_excision = false;
    if (use_excise_) { fofc_excision = excision_flux_(m,k,j,i); }

    // Apply FOFC
    if (fofc_flag || fofc_excision || fofc_scalar_any) {
      // Reconstruct states
      Real wli[NPRIM], wri[NPRIM];
      Real bli[NMAG], bri[NMAG];
      ExtractPrimitives(wli, w0_,eos_, nmhd_, nscal_, m, k, j, i-1);
      ExtractPrimitives(wri, w0_, eos_, nmhd_, nscal_, m, k, j, i);
      Real g3d[NSPMETRIC], beta_u[3], alpha;
      Real flux[NCONS], bflux[NMAG];
      if (fofc_flag || fofc_excision) {
        ExtractBField(bli, bcc0_, IBX, IBY, IBZ, m, k, j, i-1);
        ExtractBField(bri, bcc0_, IBX, IBY, IBZ, m, k, j, i);
        bli[IBX] = bri[IBX] = b0_.x1f(m, k, j, i);

        // Compute the metric terms at i-1/2
        adm::Face1Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);

        // compute new 1st-order LLF flux at i-face
        if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
          SingleStateLLF_DYNGR<IVX>(eos_, wli, wri, bli, bri, nmhd_, nscal_,
                                    g3d, beta_u, alpha, flux, bflux);
        } else if (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
          SingleStateHLLE_DYNGR<IVX>(eos_, wli, wri, bli, bri, nmhd_, nscal_,
                                    g3d, beta_u, alpha, flux, bflux);
        }

        // Store 1st-order fluxes at i-1/2
        InsertFluxes(flux, flx1, m, k, j, i);
        e3x1_(m, k, j, i) = bflux[IBY];
        e2x1_(m, k, j, i) = bflux[IBZ];
      }

      // Calculate fluxes of scalars
      for (int n = 0; n < nscal_; n++) {
        if (fofc_flag || fofc_scal_(m,n,k,j,i)) {
          if (flx1(m, IDN, k, j, i) >= 0.0) {
            flx1(m, nmhd_ + n, k, j, i) = flx1(m,IDN,k,j,i)*wli[PYF + n];
          } else {
            flx1(m, nmhd_ + n, k, j, i) = flx1(m,IDN,k,j,i)*wri[PYF + n];
          }
        }
      }

      if (multi_d) {
        Real wlj[NPRIM], *wrj;
        Real blj[NMAG], brj[NMAG];
        // Reconstruct states
        ExtractPrimitives(wlj, w0_, eos_, nmhd_, nscal_, m, k, j-1, i);
        wrj = &wri[0];
        if (fofc_flag || fofc_excision) {
          ExtractBField(blj, bcc0_, IBY, IBZ, IBX, m, k, j-1, i);
          ExtractBField(brj, bcc0_, IBY, IBZ, IBX, m, k, j, i);
          blj[IBY] = brj[IBY] = b0_.x2f(m, k, j, i);

          // Compute the metric terms at j-1/2
          adm::Face2Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha,
                            g3d, beta_u, alpha);

          // Compute new 1st-order LLF flux at j-face
          if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
            SingleStateLLF_DYNGR<IVY>(eos_, wlj, wrj, blj, brj, nmhd_, nscal_,
                                      g3d, beta_u, alpha, flux, bflux);
          } else if (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
            SingleStateHLLE_DYNGR<IVY>(eos_, wlj, wrj, blj, brj, nmhd_, nscal_,
                                      g3d, beta_u, alpha, flux, bflux);
          }

          // Store 1st-order fluxes at j-1/2
          InsertFluxes(flux, flx2, m, k, j, i);
          e1x2_(m,k,j,i) = bflux[IBY];
          e3x2_(m,k,j,i) = bflux[IBZ];
        }

        // Calculate fluxes of scalars
        for (int n = 0; n < nscal_; n++) {
          if (fofc_flag || fofc_scal_(m,n,k,j,i)) {
            if (flx2(m, IDN, k, j, i) >= 0.0) {
              flx2(m, nmhd_ + n, k, j, i) = flx2(m,IDN,k,j,i)*wlj[PYF + n];
            } else {
              flx2(m, nmhd_ + n, k, j, i) = flx2(m,IDN,k,j,i)*wrj[PYF + n];
            }
          }
        }
      }

      if (three_d) {
        Real wmk[NPRIM], *wpk;
        Real bmk[NPRIM], bpk[NMAG];
        // Reconstruct states
        ExtractPrimitives(wmk, w0_, eos_, nmhd_, nscal_, m, k-1, j, i);
        wpk = &wri[0];
        if (fofc_flag || fofc_excision) {
          ExtractBField(bmk, bcc0_, IBZ, IBX, IBY, m, k-1, j, i);
          ExtractBField(bpk, bcc0_, IBZ, IBX, IBY, m, k, j, i);
          bmk[IBZ] = bpk[IBZ] = b0_.x3f(m, k, j, i);

          // Compute the metric terms at k-1/2
          adm::Face3Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha,
                            g3d, beta_u, alpha);

          // Compute new 1st-order LLF flux at k-face
          if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
            SingleStateLLF_DYNGR<IVZ>(eos_, wmk, wpk, bmk, bpk, nmhd_, nscal_,
                                      g3d, beta_u, alpha, flux, bflux);
          } else if (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
            SingleStateHLLE_DYNGR<IVZ>(eos_, wmk, wpk, bmk, bpk, nmhd_, nscal_,
                                      g3d, beta_u, alpha, flux, bflux);
          }

          // Store 1st-order fluxes at k-1/2
          InsertFluxes(flux, flx3, m, k, j, i);
          e2x3_(m,k,j,i) = bflux[IBY];
          e1x3_(m,k,j,i) = bflux[IBZ];
        }

        // Calculate fluxes of scalars
        for (int n = 0; n < nscal_; n++) {
          if (fofc_flag || fofc_scal_(m,n,k,j,i)) {
            if (flx3(m, IDN, k, j, i) >= 0.0) {
              flx3(m, nmhd_ + n, k, j, i) = flx3(m,IDN,k,j,i)*wmk[PYF + n];
            } else {
              flx3(m, nmhd_ + n, k, j, i) = flx3(m,IDN,k,j,i)*wpk[PYF + n];
            }
          }
        }
      }
    }
  });

  // Replace fluxes with first-order LLF fluxes at i+1,j+1,k+1 faces for any cell where
  // FOFC and/or excision is used (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Initialize FOFC scalar flag to be FOFC flag
    bool fofc_scalar_any = fofc_flag;
    if (use_fofc_ && nscal_ > 0 && !fofc_flag) {
      // Check if any scalar flag
      for (int n=0; n < nscal_; ++n) {
        fofc_scalar_any |= fofc_scal_(m,n,k,j,i);
      }
    }

    // Check for GR + excision
    bool fofc_excision = false;
    if (use_excise_) { fofc_excision = excision_flux_(m,k,j,i); }

    // Apply FOFC
    if (fofc_flag || fofc_excision || fofc_scalar_any) {
      // Reconstruct states
      Real wli[NPRIM], wri[NPRIM];
      Real bli[NMAG], bri[NMAG];
      ExtractPrimitives(wli, w0_,eos_, nmhd_, nscal_, m, k, j, i);
      ExtractPrimitives(wri, w0_, eos_, nmhd_, nscal_, m, k, j, i+1);
      Real g3d[NSPMETRIC], beta_u[3], alpha;
      Real flux[NCONS], bflux[NMAG];
      if (fofc_flag || fofc_excision) {
        ExtractBField(bli, bcc0_, IBX, IBY, IBZ, m, k, j, i);
        ExtractBField(bri, bcc0_, IBX, IBY, IBZ, m, k, j, i+1);
        bli[IBX] = bri[IBX] = b0_.x1f(m, k, j, i+1);

        // Compute the metric terms at i+1/2
        adm::Face1Metric(m, k, j, i+1, adm.g_dd, adm.beta_u, adm.alpha,
                          g3d, beta_u, alpha);

        // compute new 1st-order LLF flux at (i+1)-face
        if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
          SingleStateLLF_DYNGR<IVX>(eos_, wli, wri, bli, bri, nmhd_, nscal_,
                               g3d, beta_u, alpha, flux, bflux);
        } else if (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
          SingleStateHLLE_DYNGR<IVX>(eos_, wli, wri, bli, bri, nmhd_, nscal_,
                               g3d, beta_u, alpha, flux, bflux);
        }

        // Store 1st-order fluxes at i+1/2
        InsertFluxes(flux, flx1, m, k, j, i+1);
        e3x1_(m, k, j, i+1) = bflux[IBY];
        e2x1_(m, k, j, i+1) = bflux[IBZ];
      }

      // Calculate fluxes of scalars
      for (int n = 0; n < nscal_; n++) {
        if (fofc_flag || fofc_scal_(m,n,k,j,i)) {
          if (flx1(m, IDN, k, j, i+1) >= 0.0) {
            flx1(m, nmhd_ + n, k, j, i+1) = flx1(m,IDN,k,j,i+1)*wli[PYF + n];
          } else {
            flx1(m, nmhd_ + n, k, j, i+1) = flx1(m,IDN,k,j,i+1)*wri[PYF + n];
          }
        }
      }

      if (multi_d) {
        Real *wlj, wrj[NPRIM];
        Real blj[NMAG], brj[NMAG];
        // Reconstruct states
        wlj = &wli[0];
        ExtractPrimitives(wrj, w0_, eos_, nmhd_, nscal_, m, k, j+1, i);
        if (fofc_flag || fofc_excision) {
          ExtractBField(blj, bcc0_, IBY, IBZ, IBX, m, k, j, i);
          ExtractBField(brj, bcc0_, IBY, IBZ, IBX, m, k, j+1, i);
          blj[IBY] = brj[IBY] = b0_.x2f(m, k, j+1, i);

          // Compute the metric terms at j+1/2
          adm::Face2Metric(m, k, j+1, i, adm.g_dd, adm.beta_u, adm.alpha,
                           g3d, beta_u, alpha);

          // Compute new 1st-order LLF flux at j-face
          if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
            SingleStateLLF_DYNGR<IVY>(eos_, wlj, wrj, blj, brj, nmhd_, nscal_,
                                      g3d, beta_u, alpha, flux, bflux);
          } else if (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
            SingleStateHLLE_DYNGR<IVY>(eos_, wlj, wrj, blj, brj, nmhd_, nscal_,
                                      g3d, beta_u, alpha, flux, bflux);
          }

          // Store 1st-order fluxes at j+1/2
          InsertFluxes(flux, flx2, m, k, j+1, i);
          e1x2_(m,k,j+1,i) = bflux[IBY];
          e3x2_(m,k,j+1,i) = bflux[IBZ];
        }

        // Calculate fluxes of scalars
        for (int n = 0; n < nscal_; n++) {
          if (fofc_flag || fofc_scal_(m,n,k,j,i)) {
            if (flx2(m, IDN, k, j+1, i) >= 0.0) {
              flx2(m, nmhd_ + n, k, j+1, i) = flx2(m,IDN,k,j+1,i)*wlj[PYF + n];
            } else {
              flx2(m, nmhd_ + n, k, j+1, i) = flx2(m,IDN,k,j+1,i)*wrj[PYF + n];
            }
          }
        }
      }

      if (three_d) {
        Real *wmk, wpk[NPRIM];
        Real bmk[NPRIM], bpk[NMAG];
        // Reconstruct states
        wmk = &wli[0];
        ExtractPrimitives(wpk, w0_, eos_, nmhd_, nscal_, m, k+1, j, i);
        if (fofc_flag || fofc_excision) {
          ExtractBField(bmk, bcc0_, IBZ, IBX, IBY, m, k, j, i);
          ExtractBField(bpk, bcc0_, IBZ, IBX, IBY, m, k+1, j, i);
          bmk[IBZ] = bpk[IBZ] = b0_.x3f(m, k+1, j, i);

          // Compute the metric terms at k+1/2
          adm::Face3Metric(m, k+1, j, i, adm.g_dd, adm.beta_u, adm.alpha,
                           g3d, beta_u, alpha);

          // Compute new 1st-order LLF flux at k-face
          if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
            SingleStateLLF_DYNGR<IVZ>(eos_, wmk, wpk, bmk, bpk, nmhd_, nscal_,
                                      g3d, beta_u, alpha, flux, bflux);
          } else if (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
            SingleStateHLLE_DYNGR<IVZ>(eos_, wmk, wpk, bmk, bpk, nmhd_, nscal_,
                                      g3d, beta_u, alpha, flux, bflux);
          }

          // Store 1st-order fluxes at k+1/2
          InsertFluxes(flux, flx3, m, k+1, j, i);
          e2x3_(m,k+1,j,i) = bflux[IBY];
          e1x3_(m,k+1,j,i) = bflux[IBZ];
        }

        // Calculate fluxes of scalars
        for (int n = 0; n < nscal_; n++) {
          if (fofc_flag || fofc_scal_(m,n,k,j,i)) {
            if (flx3(m, IDN, k+1, j, i) >= 0.0) {
              flx3(m, nmhd_ + n, k+1, j, i) = flx3(m,IDN,k+1,j,i)*wmk[PYF + n];
            } else {
              flx3(m, nmhd_ + n, k+1, j, i) = flx3(m,IDN,k+1,j,i)*wpk[PYF + n];
            }
          }
        }
      }
    }
  });

  if (use_fofc_ && scalar_pplimiter && nscal_ > 0) {
    Real &gam0 = pdriver->gam0[stage-1];
    Real &gam1 = pdriver->gam1[stage-1];
    Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

    auto &u0_ = pmy_pack->pmhd->u0;
    auto &u1_ = pmy_pack->pmhd->u1;
    auto &utest_ = pmy_pack->pmhd->utest;

    // Estimate updated density
    par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      utest_(m,IDN,k,j,i) = gam0*u0_(m,IDN,k,j,i) + gam1*u1_(m,IDN,k,j,i);
      for (int n=0; n<nscal_; ++n) {
        utest_(m,nmhd_+n,k,j,i) = gam0*u0_(m,nmhd_+n,k,j,i) + gam1*u1_(m,nmhd_+n,k,j,i);
      }
    });

    // Index bounds
    Real alp_pp = 2.0;
    il = is, iu = ie+1;
    jl = js, ju = je;
    kl = ks, ku = ke;
    if (multi_d) { alp_pp = 4.0, jl = js, ju = je+1; }
    if (three_d) { alp_pp = 6.0, kl = ks, ku = ke+1; }

    // Positivity Presserving Limiter for scalar
    par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size.d_view(m).dx1;

      Real bet_pp = alp_pp * dtodx1;
      Real bet_ppi = 1.0 / bet_pp;

      Real wthe_m[MAX_SPECIES] = {0.0};
      Real wthe_p[MAX_SPECIES] = {0.0};
      Real wthe[MAX_SPECIES] = {0.0};
      Real flx_llf[MAX_SPECIES] = {0.0};

      // Evaluate LLF flux for scalar
      if ( flx1(m,IDN,k,j,i) >= 0.0 ) {
        for (int n=0; n < nscal_; ++n) {
          flx_llf[n] = w0_(m,nmhd_+n,k,j,i-1) * flx1(m,IDN,k,j,i);
        }
      } else {
        for (int n=0; n < nscal_; ++n) {
          flx_llf[n] = w0_(m,nmhd_+n,k,j,i) * flx1(m,IDN,k,j,i);
        }
      }

      // Estimate density D^- at boundary from i-1
      Real uD = utest_(m,IDN,k,j,i-1) - bet_pp * flx1(m,IDN,k,j,i);
      if ( uD > 0.0 ) {
        for (int n=0; n < nscal_; ++n) {
          Real uY_m = utest_(m,nmhd_+n,k,j,i-1) - bet_pp * flx1(m,nmhd_+n,k,j,i);
          Real min_DY_ = (eos_min_Y(n) + DBL_EPSILON) * uD;
          Real max_DY_ = (eos_max_Y(n) - DBL_EPSILON) * uD;
          if ( uY_m < min_DY_ ) {
            wthe_m[n] = ( ( utest_(m,nmhd_+n,k,j,i-1) - min_DY_ ) * bet_ppi
              - flx_llf[n] ) / ( flx1(m,nmhd_+n,k,j,i) - flx_llf[n] );
          } else if ( uY_m > max_DY_ ) {
            wthe_m[n] = ( ( max_DY_ - utest_(m,nmhd_+n,k,j,i-1) ) * bet_ppi
              + flx_llf[n] ) / ( flx_llf[n] - flx1(m,nmhd_+n,k,j,i) );
          } else {
            wthe_m[n] = 1.0;
          }
        }
      } else {
        for (int n=0; n < nscal_; ++n) {wthe_m[n] = 0.0;}
      }
      // Estimate density D^+ at boundary from i
      uD = utest_(m,IDN,k,j,i) + bet_pp * flx1(m,IDN,k,j,i);
      if ( uD > 0.0 ) {
        for (int n=0; n < nscal_; ++n) {
          Real uY_p = utest_(m,nmhd_+n,k,j,i) + bet_pp * flx1(m,nmhd_+n,k,j,i);
          Real min_DY_ = (eos_min_Y(n) + DBL_EPSILON) * uD;
          Real max_DY_ = (eos_max_Y(n) - DBL_EPSILON) * uD;
          if ( uY_p < min_DY_ ) {
            wthe_p[n] = ( ( utest_(m,nmhd_+n,k,j,i) - min_DY_ ) * bet_ppi
              + flx_llf[n] ) / ( flx_llf[n] - flx1(m,nmhd_+n,k,j,i) );
          } else if ( uY_p > max_DY_ ) {
            wthe_p[n] = ( ( max_DY_ - utest_(m,nmhd_+n,k,j,i) ) * bet_ppi
              - flx_llf[n] ) / ( flx1(m,nmhd_+n,k,j,i) - flx_llf[n] );
          } else {
            wthe_p[n] = 1.0;
          }
        }
      } else {
        for (int n=0; n < nscal_; ++n) {wthe_p[n] = 0.0;}
      }
      for (int n=0; n < nscal_; ++n) {
        wthe[n] = fmax(0.0, fmin(1.0, fmin(wthe_m[n], wthe_p[n])));
        flx1(m,nmhd_+n,k,j,i) = flx_llf[n]
          + wthe[n] * (flx1(m,nmhd_+n,k,j,i) - flx_llf[n]);
      }

      if (multi_d) {
        Real dtodx2 = beta_dt/size.d_view(m).dx2;
        bet_pp = alp_pp * dtodx2;
        bet_ppi = 1.0 / bet_pp;

        if ( flx2(m,IDN,k,j,i) >= 0.0 ) {
          for (int n=0; n < nscal_; ++n) {
            flx_llf[n] = w0_(m,nmhd_+n,k,j-1,i) * flx2(m,IDN,k,j,i);
          }
        } else {
          for (int n=0; n < nscal_; ++n) {
            flx_llf[n] = w0_(m,nmhd_+n,k,j,i) * flx2(m,IDN,k,j,i);
          }
        }

        uD = utest_(m,IDN,k,j-1,i) - bet_pp * flx2(m,IDN,k,j,i);
        if ( uD > 0.0 ) {
          for (int n=0; n < nscal_; ++n) {
            Real uY_m = utest_(m,nmhd_+n,k,j-1,i) - bet_pp * flx2(m,nmhd_+n,k,j,i);
            Real min_DY_ = (eos_min_Y(n) + DBL_EPSILON) * uD;
            Real max_DY_ = (eos_max_Y(n) - DBL_EPSILON) * uD;
            if ( uY_m < min_DY_ ) {
              wthe_m[n] = ( ( utest_(m,nmhd_+n,k,j-1,i) - min_DY_ ) * bet_ppi
                - flx_llf[n] ) / ( flx2(m,nmhd_+n,k,j,i) - flx_llf[n] );
            } else if ( uY_m > max_DY_ ) {
              wthe_m[n] = ( ( max_DY_ - utest_(m,nmhd_+n,k,j-1,i) ) * bet_ppi
                + flx_llf[n] ) / ( flx_llf[n] - flx2(m,nmhd_+n,k,j,i) );
            } else {
              wthe_m[n] = 1.0;
            }
          }
        } else {
          for (int n=0; n < nscal_; ++n) {wthe_m[n] = 0.0;}
        }
        uD = utest_(m,IDN,k,j,i) + bet_pp * flx2(m,IDN,k,j,i);
        if ( uD > 0.0 ) {
          for (int n=0; n < nscal_; ++n) {
            Real uY_p = utest_(m,nmhd_+n,k,j,i) + bet_pp * flx2(m,nmhd_+n,k,j,i);
            Real min_DY_ = (eos_min_Y(n) + DBL_EPSILON) * uD;
            Real max_DY_ = (eos_max_Y(n) - DBL_EPSILON) * uD;
            if ( uY_p < min_DY_ ) {
              wthe_p[n] = ( ( utest_(m,nmhd_+n,k,j,i) - min_DY_ ) * bet_ppi
                + flx_llf[n] ) / ( flx_llf[n] - flx2(m,nmhd_+n,k,j,i) );
            } else if ( uY_p > max_DY_ ) {
              wthe_p[n] = ( ( max_DY_ - utest_(m,nmhd_+n,k,j,i) ) * bet_ppi
                - flx_llf[n] ) / ( flx2(m,nmhd_+n,k,j,i) - flx_llf[n] );
            } else {
              wthe_p[n] = 1.0;
            }
          }
        } else {
          for (int n=0; n < nscal_; ++n) {wthe_p[n] = 0.0;}
        }
        for (int n=0; n < nscal_; ++n) {
          wthe[n] = fmax(0.0, fmin(1.0, fmin(wthe_m[n], wthe_p[n])));
          flx2(m,nmhd_+n,k,j,i) = flx_llf[n]
            + wthe[n] * (flx2(m,nmhd_+n,k,j,i) - flx_llf[n]);
        }
      }

      if (three_d) {
        Real dtodx3 = beta_dt/size.d_view(m).dx3;
        bet_pp = alp_pp * dtodx3;
        bet_ppi = 1.0 / bet_pp;

        if ( flx3(m,IDN,k,j,i) >= 0.0 ) {
          for (int n=0; n < nscal_; ++n) {
            flx_llf[n] = w0_(m,nmhd_+n,k-1,j,i) * flx3(m,IDN,k,j,i);
          }
        } else {
          for (int n=0; n < nscal_; ++n) {
            flx_llf[n] = w0_(m,nmhd_+n,k,j,i) * flx3(m,IDN,k,j,i);
          }
          bet_pp = - utest_(m,IDN,k,j,i) / flx3(m,IDN,k,j,i);
        }

        uD = utest_(m,IDN,k-1,j,i) - bet_pp * flx3(m,IDN,k,j,i);
        if ( uD > 0.0 ) {
          for (int n=0; n < nscal_; ++n) {
            Real uY_m = utest_(m,nmhd_+n,k-1,j,i) - bet_pp * flx3(m,nmhd_+n,k,j,i);
            Real min_DY_ = (eos_min_Y(n) + DBL_EPSILON) * uD;
            Real max_DY_ = (eos_max_Y(n) - DBL_EPSILON) * uD;
            if ( uY_m < min_DY_ ) {
              wthe_m[n] = ( ( utest_(m,nmhd_+n,k-1,j,i) - min_DY_ ) * bet_ppi
                - flx_llf[n] ) / ( flx3(m,nmhd_+n,k,j,i) - flx_llf[n] );
            } else if ( uY_m > max_DY_ ) {
              wthe_m[n] = ( ( max_DY_ - utest_(m,nmhd_+n,k-1,j,i) ) * bet_ppi
                + flx_llf[n] ) / ( flx_llf[n] - flx3(m,nmhd_+n,k,j,i) );
            } else {
              wthe_m[n] = 1.0;
            }
          }
        } else {
          for (int n=0; n < nscal_; ++n) {wthe_m[n] = 0.0;}
        }
        uD = utest_(m,IDN,k,j,i) + bet_pp * flx3(m,IDN,k,j,i);
        if ( uD > 0.0 ) {
          for (int n=0; n < nscal_; ++n) {
            Real uY_p = utest_(m,nmhd_+n,k,j,i) + bet_pp * flx3(m,nmhd_+n,k,j,i);
            Real min_DY_ = (eos_min_Y(n) + DBL_EPSILON) * uD;
            Real max_DY_ = (eos_max_Y(n) - DBL_EPSILON) * uD;
            if ( uY_p < min_DY_ ) {
              wthe_p[n] = ( ( utest_(m,nmhd_+n,k,j,i) - min_DY_ ) * bet_ppi
                + flx_llf[n] ) / ( flx_llf[n] - flx3(m,nmhd_+n,k,j,i) );
            } else if ( uY_p > max_DY_ ) {
              wthe_p[n] = ( ( max_DY_ - utest_(m,nmhd_+n,k,j,i) ) * bet_ppi
                - flx_llf[n] ) / ( flx3(m,nmhd_+n,k,j,i) - flx_llf[n] );
            } else {
              wthe_p[n] = 1.0;
            }
          }
        } else {
          for (int n=0; n < nscal_; ++n) {wthe_p[n] = 0.0;}
        }
        for (int n=0; n < nscal_; ++n) {
          wthe[n] = fmax(0.0, fmin(1.0, fmin(wthe_m[n], wthe_p[n])));
          flx3(m,nmhd_+n,k,j,i) = flx_llf[n]
            + wthe[n] * (flx3(m,nmhd_+n,k,j,i) - flx_llf[n]);
        }
      }
    });
  }

  // reset FOFC flag (do not reset excision flag)
  if (use_fofc_) {
    Kokkos::deep_copy(fofc_, false);
    if (nscal_ > 0) {Kokkos::deep_copy(fofc_scal_, false);}
  }

  return;
}

// function definitions for each template parameter
#define INSTANTIATE_FOFC(EOSPolicy, ErrorPolicy) \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::\
  FOFC<DynGRMHD_RSolver::llf_dyngr>(Driver *pdriver, int stage);\
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::\
  FOFC<DynGRMHD_RSolver::hlle_dyngr>(Driver *pdriver, int stage);

INSTANTIATE_FOFC(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_FOFC(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_FOFC(Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_FOFC(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor)
INSTANTIATE_FOFC(Primitive::EOSHybrid<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_FOFC(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor)
} // namespace dyngr
