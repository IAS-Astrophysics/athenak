//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_update.cpp
//  \brief Performs update of Radiation conserved variables (i0) for each stage of
//   explicit SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and
//   partial time step appropriate to stage.
//  Explicit (not implicit) dyn_radiation source terms are included in this update.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "srcterms/srcterms.hpp"
#include "dyn_radiation.hpp"

namespace dyn_radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::Update
//  \brief Explicit RK update of flux divergence and physical source terms

TaskStatus DynRadiation::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nang1 = prgeo->nangles - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &mbsize  = pmy_pack->pmb->mb_size;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  auto &i0_ = i0;
  auto &i1_ = i1;
  auto &flx1 = iflx.x1f;
  auto &flx2 = iflx.x2f;
  auto &flx3 = iflx.x3f;

  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;

  auto &angular_fluxes_ = angular_fluxes;
  auto &divfa_ = divfa;
  bool use_adm_geometry_ = use_adm_geometry;
  // With killing_weight the evolved variable W = -n_0 U is conserved along geodesics,
  // so the transport is source-free; red/blueshift lives in the reconstruction weight.
  bool adm_metric_source_ = adm_metric_source && !(killing_weight);
  auto &solid_angles_ = prgeo->solid_angles;

  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = n_0_floor;
  bool n0_absorb_ = excision_n0_absorb;
  const bool fm_ = frequency_moments;
  const int nang_ = prgeo->nangles;
  const Real nu_cap_ = nu_cap;
  auto &deleted_budget_ = deleted_budget;

  if (use_adm_geometry_) {
    auto &adm_alpha_c_ = adm_alpha_c;
    auto &adm_K_dd_c_ = adm_K_dd_c;
    auto &adm_grad_alpha_c_ = adm_grad_alpha_c;
    par_for("dynrad_adm_update",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      Real divf_s = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
      if (multi_d) {
        divf_s += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
      }
      if (three_d) {
        divf_s += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
      }
      Real i_stage = i0_(m,n,k,j,i);
      Real i_new = gam0*i_stage + gam1*i1_(m,n,k,j,i) - beta_dt*divf_s;
      if (angular_fluxes_) { i_new -= beta_dt*divfa_(m,n,k,j,i); }

      if (adm_metric_source_) {
        Real s[3] = {0.0, 0.0, 0.0};
        for (int a=0; a<3; ++a) {
          for (int d=0; d<3; ++d) {
            s[d] += tt(m,a+1,d+1,k,j,i)*nh_c_.d_view(n,a+1);
          }
        }

        Real grad_alpha[3] = {adm_grad_alpha_c_(m,0,k,j,i),
                              adm_grad_alpha_c_(m,1,k,j,i),
                              adm_grad_alpha_c_(m,2,k,j,i)};

        Real kss = 0.0;
        Real sdalpha = 0.0;
        for (int a=0; a<3; ++a) {
          sdalpha += s[a]*grad_alpha[a];
          for (int b=0; b<3; ++b) {
            kss += adm_K_dd_c_(m,a,b,k,j,i)*s[a]*s[b];
          }
        }
        Real geom = adm_alpha_c_(m,k,j,i)*kss - sdalpha;
        i_new += i_stage*(exp(beta_dt*geom) - 1.0);
      }

      i0_(m,n,k,j,i) = i_new;

      // Photon-number field: same flux-divergence update, but NO geometric source
      // (photon number is exactly conserved along geodesics).
      if (fm_) {
        const int nn = nang_ + n;
        Real divn_s = (flx1(m,nn,k,j,i+1) - flx1(m,nn,k,j,i))/mbsize.d_view(m).dx1;
        if (multi_d) {
          divn_s += (flx2(m,nn,k,j+1,i) - flx2(m,nn,k,j,i))/mbsize.d_view(m).dx2;
        }
        if (three_d) {
          divn_s += (flx3(m,nn,k+1,j,i) - flx3(m,nn,k,j,i))/mbsize.d_view(m).dx3;
        }
        Real n_new = gam0*i0_(m,nn,k,j,i) + gam1*i1_(m,nn,k,j,i) - beta_dt*divn_s;
        if (angular_fluxes_) { n_new -= beta_dt*divfa_(m,nn,k,j,i); }
        i0_(m,nn,k,j,i) = n_new;
      }

      if (excise) {
        if (rad_mask_(m,k,j,i)) {
          i0_(m,n,k,j,i) = 0.0;
          if (fm_) { i0_(m,nang_+n,k,j,i) = 0.0; }
        } else if (n0_absorb_) {
          // Absorb bins with near-zero Killing energy (n_0 = -alpha + beta.s ~ 0).
          // Such photons exist only inside the ergosphere and cannot escape; they
          // are also the directions whose Eulerian blueshift (the exp(geom*dt)
          // gain) is unbounded, so removing them caps the accumulated gain and
          // breaks the discrete gain/diffusion feedback loop at the mask edge.
          Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
                     tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                     tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
                     tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          if (fabs(n_0) < n_0_floor_) {
            i0_(m,n,k,j,i) = 0.0;
            if (fm_) { i0_(m,nang_+n,k,j,i) = 0.0; }
          }
        }
      }
    });
    par_for("dynrad_adm_update_positivity",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      ConservativeAngularFloor(i0_, solid_angles_, m, k, j, i, nang1);
      if (fm_) {
        ConservativeAngularFloor(i0_, solid_angles_, m, k, j, i, nang1, nang_);
      }
    });
    if (fm_) {
      // Frequency cap: bins whose mean frequency E/N exceeds nu_cap have blueshifted
      // off the top of the (one-group) frequency grid and are removed.  Physical
      // photons in these problems never accumulate more than a factor of a few of
      // blueshift before capture or escape, so the cap only bites numerically
      // manufactured trapped content.  The deleted conserved budget is accumulated
      // for the run log.
      par_for("dynrad_adm_nu_cap",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real dvol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2
                         *mbsize.d_view(m).dx3;
        for (int n=0; n<nang_; ++n) {
          const Real e = i0_(m,n,k,j,i);
          if (!(e > 0.0)) { continue; }
          const Real nn = i0_(m,nang_+n,k,j,i);
          if (e > nu_cap_*nn) {
            Kokkos::atomic_add(&deleted_budget_(0),
                               e*solid_angles_.d_view(n)*dvol);
            Kokkos::atomic_add(&deleted_budget_(1),
                               fmax(nn,0.0)*solid_angles_.d_view(n)*dvol);
            i0_(m,n,k,j,i) = 0.0;
            i0_(m,nang_+n,k,j,i) = 0.0;
          }
        }
      });
    }
    ApplyExcisionToIntensity(i0);
    return TaskStatus::complete;
  }

  par_for("r_update",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    // spatial fluxes
    Real divf_s = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      divf_s += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
    }
    if (three_d) {
      divf_s += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
    }
    i0_(m,n,k,j,i) = gam0*i0_(m,n,k,j,i)+gam1*i1_(m,n,k,j,i)-beta_dt*divf_s;

    // angular fluxes
    if (angular_fluxes_) { i0_(m,n,k,j,i) -= beta_dt*divfa_(m,n,k,j,i); }

    Real n_0 = 1.0;
    // zero intensity if negative
    Real n0  = tt(m,0,0,k,j,i);
    n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
          tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
          tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
          tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
    // handle excision
    // NOTE(@pdmullen): exicision criterion are not finalized.  The below zeroes all
    // intensities within rks <= 1.0 and zeroes intensities within angles where n_0
    // is about zero.  This needs future attention.
    if (excise) {
      if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) {
        i0_(m,n,k,j,i) = 0.0;
      }
    }
  });
  par_for("r_update_positivity",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    ConservativePrimitiveAngularFloor(i0_, solid_angles_, tt, tc, nh_c_,
                                      m, k, j, i, nang1);
  });
  ApplyExcisionToIntensity(i0);
  return TaskStatus::complete;
}
} // namespace dyn_radiation
