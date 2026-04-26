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
  bool adm_metric_source_ = adm_metric_source;

  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = n_0_floor;

  if (use_adm_geometry_) {
    auto &adm_ = pmy_pack->padm->adm;
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
            kss += adm_.vK_dd(m,a,b,k,j,i)*s[a]*s[b];
          }
        }
        Real geom = adm_.alpha(m,k,j,i)*kss - sdalpha;
        i_new += beta_dt*i_stage*geom;
      }

      i0_(m,n,k,j,i) = fmax(i_new, 0.0);
      if (excise && rad_mask_(m,k,j,i)) { i0_(m,n,k,j,i) = 0.0; }
    });
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
    i0_(m,n,k,j,i) = n0*n_0*fmax((i0_(m,n,k,j,i)/(n0*n_0)), 0.0);

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
  return TaskStatus::complete;
}
} // namespace dyn_radiation
