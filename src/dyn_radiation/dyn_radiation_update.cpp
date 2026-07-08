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

#include <iostream>
#include <iomanip>

#include "athena.hpp"
#include "globals.hpp"
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

  // Per-task W budget diagnostic: totals before/after each sub-operation isolate
  // which operator fails to conserve the Killing-energy total.
  const bool wbud = debug_w_budget && (global_variable::my_rank == 0);
  Real wb_pre = 0.0, wb_i1 = 0.0;
  if (wbud) {
    wb_pre = DebugWTotal();
    auto i0_saved = i0;
    i0 = i1; wb_i1 = DebugWTotal(); i0 = i0_saved;
  }

  // Face-integrated flux sums per meshblock: conservation demands the two sides
  // of every interior seam agree exactly, so a mismatch localizes the leak.
  if (wbud && (stage == 1) && (pmy_pack->pmesh->ncycle % 20 == 0)) {
    auto &solid_angles_dbg = prgeo->solid_angles;
    auto &fx1 = iflx.x1f;
    auto &fx2 = iflx.x2f;
    auto &fx3 = iflx.x3f;
    const int nang_tot = prgeo->nangles;
    for (int m=0; m<pmy_pack->nmb_thispack; ++m) {
      Real fsum[6];
      const Real dx1 = pmy_pack->pmb->mb_size.h_view(m).dx1;
      const Real dx2 = pmy_pack->pmb->mb_size.h_view(m).dx2;
      const Real dx3 = pmy_pack->pmb->mb_size.h_view(m).dx3;
      const int nkj = (ke-ks+1)*(je-js+1);
      const int nki = (ke-ks+1)*(ie-is+1);
      const int nji_d = (je-js+1)*(ie-is+1);
      for (int f=0; f<6; ++f) {
        Real s = 0.0;
        if (f < 2) {
          const int ii = (f == 0) ? is : ie+1;
          Kokkos::parallel_reduce("dbg_fx1",
            Kokkos::RangePolicy<>(DevExeSpace(), 0, nang_tot*nkj),
          KOKKOS_LAMBDA(const int idx, Real &acc) {
            const int n = idx/nkj;
            const int k = ks + (idx - n*nkj)/(je-js+1);
            const int j = js + (idx - n*nkj)%(je-js+1);
            acc += fx1(m,n,k,j,ii)*solid_angles_dbg.d_view(n);
          }, Kokkos::Sum<Real>(s));
          s *= dx2*dx3;
        } else if (f < 4) {
          const int jj = (f == 2) ? js : je+1;
          Kokkos::parallel_reduce("dbg_fx2",
            Kokkos::RangePolicy<>(DevExeSpace(), 0, nang_tot*nki),
          KOKKOS_LAMBDA(const int idx, Real &acc) {
            const int n = idx/nki;
            const int k = ks + (idx - n*nki)/(ie-is+1);
            const int i = is + (idx - n*nki)%(ie-is+1);
            acc += fx2(m,n,k,jj,i)*solid_angles_dbg.d_view(n);
          }, Kokkos::Sum<Real>(s));
          s *= dx1*dx3;
        } else {
          const int kk = (f == 4) ? ks : ke+1;
          Kokkos::parallel_reduce("dbg_fx3",
            Kokkos::RangePolicy<>(DevExeSpace(), 0, nang_tot*nji_d),
          KOKKOS_LAMBDA(const int idx, Real &acc) {
            const int n = idx/nji_d;
            const int j = js + (idx - n*nji_d)/(ie-is+1);
            const int i = is + (idx - n*nji_d)%(ie-is+1);
            acc += fx3(m,n,kk,j,i)*solid_angles_dbg.d_view(n);
          }, Kokkos::Sum<Real>(s));
          s *= dx1*dx2;
        }
        fsum[f] = s;
      }
      std::cout << "WFACE cyc=" << pmy_pack->pmesh->ncycle << " m=" << m
                << std::scientific << std::setprecision(10)
                << " x[" << pmy_pack->pmb->mb_size.h_view(m).x1min << ","
                << pmy_pack->pmb->mb_size.h_view(m).x2min << "]"
                << " xL=" << fsum[0] << " xR=" << fsum[1]
                << " yL=" << fsum[2] << " yR=" << fsum[3]
                << " zL=" << fsum[4] << " zR=" << fsum[5] << std::endl;
    }
  }

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

      if (excise) {
        if (rad_mask_(m,k,j,i)) {
          i0_(m,n,k,j,i) = 0.0;
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
          }
        }
      }
    });
    Real wb_flux = wbud ? DebugWTotal() : 0.0;
    par_for("dynrad_adm_update_positivity",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      ConservativeAngularFloor(i0_, solid_angles_, m, k, j, i, nang1);
    });
    Real wb_floor = wbud ? DebugWTotal() : 0.0;
    ApplyExcisionToIntensity(i0);
    if (wbud) {
      Real wb_exc = DebugWTotal();
      Real expect = gam0*wb_pre + gam1*wb_i1;
      std::cout << "WBUD cyc=" << pmy_pack->pmesh->ncycle << " stg=" << stage
                << std::scientific << std::setprecision(6)
                << " pre=" << wb_pre << " expect=" << expect
                << " dflux=" << (wb_flux - expect)
                << " dfloor=" << (wb_floor - wb_flux)
                << " dexc=" << (wb_exc - wb_floor)
                << " post=" << wb_exc << std::endl;
    }
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
