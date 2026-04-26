//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_newdt.cpp
//! \brief function to compute rad timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>
#include <float.h>

#include <limits>
#include <iostream>
#include <iomanip>    // std::setprecision()
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "driver/driver.hpp"
#include "dyn_radiation.hpp"
#include "dyn_radiation/dyn_radiation_tetrad.hpp"

namespace dyn_radiation {

//----------------------------------------------------------------------------------------
// \!fn void DynRadiation::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for dyn_radiation problems.
//        Only computed once at beginning of calculation.

TaskStatus DynRadiation::NewTimeStep(Driver *pdriver, int stage) {
  if (use_adm_geometry) {
    PrepareADMGeometry();
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &nx1 = indcs.nx1;
  int &js = indcs.js, &nx2 = indcs.nx2;
  int &ks = indcs.ks, &nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();
  Real dta = std::numeric_limits<float>::max();
  Real dtg = std::numeric_limits<float>::max();

  // setup indicies for Kokkos parallel reduce
  auto &size = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nang1 = prgeo->nangles - 1;

  // data needed to compute angular dt
  bool &angular_fluxes_ = angular_fluxes;
  auto &nh_c_ = nh_c;
  auto &na_ = na;
  auto &tet_c_ = tet_c;
  auto &t1d1 = tet_d1_x1f;
  auto &t2d2 = tet_d2_x2f;
  auto &t3d3 = tet_d3_x3f;
  bool use_adm_geometry_ = use_adm_geometry;
  bool adm_metric_source_ = adm_metric_source;
  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  auto &numn = prgeo->num_neighbors;
  auto &indn = prgeo->ind_neighbors;

  // find smallest (dx/c) and (dangle/na) in each direction for dyn_radiation problems
  Kokkos::parallel_reduce("RadiationNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx,Real &min_dt1,Real &min_dt2,Real &min_dt3,Real &min_dta) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real tmp_min_dta = (FLT_MAX);
    if (angular_fluxes_) {
      for (int n=0; n<=nang1; ++n) {
        // find position at angle center
        Real x = nh_c_.d_view(n,1);
        Real y = nh_c_.d_view(n,2);
        Real z = nh_c_.d_view(n,3);
        for (int nb=0; nb<numn.d_view(n); ++nb) {
          // find position at neighbor's angle center
          Real xn = nh_c_.d_view(indn.d_view(n,nb),1);
          Real yn = nh_c_.d_view(indn.d_view(n,nb),2);
          Real zn = nh_c_.d_view(indn.d_view(n,nb),3);
          // compute timestep limitation
          Real dot = fmin(1.0, fmax(-1.0, x*xn+y*yn+z*zn));
          Real omega = fabs(na_(m,n,k,j,i,nb));
          if (!(use_adm_geometry_)) {
            Real n0 = tet_c_(m,0,0,k,j,i);
            omega = fabs(na_(m,n,k,j,i,nb)/n0);
          }
          Real adt = tmp_min_dta;
          if (omega > 1.0e-300) {
            adt = fmin(tmp_min_dta, acos(dot)/omega);
          }
          // set timestep limitation if not excising this cell
          if (excise) {
            if (!(rad_mask_(m,k,j,i))) { tmp_min_dta = adt; }
          } else {
            tmp_min_dta = adt;
          }
        }
      }
    }
    Real cmax1 = 1.0;
    Real cmax2 = 1.0;
    Real cmax3 = 1.0;
    if (use_adm_geometry_) {
      cmax1 = 1.0e-300;
      cmax2 = 1.0e-300;
      cmax3 = 1.0e-300;
      for (int n=0; n<=nang1; ++n) {
        Real v1 = t1d1(m,0,k,j,i)*nh_c_.d_view(n,0) +
                  t1d1(m,1,k,j,i)*nh_c_.d_view(n,1) +
                  t1d1(m,2,k,j,i)*nh_c_.d_view(n,2) +
                  t1d1(m,3,k,j,i)*nh_c_.d_view(n,3);
        Real v1p = t1d1(m,0,k,j,i+1)*nh_c_.d_view(n,0) +
                   t1d1(m,1,k,j,i+1)*nh_c_.d_view(n,1) +
                   t1d1(m,2,k,j,i+1)*nh_c_.d_view(n,2) +
                   t1d1(m,3,k,j,i+1)*nh_c_.d_view(n,3);
        cmax1 = fmax(cmax1, fabs(v1));
        cmax1 = fmax(cmax1, fabs(v1p));
        if (nx2 > 1) {
          Real v2 = t2d2(m,0,k,j,i)*nh_c_.d_view(n,0) +
                    t2d2(m,1,k,j,i)*nh_c_.d_view(n,1) +
                    t2d2(m,2,k,j,i)*nh_c_.d_view(n,2) +
                    t2d2(m,3,k,j,i)*nh_c_.d_view(n,3);
          Real v2p = t2d2(m,0,k,j+1,i)*nh_c_.d_view(n,0) +
                     t2d2(m,1,k,j+1,i)*nh_c_.d_view(n,1) +
                     t2d2(m,2,k,j+1,i)*nh_c_.d_view(n,2) +
                     t2d2(m,3,k,j+1,i)*nh_c_.d_view(n,3);
          cmax2 = fmax(cmax2, fabs(v2));
          cmax2 = fmax(cmax2, fabs(v2p));
        }
        if (nx3 > 1) {
          Real v3 = t3d3(m,0,k,j,i)*nh_c_.d_view(n,0) +
                    t3d3(m,1,k,j,i)*nh_c_.d_view(n,1) +
                    t3d3(m,2,k,j,i)*nh_c_.d_view(n,2) +
                    t3d3(m,3,k,j,i)*nh_c_.d_view(n,3);
          Real v3p = t3d3(m,0,k+1,j,i)*nh_c_.d_view(n,0) +
                     t3d3(m,1,k+1,j,i)*nh_c_.d_view(n,1) +
                     t3d3(m,2,k+1,j,i)*nh_c_.d_view(n,2) +
                     t3d3(m,3,k+1,j,i)*nh_c_.d_view(n,3);
          cmax3 = fmax(cmax3, fabs(v3));
          cmax3 = fmax(cmax3, fabs(v3p));
        }
      }
    }
    min_dt1 = fmin((size.d_view(m).dx1/cmax1), min_dt1);
    min_dt2 = fmin((size.d_view(m).dx2/cmax2), min_dt2);
    min_dt3 = fmin((size.d_view(m).dx3/cmax3), min_dt3);
    min_dta = fmin((tmp_min_dta),        min_dta);
  }, Kokkos::Min<Real>(dt1),  Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3),
     Kokkos::Min<Real>(dta));

  if (use_adm_geometry_ && adm_metric_source_) {
    auto &adm_ = pmy_pack->padm->adm;
    auto &adm_grad_alpha_c_ = adm_grad_alpha_c;
    Kokkos::parallel_reduce("RadiationGeomDt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dtg) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real tmp_min_dtg = (FLT_MAX);
      for (int n=0; n<=nang1; ++n) {
        Real s[3] = {0.0, 0.0, 0.0};
        for (int a=0; a<3; ++a) {
          for (int d=0; d<3; ++d) {
            s[d] += tet_c_(m,a+1,d+1,k,j,i)*nh_c_.d_view(n,a+1);
          }
        }
        Real kss = 0.0;
        Real sdalpha = 0.0;
        for (int a=0; a<3; ++a) {
          sdalpha += s[a]*adm_grad_alpha_c_(m,a,k,j,i);
          for (int b=0; b<3; ++b) {
            kss += adm_.vK_dd(m,a,b,k,j,i)*s[a]*s[b];
          }
        }
        Real geom = adm_.alpha(m,k,j,i)*kss - sdalpha;
        if (fabs(geom) > 1.0e-300) {
          tmp_min_dtg = fmin(tmp_min_dtg, 1.0/fabs(geom));
        }
      }
      min_dtg = fmin(tmp_min_dtg, min_dtg);
    }, Kokkos::Min<Real>(dtg));
  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }
  if (angular_fluxes_) { dtnew = std::min(dtnew, dta); }
  if (use_adm_geometry && adm_metric_source) { dtnew = std::min(dtnew, dtg); }

  return TaskStatus::complete;
}
} // namespace dyn_radiation
