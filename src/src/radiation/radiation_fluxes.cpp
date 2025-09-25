//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fluxes.cpp
//  \brief Calculate 3D fluxes for radiation

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "radiation.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CalculateFluxes
//! \brief Compute radiation fluxes

TaskStatus Radiation::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nang1 = prgeo->nangles - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  const auto &recon_method_ = recon_method;

  auto &i0_ = i0;
  auto &nh_c_ = nh_c;
  auto &tet_c_ = tet_c;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto &t1d1 = tet_d1_x1f;
  auto &flx1 = iflx.x1f;
  par_for("rflux_x1",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie+1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    // calculate n^1 (hence determining upwinding direction)
    Real n1 = t1d1(m,0,k,j,i)*nh_c_.d_view(n,0) + t1d1(m,1,k,j,i)*nh_c_.d_view(n,1)
            + t1d1(m,2,k,j,i)*nh_c_.d_view(n,2) + t1d1(m,3,k,j,i)*nh_c_.d_view(n,3);

    // convert to primitive n_0 I
    Real iim1, iicc, iim2, iip1, iim3, iip2;
    iim1 = i0_(m,n,k,j,i-1)/tet_c_(m,0,0,k,j,i-1);
    iicc = i0_(m,n,k,j,i  )/tet_c_(m,0,0,k,j,i  );
    if (recon_method_ > 0) {
      iim2 = i0_(m,n,k,j,i-2)/tet_c_(m,0,0,k,j,i-2);
      iip1 = i0_(m,n,k,j,i+1)/tet_c_(m,0,0,k,j,i+1);
    }
    if (recon_method_ > 1) {
      iim3 = i0_(m,n,k,j,i-3)/tet_c_(m,0,0,k,j,i-3);
      iip2 = i0_(m,n,k,j,i+2)/tet_c_(m,0,0,k,j,i+2);
    }

    // reconstruct primitive intensity
    Real iiu, scr;
    switch (recon_method_) {
      case ReconstructionMethod::dc:
        if (n1 > 0.0) iiu = iim1;
        else          iiu = iicc;
        break;
      case ReconstructionMethod::plm:
        if (n1 > 0.0) PLM(iim2, iim1, iicc, iiu, scr);
        else          PLM(iim1, iicc, iip1, scr, iiu);
        break;
      case ReconstructionMethod::ppm4:
        if (n1 > 0.0) PPM4(iim3, iim2, iim1, iicc, iip1, iiu, scr);
        else          PPM4(iim2, iim1, iicc, iip1, iip2, scr, iiu);
        break;
      case ReconstructionMethod::ppmx:
        if (n1 > 0.0) PPMX(iim3, iim2, iim1, iicc, iip1, iiu, scr);
        else          PPMX(iim2, iim1, iicc, iip1, iip2, scr, iiu);
        break;
      case ReconstructionMethod::wenoz:
        if (n1 > 0.0) WENOZ(iim3, iim2, iim1, iicc, iip1, iiu, scr);
        else          WENOZ(iim2, iim1, iicc, iip1, iip2, scr, iiu);
        break;
      default:
        break;
    }

    // compute x1flux
    flx1(m,n,k,j,i) = n1*iiu;
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    auto &t2d2 = tet_d2_x2f;
    auto &flx2 = iflx.x2f;
    par_for("rflux_x2",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // calculate n^2 (hence determining upwinding direction)
      Real n2 = t2d2(m,0,k,j,i)*nh_c_.d_view(n,0) + t2d2(m,1,k,j,i)*nh_c_.d_view(n,1)
              + t2d2(m,2,k,j,i)*nh_c_.d_view(n,2) + t2d2(m,3,k,j,i)*nh_c_.d_view(n,3);

      // convert to primitive n_0 I
      Real iim1, iicc, iim2, iip1, iim3, iip2;
      iim1 = i0_(m,n,k,j-1,i)/tet_c_(m,0,0,k,j-1,i);
      iicc = i0_(m,n,k,j  ,i)/tet_c_(m,0,0,k,j  ,i);
      if (recon_method_ > 0) {
        iim2 = i0_(m,n,k,j-2,i)/tet_c_(m,0,0,k,j-2,i);
        iip1 = i0_(m,n,k,j+1,i)/tet_c_(m,0,0,k,j+1,i);
      }
      if (recon_method_ > 1) {
        iim3 = i0_(m,n,k,j-3,i)/tet_c_(m,0,0,k,j-3,i);
        iip2 = i0_(m,n,k,j+2,i)/tet_c_(m,0,0,k,j+2,i);
      }

      // reconstruct primitive intensity
      Real iiu, scr;
      switch (recon_method_) {
        case ReconstructionMethod::dc:
          if (n2 > 0.0) iiu = iim1;
          else          iiu = iicc;
          break;
        case ReconstructionMethod::plm:
          if (n2 > 0.0) PLM(iim2, iim1, iicc, iiu, scr);
          else          PLM(iim1, iicc, iip1, scr, iiu);
          break;
        case ReconstructionMethod::ppm4:
          if (n2 > 0.0) PPM4(iim3, iim2, iim1, iicc, iip1, iiu, scr);
          else          PPM4(iim2, iim1, iicc, iip1, iip2, scr, iiu);
          break;
        case ReconstructionMethod::ppmx:
          if (n2 > 0.0) PPMX(iim3, iim2, iim1, iicc, iip1, iiu, scr);
          else          PPMX(iim2, iim1, iicc, iip1, iip2, scr, iiu);
          break;
        case ReconstructionMethod::wenoz:
          if (n2 > 0.0) WENOZ(iim3, iim2, iim1, iicc, iip1, iiu, scr);
          else          WENOZ(iim2, iim1, iicc, iip1, iip2, scr, iiu);
          break;
        default:
          break;
      }

      // compute x2flux
      flx2(m,n,k,j,i) = n2*iiu;
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    auto &t3d3 = tet_d3_x3f;
    auto &flx3 = iflx.x3f;
    par_for("rflux_x3",DevExeSpace(),0,nmb1,0,nang1,ks,ke+1,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // calculate n^3 (hence determining upwinding direction)
      Real n3 = t3d3(m,0,k,j,i)*nh_c_.d_view(n,0) + t3d3(m,1,k,j,i)*nh_c_.d_view(n,1)
              + t3d3(m,2,k,j,i)*nh_c_.d_view(n,2) + t3d3(m,3,k,j,i)*nh_c_.d_view(n,3);

      // convert to primitive n_0 I
      Real iim1, iicc, iim2, iip1, iim3, iip2;
      iim1 = i0_(m,n,k-1,j,i)/tet_c_(m,0,0,k-1,j,i);
      iicc = i0_(m,n,k  ,j,i)/tet_c_(m,0,0,k  ,j,i);
      if (recon_method_ > 0) {
        iim2 = i0_(m,n,k-2,j,i)/tet_c_(m,0,0,k-2,j,i);
        iip1 = i0_(m,n,k+1,j,i)/tet_c_(m,0,0,k+1,j,i);
      }
      if (recon_method_ > 1) {
        iim3 = i0_(m,n,k-3,j,i)/tet_c_(m,0,0,k-3,j,i);
        iip2 = i0_(m,n,k+2,j,i)/tet_c_(m,0,0,k+2,j,i);
      }

      // reconstruct primitive intensity
      Real iiu, scr;
      switch (recon_method_) {
        case ReconstructionMethod::dc:
          if (n3 > 0.0) iiu = iim1;
          else          iiu = iicc;
          break;
        case ReconstructionMethod::plm:
          if (n3 > 0.0) PLM(iim2, iim1, iicc, iiu, scr);
          else          PLM(iim1, iicc, iip1, scr, iiu);
          break;
        case ReconstructionMethod::ppm4:
          if (n3 > 0.0) PPM4(iim3, iim2, iim1, iicc, iip1, iiu, scr);
          else          PPM4(iim2, iim1, iicc, iip1, iip2, scr, iiu);
          break;
        case ReconstructionMethod::ppmx:
          if (n3 > 0.0) PPMX(iim3, iim2, iim1, iicc, iip1, iiu, scr);
          else          PPMX(iim2, iim1, iicc, iip1, iip2, scr, iiu);
          break;
        case ReconstructionMethod::wenoz:
          if (n3 > 0.0) WENOZ(iim3, iim2, iim1, iicc, iip1, iiu, scr);
          else          WENOZ(iim2, iim1, iicc, iip1, iip2, scr, iiu);
          break;
        default:
          break;
      }

      // compute x3flux
      flx3(m,n,k,j,i) = n3*iiu;
    });
  }

  //--------------------------------------------------------------------------------------
  // Angular Fluxes

  if (angular_fluxes) {
    auto &numn = prgeo->num_neighbors;
    auto &indn = prgeo->ind_neighbors;
    auto &arcl = prgeo->arc_lengths;
    auto &solid_angles_ = prgeo->solid_angles;

    auto &na_ = na;
    auto &divfa_ = divfa;

    par_for("rflux_angular",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      divfa_(m,n,k,j,i) = 0.0;
      for (int nb=0; nb<numn.d_view(n); ++nb) {
        Real flx_edge = na_(m,n,k,j,i,nb) *
                        ((na_(m,n,k,j,i,nb) < 0.0) ?
                         i0_(m,indn.d_view(n,nb),k,j,i)/tet_c_(m,0,0,k,j,i) :
                         i0_(m,n,k,j,i)/tet_c_(m,0,0,k,j,i));
        divfa_(m,n,k,j,i) += (arcl.d_view(n,nb)*flx_edge/solid_angles_.d_view(n));
      }
    });
  }

  return TaskStatus::complete;
}

} // namespace radiation
