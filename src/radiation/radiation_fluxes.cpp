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
#include "radiation_multi_freq.hpp"
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
  int &nfrq = nfreq;
  int &nang  = prgeo->nangles;
  int nang1 = nang - 1;
  int nfreq1 = nfrq - 1;
  int nfr_ang1 = nfrq*nang - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  const auto &recon_method_ = recon_method;

  auto &i0_ = i0;
  auto &nh_c_ = nh_c;
  auto &tet_c_ = tet_c;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto &t1d1 = tet_d1_x1f;
  auto &flx1 = iflx.x1f;
  par_for("rflux_x1",DevExeSpace(),0,nmb1,0,nfr_ang1,ks,ke,js,je,is,ie+1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    // compute frequency and angle indices
    int ifr, iang;
    getFreqAngIndices(n, nang, ifr, iang);

    // calculate n^1 (hence determining upwinding direction)
    Real n1 = t1d1(m,0,k,j,i)*nh_c_.d_view(iang,0) + t1d1(m,1,k,j,i)*nh_c_.d_view(iang,1)
            + t1d1(m,2,k,j,i)*nh_c_.d_view(iang,2) + t1d1(m,3,k,j,i)*nh_c_.d_view(iang,3);

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
    par_for("rflux_x2",DevExeSpace(),0,nmb1,0,nfr_ang1,ks,ke,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // compute frequency and angle indices
      int ifr, iang;
      getFreqAngIndices(n, nang, ifr, iang);

      // calculate n^2 (hence determining upwinding direction)
      Real n2 = t2d2(m,0,k,j,i)*nh_c_.d_view(iang,0) + t2d2(m,1,k,j,i)*nh_c_.d_view(iang,1)
              + t2d2(m,2,k,j,i)*nh_c_.d_view(iang,2) + t2d2(m,3,k,j,i)*nh_c_.d_view(iang,3);

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
    par_for("rflux_x3",DevExeSpace(),0,nmb1,0,nfr_ang1,ks,ke+1,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // compute frequency and angle indices
      int ifr, iang;
      getFreqAngIndices(n, nang, ifr, iang);

      // calculate n^3 (hence determining upwinding direction)
      Real n3 = t3d3(m,0,k,j,i)*nh_c_.d_view(iang,0) + t3d3(m,1,k,j,i)*nh_c_.d_view(iang,1)
              + t3d3(m,2,k,j,i)*nh_c_.d_view(iang,2) + t3d3(m,3,k,j,i)*nh_c_.d_view(iang,3);

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

    par_for("rflux_angular",DevExeSpace(),0,nmb1,0,nfr_ang1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // compute frequency and angle indices
      int ifr, iang;
      getFreqAngIndices(n, nang, ifr, iang);
      // compute angular fluxes
      Real divfa_tmp = 0.0;
      Real tet_c_tmp = tet_c_(m,0,0,k,j,i);
      Real solid_angles_tmp = solid_angles_.d_view(iang);
      Real i0_n = i0_(m,n,k,j,i);
      for (int nb=0; nb<numn.d_view(iang); ++nb) {
        Real na_tmp = na_(m,iang,k,j,i,nb);
        Real flx_edge = na_tmp/tet_c_tmp;
        if (na_tmp < 0.0) {
          int ifr_ang = getFreqAngIndex(ifr, indn.d_view(iang,nb), nang);
          flx_edge *= i0_(m,ifr_ang,k,j,i);
        } else {
          flx_edge *= i0_n;
        }
        divfa_tmp += (arcl.d_view(iang,nb)*flx_edge/solid_angles_tmp);
      }
      divfa_(m,n,k,j,i) = divfa_tmp;
    });
  }

  //--------------------------------------------------------------------------------------
  // Frequency Fluxes
  if (freq_fluxes) {
    auto &divfa_ = divfa;
    auto &nu_tet = freq_grid;
    auto &nnu_coeff_ = nnu_coeff;

    if (!angular_fluxes) Kokkos::deep_copy(divfa_, 0);

    par_for("rflux_freq",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int iang, int k, int j, int i) {

      Real n0 = tet_c_(m,0,0,k,j,i);

      for (int ifr=1; ifr<=nfreq1; ++ifr) {
        Real &nu_fm1 = nu_tet(ifr-1);
        Real &nu_f   = nu_tet(ifr);
        int nl = getFreqAngIndex(ifr-1, iang, nang);
        int n  = getFreqAngIndex(ifr, iang, nang);
        Real i0_l = i0_(m,nl,k,j,i)/n0;
        Real i0_r = i0_(m,n,k,j,i) /n0;

        // direction
        Real nnu_f = -nu_f * nnu_coeff_(m,iang,k,j,i);

        // upwind flux
        Real i0_u = (nnu_f > 0) ? i0_l : i0_r;
        Real flx_f = nnu_f*i0_u/(nu_f-nu_fm1);

        // divergence
        divfa_(m,nl,k,j,i) += -flx_f;
        divfa_(m,n,k,j,i)  += flx_f;
      } // endfor ifr

    });

  } // endfor freq_fluxes

  return TaskStatus::complete;
}

} // namespace radiation
