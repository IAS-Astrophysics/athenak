//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fluxes.cpp
//  \brief Calculate 3D fluxes for dyn_radiation

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "dyn_radiation.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenomz.hpp"
#include "reconstruct/wenoz.hpp"

namespace dyn_radiation {

//----------------------------------------------------------------------------------------
//! \fn Real ReconFaceUpwind
//! \brief Upwind face reconstruction of the primitive intensity, shared by the energy
//! and (with frequency_moments) photon-number fields so the advection speed, upwind
//! decision, and geometric normalizations are computed once per face.

KOKKOS_INLINE_FUNCTION
Real ReconFaceUpwind(const ReconstructionMethod rm, const bool near_excision,
                     const Real nsign, const Real im3, const Real im2, const Real im1,
                     const Real icc, const Real ip1, const Real ip2) {
  Real iiu = 0.0, scr;
  if (near_excision) {
    if (nsign > 0.0) iiu = im1;
    else             iiu = icc;
    return iiu;
  }
  switch (rm) {
    case ReconstructionMethod::dc:
      if (nsign > 0.0) iiu = im1;
      else             iiu = icc;
      break;
    case ReconstructionMethod::plm:
      if (nsign > 0.0) PLM(im2, im1, icc, iiu, scr);
      else             PLM(im1, icc, ip1, scr, iiu);
      break;
    case ReconstructionMethod::ppm4:
      if (nsign > 0.0) PPM4(im3, im2, im1, icc, ip1, iiu, scr);
      else             PPM4(im2, im1, icc, ip1, ip2, scr, iiu);
      break;
    case ReconstructionMethod::ppmx:
      if (nsign > 0.0) PPMX(im3, im2, im1, icc, ip1, iiu, scr);
      else             PPMX(im2, im1, icc, ip1, ip2, scr, iiu);
      break;
    case ReconstructionMethod::wenoz:
      if (nsign > 0.0) WENOZ(im3, im2, im1, icc, ip1, iiu, scr);
      else             WENOZ(im2, im1, icc, ip1, ip2, scr, iiu);
      break;
    case ReconstructionMethod::wenomz:
      if (nsign > 0.0) WENOMZ(im3, im2, im1, icc, ip1, iiu, scr);
      else             WENOMZ(im2, im1, icc, ip1, ip2, scr, iiu);
      break;
    default:
      break;
  }
  return iiu;
}

//----------------------------------------------------------------------------------------
//! \fn  void DynRadiation::CalculateFluxes
//! \brief Compute dyn_radiation fluxes

TaskStatus DynRadiation::CalculateFluxes(Driver *pdriver, int stage) {
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
  auto &sqrt_detg_c_ = sqrt_detg_c;
  bool use_adm_geometry_ = use_adm_geometry;
  bool excise = pmy_pack->pcoord->coord_data.bh_excise && excision_donor_cell;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  const bool fm_ = frequency_moments;
  const int nang_ = prgeo->nangles;

  ApplyExcisionToIntensity(i0);

  //--------------------------------------------------------------------------------------
  // i-direction

  auto &t1d1 = tet_d1_x1f;
  auto &sqrt_detg_x1f_ = sqrt_detg_x1f;
  auto &flx1 = iflx.x1f;
  par_for("rflux_x1",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie+1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    // calculate n^1 (hence determining upwinding direction)
    Real n1 = t1d1(m,0,k,j,i)*nh_c_.d_view(n,0) + t1d1(m,1,k,j,i)*nh_c_.d_view(n,1)
            + t1d1(m,2,k,j,i)*nh_c_.d_view(n,2) + t1d1(m,3,k,j,i)*nh_c_.d_view(n,3);

    // convert to primitive n_0 I
    Real iim1, iicc, iim2 = 0.0, iip1 = 0.0, iim3 = 0.0, iip2 = 0.0;
    Real norm_im1 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i-1) : tet_c_(m,0,0,k,j,i-1);
    Real norm_i   = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i  ) : tet_c_(m,0,0,k,j,i  );
    iim1 = i0_(m,n,k,j,i-1)/norm_im1;
    iicc = i0_(m,n,k,j,i  )/norm_i;
    if (recon_method_ > 0) {
      Real norm_im2 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i-2) : tet_c_(m,0,0,k,j,i-2);
      Real norm_ip1 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i+1) : tet_c_(m,0,0,k,j,i+1);
      iim2 = i0_(m,n,k,j,i-2)/norm_im2;
      iip1 = i0_(m,n,k,j,i+1)/norm_ip1;
    }
    if (recon_method_ > 1) {
      Real norm_im3 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i-3) : tet_c_(m,0,0,k,j,i-3);
      Real norm_ip2 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i+2) : tet_c_(m,0,0,k,j,i+2);
      iim3 = i0_(m,n,k,j,i-3)/norm_im3;
      iip2 = i0_(m,n,k,j,i+2)/norm_ip2;
    }

    // reconstruct primitive intensity
    bool near_excision = excise && (excision_flux_(m,k,j,i-1) ||
                                    excision_flux_(m,k,j,i));
    Real iiu = ReconFaceUpwind(recon_method_, near_excision, n1,
                               iim3, iim2, iim1, iicc, iip1, iip2);

    // compute x1flux
    Real face_norm = use_adm_geometry_ ? sqrt_detg_x1f_(m,k,j,i) : 1.0;
    flx1(m,n,k,j,i) = face_norm*n1*iiu;

    // photon-number field: reuse the advection speed, upwind decision, and norms
    if (fm_) {
      Real nnm1, nncc, nnm2 = 0.0, nnp1 = 0.0, nnm3 = 0.0, nnp2 = 0.0;
      nnm1 = i0_(m,nang_+n,k,j,i-1)/norm_im1;
      nncc = i0_(m,nang_+n,k,j,i  )/norm_i;
      if (recon_method_ > 0) {
        Real norm_im2 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i-2)
                                          : tet_c_(m,0,0,k,j,i-2);
        Real norm_ip1 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i+1)
                                          : tet_c_(m,0,0,k,j,i+1);
        nnm2 = i0_(m,nang_+n,k,j,i-2)/norm_im2;
        nnp1 = i0_(m,nang_+n,k,j,i+1)/norm_ip1;
      }
      if (recon_method_ > 1) {
        Real norm_im3 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i-3)
                                          : tet_c_(m,0,0,k,j,i-3);
        Real norm_ip2 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i+2)
                                          : tet_c_(m,0,0,k,j,i+2);
        nnm3 = i0_(m,nang_+n,k,j,i-3)/norm_im3;
        nnp2 = i0_(m,nang_+n,k,j,i+2)/norm_ip2;
      }
      Real nnu = ReconFaceUpwind(recon_method_, near_excision, n1,
                                 nnm3, nnm2, nnm1, nncc, nnp1, nnp2);
      flx1(m,nang_+n,k,j,i) = face_norm*n1*nnu;
    }
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    auto &t2d2 = tet_d2_x2f;
    auto &sqrt_detg_x2f_ = sqrt_detg_x2f;
    auto &flx2 = iflx.x2f;
    par_for("rflux_x2",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // calculate n^2 (hence determining upwinding direction)
      Real n2 = t2d2(m,0,k,j,i)*nh_c_.d_view(n,0) + t2d2(m,1,k,j,i)*nh_c_.d_view(n,1)
              + t2d2(m,2,k,j,i)*nh_c_.d_view(n,2) + t2d2(m,3,k,j,i)*nh_c_.d_view(n,3);

      // convert to primitive n_0 I
      Real iim1, iicc, iim2 = 0.0, iip1 = 0.0, iim3 = 0.0, iip2 = 0.0;
      Real norm_jm1 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j-1,i) : tet_c_(m,0,0,k,j-1,i);
      Real norm_j   = use_adm_geometry_ ? sqrt_detg_c_(m,k,j  ,i) : tet_c_(m,0,0,k,j  ,i);
      iim1 = i0_(m,n,k,j-1,i)/norm_jm1;
      iicc = i0_(m,n,k,j  ,i)/norm_j;
      if (recon_method_ > 0) {
        Real norm_jm2 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j-2,i) : tet_c_(m,0,0,k,j-2,i);
        Real norm_jp1 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j+1,i) : tet_c_(m,0,0,k,j+1,i);
        iim2 = i0_(m,n,k,j-2,i)/norm_jm2;
        iip1 = i0_(m,n,k,j+1,i)/norm_jp1;
      }
      if (recon_method_ > 1) {
        Real norm_jm3 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j-3,i) : tet_c_(m,0,0,k,j-3,i);
        Real norm_jp2 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j+2,i) : tet_c_(m,0,0,k,j+2,i);
        iim3 = i0_(m,n,k,j-3,i)/norm_jm3;
        iip2 = i0_(m,n,k,j+2,i)/norm_jp2;
      }

      // reconstruct primitive intensity
      bool near_excision = excise && (excision_flux_(m,k,j-1,i) ||
                                      excision_flux_(m,k,j,i));
      Real iiu = ReconFaceUpwind(recon_method_, near_excision, n2,
                                 iim3, iim2, iim1, iicc, iip1, iip2);

      // compute x2flux
      Real face_norm = use_adm_geometry_ ? sqrt_detg_x2f_(m,k,j,i) : 1.0;
      flx2(m,n,k,j,i) = face_norm*n2*iiu;

      // photon-number field: reuse the advection speed, upwind decision, and norms
      if (fm_) {
        Real nnm1, nncc, nnm2 = 0.0, nnp1 = 0.0, nnm3 = 0.0, nnp2 = 0.0;
        nnm1 = i0_(m,nang_+n,k,j-1,i)/norm_jm1;
        nncc = i0_(m,nang_+n,k,j  ,i)/norm_j;
        if (recon_method_ > 0) {
          Real norm_jm2 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j-2,i)
                                            : tet_c_(m,0,0,k,j-2,i);
          Real norm_jp1 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j+1,i)
                                            : tet_c_(m,0,0,k,j+1,i);
          nnm2 = i0_(m,nang_+n,k,j-2,i)/norm_jm2;
          nnp1 = i0_(m,nang_+n,k,j+1,i)/norm_jp1;
        }
        if (recon_method_ > 1) {
          Real norm_jm3 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j-3,i)
                                            : tet_c_(m,0,0,k,j-3,i);
          Real norm_jp2 = use_adm_geometry_ ? sqrt_detg_c_(m,k,j+2,i)
                                            : tet_c_(m,0,0,k,j+2,i);
          nnm3 = i0_(m,nang_+n,k,j-3,i)/norm_jm3;
          nnp2 = i0_(m,nang_+n,k,j+2,i)/norm_jp2;
        }
        Real nnu = ReconFaceUpwind(recon_method_, near_excision, n2,
                                   nnm3, nnm2, nnm1, nncc, nnp1, nnp2);
        flx2(m,nang_+n,k,j,i) = face_norm*n2*nnu;
      }
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    auto &t3d3 = tet_d3_x3f;
    auto &sqrt_detg_x3f_ = sqrt_detg_x3f;
    auto &flx3 = iflx.x3f;
    par_for("rflux_x3",DevExeSpace(),0,nmb1,0,nang1,ks,ke+1,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      // calculate n^3 (hence determining upwinding direction)
      Real n3 = t3d3(m,0,k,j,i)*nh_c_.d_view(n,0) + t3d3(m,1,k,j,i)*nh_c_.d_view(n,1)
              + t3d3(m,2,k,j,i)*nh_c_.d_view(n,2) + t3d3(m,3,k,j,i)*nh_c_.d_view(n,3);

      // convert to primitive n_0 I
      Real iim1, iicc, iim2 = 0.0, iip1 = 0.0, iim3 = 0.0, iip2 = 0.0;
      Real norm_km1 = use_adm_geometry_ ? sqrt_detg_c_(m,k-1,j,i) : tet_c_(m,0,0,k-1,j,i);
      Real norm_k   = use_adm_geometry_ ? sqrt_detg_c_(m,k  ,j,i) : tet_c_(m,0,0,k  ,j,i);
      iim1 = i0_(m,n,k-1,j,i)/norm_km1;
      iicc = i0_(m,n,k  ,j,i)/norm_k;
      if (recon_method_ > 0) {
        Real norm_km2 = use_adm_geometry_ ? sqrt_detg_c_(m,k-2,j,i) : tet_c_(m,0,0,k-2,j,i);
        Real norm_kp1 = use_adm_geometry_ ? sqrt_detg_c_(m,k+1,j,i) : tet_c_(m,0,0,k+1,j,i);
        iim2 = i0_(m,n,k-2,j,i)/norm_km2;
        iip1 = i0_(m,n,k+1,j,i)/norm_kp1;
      }
      if (recon_method_ > 1) {
        Real norm_km3 = use_adm_geometry_ ? sqrt_detg_c_(m,k-3,j,i) : tet_c_(m,0,0,k-3,j,i);
        Real norm_kp2 = use_adm_geometry_ ? sqrt_detg_c_(m,k+2,j,i) : tet_c_(m,0,0,k+2,j,i);
        iim3 = i0_(m,n,k-3,j,i)/norm_km3;
        iip2 = i0_(m,n,k+2,j,i)/norm_kp2;
      }

      // reconstruct primitive intensity
      bool near_excision = excise && (excision_flux_(m,k-1,j,i) ||
                                      excision_flux_(m,k,j,i));
      Real iiu = ReconFaceUpwind(recon_method_, near_excision, n3,
                                 iim3, iim2, iim1, iicc, iip1, iip2);

      // compute x3flux
      Real face_norm = use_adm_geometry_ ? sqrt_detg_x3f_(m,k,j,i) : 1.0;
      flx3(m,n,k,j,i) = face_norm*n3*iiu;

      // photon-number field: reuse the advection speed, upwind decision, and norms
      if (fm_) {
        Real nnm1, nncc, nnm2 = 0.0, nnp1 = 0.0, nnm3 = 0.0, nnp2 = 0.0;
        nnm1 = i0_(m,nang_+n,k-1,j,i)/norm_km1;
        nncc = i0_(m,nang_+n,k  ,j,i)/norm_k;
        if (recon_method_ > 0) {
          Real norm_km2 = use_adm_geometry_ ? sqrt_detg_c_(m,k-2,j,i)
                                            : tet_c_(m,0,0,k-2,j,i);
          Real norm_kp1 = use_adm_geometry_ ? sqrt_detg_c_(m,k+1,j,i)
                                            : tet_c_(m,0,0,k+1,j,i);
          nnm2 = i0_(m,nang_+n,k-2,j,i)/norm_km2;
          nnp1 = i0_(m,nang_+n,k+1,j,i)/norm_kp1;
        }
        if (recon_method_ > 1) {
          Real norm_km3 = use_adm_geometry_ ? sqrt_detg_c_(m,k-3,j,i)
                                            : tet_c_(m,0,0,k-3,j,i);
          Real norm_kp2 = use_adm_geometry_ ? sqrt_detg_c_(m,k+2,j,i)
                                            : tet_c_(m,0,0,k+2,j,i);
          nnm3 = i0_(m,nang_+n,k-3,j,i)/norm_km3;
          nnp2 = i0_(m,nang_+n,k+2,j,i)/norm_kp2;
        }
        Real nnu = ReconFaceUpwind(recon_method_, near_excision, n3,
                                   nnm3, nnm2, nnm1, nncc, nnp1, nnp2);
        flx3(m,nang_+n,k,j,i) = face_norm*n3*nnu;
      }
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
      Real divn = 0.0;
      for (int nb=0; nb<numn.d_view(n); ++nb) {
        const Real na_edge = na_(m,n,k,j,i,nb);
        const int n_upw = (na_edge < 0.0) ? indn.d_view(n,nb) : n;
        Real flx_edge;
        if (use_adm_geometry_) {
          // ADM angular drift is a coordinate-time angular velocity, so the
          // finite-volume unknown is U=sqrt(gamma) I itself.
          flx_edge = na_edge*i0_(m,n_upw,k,j,i);
        } else {
          // Preserve the legacy CKS normalization: the angular flux advects
          // the invariant intensity primitive U/k^0.
          flx_edge = na_edge*i0_(m,n_upw,k,j,i)/tet_c_(m,0,0,k,j,i);
        }
        divfa_(m,n,k,j,i) += (arcl.d_view(n,nb)*flx_edge/solid_angles_.d_view(n));
        if (fm_) {
          // photon number advects with the same edge speeds and upwind bins
          divn += (arcl.d_view(n,nb)*na_edge*i0_(m,nang_+n_upw,k,j,i)
                   /solid_angles_.d_view(n));
        }
      }
      if (fm_) { divfa_(m,nang_+n,k,j,i) = divn; }
    });
  }

  return TaskStatus::complete;
}

} // namespace dyn_radiation
