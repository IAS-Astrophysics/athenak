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
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
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
    auto &size  = pmy_pack->pmb->mb_size;
    auto &coord = pmy_pack->pcoord->coord_data;
    bool &flat = coord.is_minkowski;
    Real &spin = coord.bh_spin;
    auto &tetcov_c_ = tetcov_c;
    auto &norm_to_tet_ = norm_to_tet;

    bool &is_hydro_enabled_ = is_hydro_enabled;
    bool &is_mhd_enabled_ = is_mhd_enabled;

    Real &arad_ = arad;
    auto &divfa_ = divfa;
    auto &nu_tet = freq_grid;
    auto &nnu_coeff_ = nnu_coeff;

    int order_freq_fluxes = 0;
    if (!angular_fluxes) Kokkos::deep_copy(divfa_, 0);

    par_for("rflux_freq",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int iang, int k, int j, int i) {

      // extract spatial position
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      // compute metric and inverse
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
      Real alpha = sqrt(-1.0/gupper[0][0]);

      // normal velocities
      Real wvx=0, wvy=0, wvz=0;
      if (is_hydro_enabled_) {
        wvx = pmy_pack->phydro->w0(m,IVX,k,j,i);
        wvy = pmy_pack->phydro->w0(m,IVY,k,j,i);
        wvz = pmy_pack->phydro->w0(m,IVZ,k,j,i);
      } else if (is_mhd_enabled_) {
        wvx = pmy_pack->pmhd->w0(m,IVX,k,j,i);
        wvy = pmy_pack->pmhd->w0(m,IVY,k,j,i);
        wvz = pmy_pack->pmhd->w0(m,IVZ,k,j,i);
      }

      // Lorentz factor
      Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
             + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz
             + glower[3][3]*wvz*wvz;
      Real gamma = sqrt(1.0 + q);
      Real u0 = gamma/alpha;

      // compute fluid velocity in tetrad frame
      Real u_tet[4];
      u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gamma + norm_to_tet_(m,0,1,k,j,i)*wvx +
                  norm_to_tet_(m,0,2,k,j,i)*wvy   + norm_to_tet_(m,0,3,k,j,i)*wvz);
      u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gamma + norm_to_tet_(m,1,1,k,j,i)*wvx +
                  norm_to_tet_(m,1,2,k,j,i)*wvy   + norm_to_tet_(m,1,3,k,j,i)*wvz);
      u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gamma + norm_to_tet_(m,2,1,k,j,i)*wvx +
                  norm_to_tet_(m,2,2,k,j,i)*wvy   + norm_to_tet_(m,2,3,k,j,i)*wvz);
      u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gamma + norm_to_tet_(m,3,1,k,j,i)*wvx +
                  norm_to_tet_(m,3,2,k,j,i)*wvy   + norm_to_tet_(m,3,3,k,j,i)*wvz);

      Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1)
                  - u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));

      // coordinate component n^0
      Real n0 = tet_c_(m,0,0,k,j,i);

      // coordinate-frame normal components
      Real n_0 = tetcov_c_(m,0,0,k,j,i)*nh_c_.d_view(iang,0) + tetcov_c_(m,1,0,k,j,i)*nh_c_.d_view(iang,1)
               + tetcov_c_(m,2,0,k,j,i)*nh_c_.d_view(iang,2) + tetcov_c_(m,3,0,k,j,i)*nh_c_.d_view(iang,3);

      // estimate inu at nu_tet[nfreq-1] assuming blackbody tail
      Real &nu_e = nu_tet(nfreq1);
      int ne = getFreqAngIndex(nfreq1, iang, nang);
      Real ir_cm_star_e = SQR(SQR(n0_cm))*i0_(m,ne,k,j,i)/(n0*n_0);
      Real teff = GetEffTemperature(ir_cm_star_e, n0_cm*nu_e, arad_);
      Real inu_e = fmax(BBSpectrum(n0_cm*nu_e, teff, arad_)/(4*M_PI)/SQR(SQR(n0_cm)), 0.0);

      // note: flux through nu_tet[0] is simply 0
      // compute flux at frequency faces from ifr=1 to ifr=nfreq-1
      for (int ifr=1; ifr <= nfreq1; ++ifr) {
        Real &nu_fm1 = nu_tet(ifr-1);
        Real &nu_f   = nu_tet(ifr);
        int nf = getFreqAngIndex(ifr, iang, nang);
        int nfm1 = getFreqAngIndex(ifr-1, iang, nang);
        Real inu_fm1h = i0_(m,nfm1,k,j,i)/(n0*n_0)/(nu_f-nu_fm1);
        Real nu_fm1h = (nu_f + nu_fm1) / 2;

        // zeroth-order upwind (default)
        // left state
        Real inu_l = inu_fm1h; // default: order=0
        // right state
        Real inu_r = inu_e; // right boundary
        if (ifr < nfreq1) {
          // ifr <= nfreq - 2
          Real &nu_fp1 = nu_tet(ifr+1);
          Real inu_fp1h = i0_(m,nf,k,j,i)/(n0*n_0)/(nu_fp1-nu_f);
          inu_r = inu_fp1h; // defaul: order=0
        } // endif (ifr < nfreq1)

        // second-order upwind (PLM, optional)
        if ((order_freq_fluxes == 2) && (ifr < nfreq1)) { // 1 <= ifr <= nfreq-2
          Real &nu_fp1 = nu_tet(ifr+1);
          Real inu_fp1h = i0_(m,nf,k,j,i)/(n0*n_0)/(nu_fp1-nu_f);
          Real nu_fp1h = (nu_fp1 + nu_f) / 2;

          // boundaries
          int boundary = 0;
          if (ifr == 1) boundary = 1; // left boundary
          if (ifr == nfreq1-1) boundary = 2; // right boundary

          // left extension
          Real nu_fm3h = -1, inu_fm3h = -1;
          if (ifr > 1) {
            Real &nu_fm2 = nu_tet(ifr-2);
            Real nu_fm3h = (nu_fm1 + nu_fm2) / 2;
            int nfm2 = getFreqAngIndex(ifr-2, iang, nang);
            inu_fm3h = i0_(m,nfm2,k,j,i)/(n0*n_0)/(nu_fm1-nu_fm2);
            nu_fm3h = (nu_fm1 + nu_fm2) / 2;
          } // endif (ifr > 1)

          // right extension
          Real nu_fp3h = -1, inu_fp3h = -1;
          if (ifr < nfreq1-1) {
            Real &nu_fp2 = nu_tet(ifr+2);
            Real nu_fp3h = (nu_fp2 + nu_fp1) / 2;
            int nfp1 = getFreqAngIndex(ifr+1, iang, nang);
            inu_fp3h = i0_(m,nfp1,k,j,i)/(n0*n_0)/(nu_fp2-nu_fp1);
            nu_fp3h = (nu_fp2 + nu_fp1) / 2;
          } // endif (ifr < nfreq1-1)

          // compute slopes for left and right states
          Real kl = GetMultiFreqRadSlope(nu_fm3h, nu_fm1, nu_fm1h, nu_f, nu_fp1h,
                                         inu_fm3h, 0, inu_fm1h, inu_e, inu_fp1h,
                                         2, 2, boundary); // PLM + van Leer limiter

          Real kr = GetMultiFreqRadSlope(nu_fm1h, nu_f, nu_fp1h, nu_fp1, nu_fp3h,
                                         inu_fm1h, 0, inu_fp1h, inu_e, inu_fp3h,
                                         2, 2, boundary); // PLM + van Leer limiter

          // compute left and right states
          Real inu_l_plm = inu_fm1h + kl*(nu_f-nu_fm1h);
          Real inu_r_plm = inu_fp1h + kr*(nu_f-nu_fp1h);
          if (inu_l_plm > 0) inu_l = inu_l_plm;
          if (inu_r_plm > 0) inu_r = inu_r_plm;
        } // endif ((order_freq_fluxes == 2) && (ifr < nfreq1))

        // direction
        Real nnu_f = -nu_f * nnu_coeff_(m,iang,k,j,i);

        // upwind flux
        Real flx_f = (nnu_f > 0) ? inu_l : inu_r;
        flx_f *= nnu_f*n_0; // n^nu n_t I

        // divergence
        divfa_(m,nfm1,k,j,i) += -flx_f;
        divfa_(m,nf,k,j,i)   +=  flx_f;

      } // endfor 1 <= ifr <= nfreq-1

    }); // end par_for

  } // endif freq_fluxes

  return TaskStatus::complete;
}

} // namespace radiation
