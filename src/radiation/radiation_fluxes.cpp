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
  // int ncells1 = indcs.nx1 + 2*(indcs.ng);

  int nang1 = prgeo->nangles - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;

  auto &i0_ = i0;
  auto &nh_c_ = nh_c;
  auto &tet_c_ = tet_c;

  //--------------------------------------------------------------------------------------
  // i-direction

  // size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*3;
  // int scr_level = 0;
  // auto &t1d1 = tet_d1_x1f;
  // auto &flx1 = iflx.x1f;
  // par_for_outer("rflux_x1",DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, nang1, ks, ke, js, je,
  // KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
  //   // assign scratch memory
  //   ScrArray1D<Real> ii(member.team_scratch(scr_level),  ncells1);
  //   ScrArray1D<Real> iil(member.team_scratch(scr_level), ncells1);
  //   ScrArray1D<Real> iir(member.team_scratch(scr_level), ncells1);
  //
  //   // compute from is-3 to ie+3 for high-order reconstruction
  //   par_for_inner(member, is-3, ie+3, [&](const int i) {
  //     // convert to primitive n_0 I
  //     ii(i) = i0_(m,n,k,j,i)/tet_c_(m,0,0,k,j,i);
  //   });
  //   // Sync all threads in the team so that scratch memory is consistent
  //   member.team_barrier();
  //
  //   // reconstruct primitive intensity
  //   switch (recon_method_) {
  //     case ReconstructionMethod::dc:
  //       par_for_inner(member, is-1, ie+1, [&](const int i) {
  //         iil(i+1) = ii(i);
  //         iir(i  ) = ii(i);
  //       });
  //       break;
  //     case ReconstructionMethod::plm:
  //       par_for_inner(member, is-1, ie+1, [&](const int i) {
  //         PLM(ii(i-1), ii(i), ii(i+1), iil(i+1), iir(i));
  //       });
  //       break;
  //     case ReconstructionMethod::ppm4:
  //       par_for_inner(member, is-1, ie+1, [&](const int i) {
  //         PPM4(ii(i-2), ii(i-1), ii(i), ii(i+1), ii(i+2), iil(i+1), iir(i));
  //       });
  //       break;
  //     case ReconstructionMethod::ppmx:
  //       par_for_inner(member, is-1, ie+1, [&](const int i) {
  //         PPMX(ii(i-2), ii(i-1), ii(i), ii(i+1), ii(i+2), iil(i+1), iir(i));
  //       });
  //       break;
  //     case ReconstructionMethod::wenoz:
  //       par_for_inner(member, is-1, ie+1, [&](const int i) {
  //         WENOZ(ii(i-2), ii(i-1), ii(i), ii(i+1), ii(i+2), iil(i+1), iir(i));
  //       });
  //       break;
  //     default:
  //       break;
  //   }
  //   // Sync all threads in the team so that scratch memory is consistent
  //   member.team_barrier();
  //
  //   Real nh_c_0 = nh_c_.d_view(n,0);
  //   Real nh_c_1 = nh_c_.d_view(n,1);
  //   Real nh_c_2 = nh_c_.d_view(n,2);
  //   Real nh_c_3 = nh_c_.d_view(n,3);
  //   par_for_inner(member, is, ie+1, [&](const int i) {
  //     // calculate n^1 (hence determining upwinding direction)
  //     Real n1 = t1d1(m,0,k,j,i)*nh_c_0 + t1d1(m,1,k,j,i)*nh_c_1
  //             + t1d1(m,2,k,j,i)*nh_c_2 + t1d1(m,3,k,j,i)*nh_c_3;
  //     // compute x1flux
  //     flx1(m,n,k,j,i) = n1 > 0.0 ? n1*iil(i) : n1*iir(i);
  //   });
  //
  // }); // endfor_outer



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
  }); // endfor


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
