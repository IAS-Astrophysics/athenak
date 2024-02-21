//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_update.cpp
//  \brief Performs update of MHD conserved variables (u0) for each stage of explicit
//  SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and partial time
//  step appropriate to stage.
//  Both the flux divergence and physical source terms are included in the update.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::AddKODissipation
//  \brief Add Kreiss-Oliger dissipation to selected prim for the black hole funnel region

void MHD::AddKODissipation() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ng = indcs.ng;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &flat = coord.is_minkowski;
  auto &spin = coord.bh_spin;
  auto &u0_ = u0;
  auto &w0_ = w0;
  auto &bcc0_ = bcc0;
  auto &w0_old_ = w0_old;
  auto eos = peos->eos_data;
  auto gm1 = eos.gamma-1;

  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &use_ko_dissipation_ = use_ko_dissipation;
  auto &sigma_ko_ = sigma_ko;
  auto &ko_dissipation_ = ko_dissipation;

  par_for("add_ko_dissipation",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    if (ko_dissipation_(m,k,j,i)) { // ko_dissipation_ flag can only be true in non-excised region
      // gas density and internal energy at center
      Real &wdn = w0_(m,IDN,k,j,i);
      Real &wvx = w0_(m,IVX,k,j,i);
      Real &wvy = w0_(m,IVY,k,j,i);
      Real &wvz = w0_(m,IVZ,k,j,i);
      Real &wen = w0_(m,IEN,k,j,i);
      Real wdn_0 = w0_old_(m,IDN,k,j,i);
      Real wen_0 = w0_old_(m,IEN,k,j,i);

      // gas density and internal energy in adjacent cells
      Real wdn_ip1, wdn_im1, wdn_ip2, wdn_im2, wdn_ip3, wdn_im3, wdn_ip4, wdn_im4;
      Real wdn_jp1, wdn_jm1, wdn_jp2, wdn_jm2, wdn_jp3, wdn_jm3, wdn_jp4, wdn_jm4;
      Real wdn_kp1, wdn_km1, wdn_kp2, wdn_km2, wdn_kp3, wdn_km3, wdn_kp4, wdn_km4;
      Real wen_ip1, wen_im1, wen_ip2, wen_im2, wen_ip3, wen_im3, wen_ip4, wen_im4;
      Real wen_jp1, wen_jm1, wen_jp2, wen_jm2, wen_jp3, wen_jm3, wen_jp4, wen_jm4;
      Real wen_kp1, wen_km1, wen_kp2, wen_km2, wen_kp3, wen_km3, wen_kp4, wen_km4;

      if (use_excise) {
        wdn_ip1 = excision_floor_(m,k,j,i+1) ? wdn_0 : w0_old_(m,IDN,k,j,i+1);
        wdn_im1 = excision_floor_(m,k,j,i-1) ? wdn_0 : w0_old_(m,IDN,k,j,i-1);
        wen_ip1 = excision_floor_(m,k,j,i+1) ? wen_0 : w0_old_(m,IEN,k,j,i+1);
        wen_im1 = excision_floor_(m,k,j,i-1) ? wen_0 : w0_old_(m,IEN,k,j,i-1);
        if (multi_d) {
          wdn_jp1 = excision_floor_(m,k,j+1,i) ? wdn_0 : w0_old_(m,IDN,k,j+1,i);
          wdn_jm1 = excision_floor_(m,k,j-1,i) ? wdn_0 : w0_old_(m,IDN,k,j-1,i);
          wen_jp1 = excision_floor_(m,k,j+1,i) ? wen_0 : w0_old_(m,IEN,k,j+1,i);
          wen_jm1 = excision_floor_(m,k,j-1,i) ? wen_0 : w0_old_(m,IEN,k,j-1,i);
        }
        if (three_d) {
          wdn_kp1 = excision_floor_(m,k+1,j,i) ? wdn_0 : w0_old_(m,IDN,k+1,j,i);
          wdn_km1 = excision_floor_(m,k-1,j,i) ? wdn_0 : w0_old_(m,IDN,k-1,j,i);
          wen_kp1 = excision_floor_(m,k+1,j,i) ? wen_0 : w0_old_(m,IEN,k+1,j,i);
          wen_km1 = excision_floor_(m,k-1,j,i) ? wen_0 : w0_old_(m,IEN,k-1,j,i);
        }
      } else {
        wdn_ip1 = w0_old_(m,IDN,k,j,i+1);
        wdn_im1 = w0_old_(m,IDN,k,j,i-1);
        wen_ip1 = w0_old_(m,IEN,k,j,i+1);
        wen_im1 = w0_old_(m,IEN,k,j,i-1);
        if (multi_d) {
          wdn_jp1 = w0_old_(m,IDN,k,j+1,i);
          wdn_jm1 = w0_old_(m,IDN,k,j-1,i);
          wen_jp1 = w0_old_(m,IEN,k,j+1,i);
          wen_jm1 = w0_old_(m,IEN,k,j-1,i);
        }
        if (three_d) {
          wdn_kp1 = w0_old_(m,IDN,k+1,j,i);
          wdn_km1 = w0_old_(m,IDN,k-1,j,i);
          wen_kp1 = w0_old_(m,IEN,k+1,j,i);
          wen_km1 = w0_old_(m,IEN,k-1,j,i);
        }
      }

      if (ng >= 2) {
        if (use_excise) {
          wdn_ip2 = excision_floor_(m,k,j,i+2) ? wdn_0 : w0_old_(m,IDN,k,j,i+2);
          wdn_im2 = excision_floor_(m,k,j,i-2) ? wdn_0 : w0_old_(m,IDN,k,j,i-2);
          wen_ip2 = excision_floor_(m,k,j,i+2) ? wen_0 : w0_old_(m,IEN,k,j,i+2);
          wen_im2 = excision_floor_(m,k,j,i-2) ? wen_0 : w0_old_(m,IEN,k,j,i-2);
          if (multi_d) {
            wdn_jp2 = excision_floor_(m,k,j+2,i) ? wdn_0 : w0_old_(m,IDN,k,j+2,i);
            wdn_jm2 = excision_floor_(m,k,j-2,i) ? wdn_0 : w0_old_(m,IDN,k,j-2,i);
            wen_jp2 = excision_floor_(m,k,j+2,i) ? wen_0 : w0_old_(m,IEN,k,j+2,i);
            wen_jm2 = excision_floor_(m,k,j-2,i) ? wen_0 : w0_old_(m,IEN,k,j-2,i);
          }
          if (three_d) {
            wdn_kp2 = excision_floor_(m,k+2,j,i) ? wdn_0 : w0_old_(m,IDN,k+2,j,i);
            wdn_km2 = excision_floor_(m,k-2,j,i) ? wdn_0 : w0_old_(m,IDN,k-2,j,i);
            wen_kp2 = excision_floor_(m,k+2,j,i) ? wen_0 : w0_old_(m,IEN,k+2,j,i);
            wen_km2 = excision_floor_(m,k-2,j,i) ? wen_0 : w0_old_(m,IEN,k-2,j,i);
          }
        } else {
          wdn_ip2 = w0_old_(m,IDN,k,j,i+2);
          wdn_im2 = w0_old_(m,IDN,k,j,i-2);
          wen_ip2 = w0_old_(m,IEN,k,j,i+2);
          wen_im2 = w0_old_(m,IEN,k,j,i-2);
          if (multi_d) {
            wdn_jp2 = w0_old_(m,IDN,k,j+2,i);
            wdn_jm2 = w0_old_(m,IDN,k,j-2,i);
            wen_jp2 = w0_old_(m,IEN,k,j+2,i);
            wen_jm2 = w0_old_(m,IEN,k,j-2,i);
          }
          if (three_d) {
            wdn_kp2 = w0_old_(m,IDN,k+2,j,i);
            wdn_km2 = w0_old_(m,IDN,k-2,j,i);
            wen_kp2 = w0_old_(m,IEN,k+2,j,i);
            wen_km2 = w0_old_(m,IEN,k-2,j,i);
          }
        }
      } // endif (ng >= 2)

      if (ng >= 3) {
        if (use_excise) {
          wdn_ip3 = excision_floor_(m,k,j,i+3) ? wdn_0 : w0_old_(m,IDN,k,j,i+3);
          wdn_im3 = excision_floor_(m,k,j,i-3) ? wdn_0 : w0_old_(m,IDN,k,j,i-3);
          wen_ip3 = excision_floor_(m,k,j,i+3) ? wen_0 : w0_old_(m,IEN,k,j,i+3);
          wen_im3 = excision_floor_(m,k,j,i-3) ? wen_0 : w0_old_(m,IEN,k,j,i-3);
          if (multi_d) {
            wdn_jp3 = excision_floor_(m,k,j+3,i) ? wdn_0 : w0_old_(m,IDN,k,j+3,i);
            wdn_jm3 = excision_floor_(m,k,j-3,i) ? wdn_0 : w0_old_(m,IDN,k,j-3,i);
            wen_jp3 = excision_floor_(m,k,j+3,i) ? wen_0 : w0_old_(m,IEN,k,j+3,i);
            wen_jm3 = excision_floor_(m,k,j-3,i) ? wen_0 : w0_old_(m,IEN,k,j-3,i);
          }
          if (three_d) {
            wdn_kp3 = excision_floor_(m,k+3,j,i) ? wdn_0 : w0_old_(m,IDN,k+3,j,i);
            wdn_km3 = excision_floor_(m,k-3,j,i) ? wdn_0 : w0_old_(m,IDN,k-3,j,i);
            wen_kp3 = excision_floor_(m,k+3,j,i) ? wen_0 : w0_old_(m,IEN,k+3,j,i);
            wen_km3 = excision_floor_(m,k-3,j,i) ? wen_0 : w0_old_(m,IEN,k-3,j,i);
          }
        } else {
          wdn_ip3 = w0_old_(m,IDN,k,j,i+3);
          wdn_im3 = w0_old_(m,IDN,k,j,i-3);
          wen_ip3 = w0_old_(m,IEN,k,j,i+3);
          wen_im3 = w0_old_(m,IEN,k,j,i-3);
          if (multi_d) {
            wdn_jp3 = w0_old_(m,IDN,k,j+3,i);
            wdn_jm3 = w0_old_(m,IDN,k,j-3,i);
            wen_jp3 = w0_old_(m,IEN,k,j+3,i);
            wen_jm3 = w0_old_(m,IEN,k,j-3,i);
          }
          if (three_d) {
            wdn_kp3 = w0_old_(m,IDN,k+3,j,i);
            wdn_km3 = w0_old_(m,IDN,k-3,j,i);
            wen_kp3 = w0_old_(m,IEN,k+3,j,i);
            wen_km3 = w0_old_(m,IEN,k-3,j,i);
          }
        }
      } // endif (ng >= 3)

      if (ng >= 4) {
        if (use_excise) {
          wdn_ip4 = excision_floor_(m,k,j,i+4) ? wdn_0 : w0_old_(m,IDN,k,j,i+4);
          wdn_im4 = excision_floor_(m,k,j,i-4) ? wdn_0 : w0_old_(m,IDN,k,j,i-4);
          wen_ip4 = excision_floor_(m,k,j,i+4) ? wen_0 : w0_old_(m,IEN,k,j,i+4);
          wen_im4 = excision_floor_(m,k,j,i-4) ? wen_0 : w0_old_(m,IEN,k,j,i-4);
          if (multi_d) {
            wdn_jp4 = excision_floor_(m,k,j+4,i) ? wdn_0 : w0_old_(m,IDN,k,j+4,i);
            wdn_jm4 = excision_floor_(m,k,j-4,i) ? wdn_0 : w0_old_(m,IDN,k,j-4,i);
            wen_jp4 = excision_floor_(m,k,j+4,i) ? wen_0 : w0_old_(m,IEN,k,j+4,i);
            wen_jm4 = excision_floor_(m,k,j-4,i) ? wen_0 : w0_old_(m,IEN,k,j-4,i);
          }
          if (three_d) {
            wdn_kp4 = excision_floor_(m,k+4,j,i) ? wdn_0 : w0_old_(m,IDN,k+4,j,i);
            wdn_km4 = excision_floor_(m,k-4,j,i) ? wdn_0 : w0_old_(m,IDN,k-4,j,i);
            wen_kp4 = excision_floor_(m,k+4,j,i) ? wen_0 : w0_old_(m,IEN,k+4,j,i);
            wen_km4 = excision_floor_(m,k-4,j,i) ? wen_0 : w0_old_(m,IEN,k-4,j,i);
          }
        } else {
          wdn_ip4 = w0_old_(m,IDN,k,j,i+4);
          wdn_im4 = w0_old_(m,IDN,k,j,i-4);
          wen_ip4 = w0_old_(m,IEN,k,j,i+4);
          wen_im4 = w0_old_(m,IEN,k,j,i-4);
          if (multi_d) {
            wdn_jp4 = w0_old_(m,IDN,k,j+4,i);
            wdn_jm4 = w0_old_(m,IDN,k,j-4,i);
            wen_jp4 = w0_old_(m,IEN,k,j+4,i);
            wen_jm4 = w0_old_(m,IEN,k,j-4,i);
          }
          if (three_d) {
            wdn_kp4 = w0_old_(m,IDN,k+4,j,i);
            wdn_km4 = w0_old_(m,IDN,k-4,j,i);
            wen_kp4 = w0_old_(m,IEN,k+4,j,i);
            wen_km4 = w0_old_(m,IEN,k-4,j,i);
          }
        }
      } // endif (ng >= 4)

      // compute dissipation terms
      Real coeff, del_wdnx, del_wdny, del_wdnz, del_wenx, del_weny, del_wenz;
      // if (ng == 3) {
      Real c3 = 1.0; Real c2 = -6.0; Real c1 = 15.0; Real c0 = -20.0;
      del_wdnx = c3*wdn_ip3 + c2*wdn_ip2 + c1*wdn_ip1 + c0*wdn_0 + c1*wdn_im1 + c2*wdn_im2 + c3*wdn_im3;
      del_wenx = c3*wen_ip3 + c2*wen_ip2 + c1*wen_ip1 + c0*wen_0 + c1*wen_im1 + c2*wen_im2 + c3*wen_im3;
      if (multi_d) {
        del_wdny = c3*wdn_jp3 + c2*wdn_jp2 + c1*wdn_jp1 + c0*wdn_0 + c1*wdn_jm1 + c2*wdn_jm2 + c3*wdn_jm3;
        del_weny = c3*wen_jp3 + c2*wen_jp2 + c1*wen_jp1 + c0*wen_0 + c1*wen_jm1 + c2*wen_jm2 + c3*wen_jm3;
      }
      if (three_d) {
        del_wdnz = c3*wdn_kp3 + c2*wdn_kp2 + c1*wdn_kp1 + c0*wdn_0 + c1*wdn_km1 + c2*wdn_km2 + c3*wdn_km3;
        del_wenz = c3*wen_kp3 + c2*wen_kp2 + c1*wen_kp1 + c0*wen_0 + c1*wen_km1 + c2*wen_km2 + c3*wen_km3;
      }
      coeff = sigma_ko_/64;
      // }

      // add dissipation terms
      wdn += coeff * del_wdnx/size.d_view(m).dx1;
      wen += coeff * del_wenx/size.d_view(m).dx1;
      if (multi_d) {
        wdn += coeff * del_wdny/size.d_view(m).dx2;
        wen += coeff * del_weny/size.d_view(m).dx2;
      }
      if (three_d) {
        wdn += coeff * del_wdnz/size.d_view(m).dx3;
        wen += coeff * del_wenz/size.d_view(m).dx3;
      }

      // check floors
      wdn = fmax(wdn, eos.dfloor);
      Real lg_sfloor_local = log10(eos.sfloor1) + (log10(wdn)-log10(eos.rho1)) * (log10(eos.sfloor2)-log10(eos.sfloor1))/(log10(eos.rho2)-log10(eos.rho1));
      Real sfloor_local = pow(10.0, lg_sfloor_local);
      Real sfloor = fmax(eos.sfloor, sfloor_local);
      Real pfloor = fmax(eos.pfloor, sfloor*pow(wdn, eos.gamma));
      Real efloor = pfloor/gm1;
      wen = fmax(wen, efloor);

      // extract components of metric
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      // Reset conserved variables
      MHDPrim1D w_in;
      w_in.d  = wdn;
      w_in.vx = wvx;
      w_in.vy = wvy;
      w_in.vz = wvz;
      w_in.e  = wen;
      w_in.bx = bcc0_(m,IBX,k,j,i);
      w_in.by = bcc0_(m,IBY,k,j,i);
      w_in.bz = bcc0_(m,IBZ,k,j,i);

      HydCons1D u_out;
      SingleP2C_IdealGRMHD(glower, gupper, w_in, eos.gamma, u_out);
      u0_(m,IDN,k,j,i) = u_out.d;
      u0_(m,IM1,k,j,i) = u_out.mx;
      u0_(m,IM2,k,j,i) = u_out.my;
      u0_(m,IM3,k,j,i) = u_out.mz;
      u0_(m,IEN,k,j,i) = u_out.e;
    } // endif ko_dissipation_(m,k,j,i)
  });

  return;
} // end void MHD::EntropyReset

} // namespace mhd
