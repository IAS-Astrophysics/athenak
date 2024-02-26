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

  // auto &use_ko_dissipation_ = use_ko_dissipation;
  auto &sigma_ko_ = sigma_ko;
  auto &ko_dissipation_ = ko_dissipation;
  auto &w_ko = w_kokernel;

  par_for("add_ko_dissipation",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    if (ko_dissipation_(m,k,j,i)) { // ko_dissipation_ flag can only be true in non-excised region
      // primitive for being dissipated
      Real &wdn = w0_(m,IDN,k,j,i);
      Real &wvx = w0_(m,IVX,k,j,i);
      Real &wvy = w0_(m,IVY,k,j,i);
      Real &wvz = w0_(m,IVZ,k,j,i);
      Real &wen = w0_(m,IEN,k,j,i);
      Real tgas = wdn/wen;

      // primitives at center and in adjacent cells
      int ic=4; int jc=4; int kc=4;
      int num_i = ng;
      int num_j = multi_d ? ng : 0;
      int num_k = three_d ? ng : 0;
      for (int k_add=-num_k; k_add <= num_k; ++k_add) {
        for (int j_add=-num_j; j_add <= num_j; ++j_add) {
          for (int i_add=-num_i; i_add <= num_i; ++i_add) {
            w_ko(IDN,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IDN,k+k_add,j+j_add,i+i_add);
            w_ko(IVX,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IVX,k+k_add,j+j_add,i+i_add);
            w_ko(IVY,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IVY,k+k_add,j+j_add,i+i_add);
            w_ko(IVZ,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IVZ,k+k_add,j+j_add,i+i_add);
            w_ko(IEN,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IEN,k+k_add,j+j_add,i+i_add);
            if (use_excise) {
              if (excision_floor_(m,k+k_add,j+j_add,i+i_add)) {
                w_ko(IDN,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IDN,k,j,i);
                w_ko(IVX,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IVX,k,j,i);
                w_ko(IVY,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IVY,k,j,i);
                w_ko(IVZ,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IVZ,k,j,i);
                w_ko(IEN,kc+k_add,jc+j_add,ic+i_add) = w0_old_(m,IEN,k,j,i);
              }
            } // endif use_excise
          } // endfor i_add
        } // endfor j_add
      } // endfor k_add


      // compute dissipation terms
      Real del_wdn[3], del_wvx[3], del_wvy[3], del_wvz[3], del_wen[3], del_tgas[3];
      Real coeff0 = sigma_ko_/64.0;
      Real coeffs[4];
      coeffs[3] = 1.0; coeffs[2] = -6.0; coeffs[1] = 15.0; coeffs[0] = -20.0;
      int n_max = 1;
      if (multi_d) n_max = 2;
      if (three_d) n_max = 3;
      for (int n=0; n < n_max; ++n) {
        del_wdn[n]  = coeffs[0]*w_ko(IDN,kc,jc,ic);
        del_wvx[n]  = coeffs[0]*w_ko(IVX,kc,jc,ic);
        del_wvy[n]  = coeffs[0]*w_ko(IVY,kc,jc,ic);
        del_wvz[n]  = coeffs[0]*w_ko(IVZ,kc,jc,ic);
        del_wen[n]  = coeffs[0]*w_ko(IEN,kc,jc,ic);
        del_tgas[n] = coeffs[0]*w_ko(IEN,kc,jc,ic)/w_ko(IDN,kc,jc,ic);
        for (int l=1; l<=3; ++l) {
          int kl = (n==2) ? kc-l : kc; int kr = (n==2) ? kc+l : kc;
          int jl = (n==1) ? jc-l : jc; int jr = (n==1) ? jc+l : jc;
          int il = (n==0) ? ic-l : ic; int ir = (n==0) ? ic+l : ic;
          del_wdn[n]  += coeffs[l] * (w_ko(IDN,kr,jr,ir) + w_ko(IDN,kl,jl,il));
          del_wvx[n]  += coeffs[l] * (w_ko(IVX,kr,jr,ir) + w_ko(IVX,kl,jl,il));
          del_wvy[n]  += coeffs[l] * (w_ko(IVY,kr,jr,ir) + w_ko(IVY,kl,jl,il));
          del_wvz[n]  += coeffs[l] * (w_ko(IVZ,kr,jr,ir) + w_ko(IVZ,kl,jl,il));
          del_wen[n]  += coeffs[l] * (w_ko(IEN,kr,jr,ir) + w_ko(IEN,kl,jl,il));
          del_tgas[n] += coeffs[l] * (w_ko(IEN,kr,jr,ir)/w_ko(IDN,kr,jr,ir) + w_ko(IEN,kl,jl,il)/w_ko(IDN,kl,jl,il));
        } // endfor l
      } // endfor n

      // add dissipation terms
      // wdn  += coeff0 * del_wdn[0]/size.d_view(m).dx1;
      // wvx  += coeff0 * del_wvx[0]/size.d_view(m).dx1;
      // wvy  += coeff0 * del_wvy[0]/size.d_view(m).dx1;
      // wvz  += coeff0 * del_wvz[0]/size.d_view(m).dx1;
      wen  += coeff0 * del_wen[0]/size.d_view(m).dx1;
      tgas += coeff0 * del_tgas[0]/size.d_view(m).dx1;

      if (multi_d) {
        // wdn  += coeff0 * del_wdn[1]/size.d_view(m).dx2;
        // wvx  += coeff0 * del_wvx[1]/size.d_view(m).dx2;
        // wvy  += coeff0 * del_wvy[1]/size.d_view(m).dx2;
        // wvz  += coeff0 * del_wvz[1]/size.d_view(m).dx2;
        wen  += coeff0 * del_wen[1]/size.d_view(m).dx2;
        tgas += coeff0 * del_tgas[1]/size.d_view(m).dx2;
      }

      if (three_d) {
        // wdn  += coeff0 * del_wdn[2]/size.d_view(m).dx3;
        // wvx  += coeff0 * del_wvx[2]/size.d_view(m).dx3;
        // wvy  += coeff0 * del_wvy[2]/size.d_view(m).dx3;
        // wvz  += coeff0 * del_wvz[2]/size.d_view(m).dx3;
        wen  += coeff0 * del_wen[2]/size.d_view(m).dx3;
        tgas += coeff0 * del_tgas[2]/size.d_view(m).dx3;
      }

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

      // check density and pressure
      // if (wdn < eos.dfloor) wdn = w0_old_(m,IDN,k,j,i);
      Real lg_sfloor_local = log10(eos.sfloor1) + (log10(wdn)-log10(eos.rho1)) * (log10(eos.sfloor2)-log10(eos.sfloor1))/(log10(eos.rho2)-log10(eos.rho1));
      Real sfloor_local = pow(10.0, lg_sfloor_local);
      Real sfloor = fmax(eos.sfloor, sfloor_local);
      Real pfloor = fmax(eos.pfloor, sfloor*pow(wdn, eos.gamma));
      Real efloor = pfloor/gm1;
      wen = fmin(wen, wdn*tgas);
      if (wen < efloor) wen = w0_old_(m,IEN,k,j,i);

      // apply velocity ceiling if necessary
      // Real tmp = glower[1][1]*SQR(wvx)
      //          + glower[2][2]*SQR(wvy)
      //          + glower[3][3]*SQR(wvz)
      //          + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
      //          + 2.0*glower[2][3]*wvy*wvz;
      // Real lor = sqrt(1.0+tmp);
      // if (lor > eos.gamma_max) {
      //   // Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
      //   // wvx *= factor;
      //   // wvy *= factor;
      //   // wvz *= factor;
      //   wvx = w0_old_(m,IVX,k,j,i);
      //   wvy = w0_old_(m,IVY,k,j,i);
      //   wvz = w0_old_(m,IVZ,k,j,i);
      // }

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
