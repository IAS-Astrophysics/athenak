//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fofc.cpp
//! \brief Implements functions for first-order flux correction (FOFC) algorithm.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/rsolvers/llf_mhd_singlestate.hpp"
#include "reconstruct/plm.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::FOFC
//! \brief Implements first-order flux-correction (FOFC) algorithm for MHD.  First an
//! estimate of the updated conserved variables is made. This estimate is then used to
//! flag any cell where floors will be required during the conversion to primitives. Then
//! the fluxes on the faces of flagged cells are replaced with first-order LLF fluxes.
//! Often this is enough to prevent floors from being needed.  The FOFC infrastructure is
//! also exploited for BH excision.  If a cell is about the horizon, FOFC is automatically
//! triggered (without estimating updated conserved variables).

void MHD::FOFC(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb = pmy_pack->nmb_thispack;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &size = pmy_pack->pmb->mb_size;

  auto &bcc0_ = bcc0;
  auto &e3x1_ = e3x1;
  auto &e2x1_ = e2x1;
  auto &e1x2_ = e1x2;
  auto &e3x2_ = e3x2;
  auto &e2x3_ = e2x3;
  auto &e1x3_ = e1x3;

  if (use_fofc || use_sofc) {
    Real &gam0 = pdriver->gam0[stage-1];
    Real &gam1 = pdriver->gam1[stage-1];
    Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

    int &nmhd_ = nmhd;
    auto &u0_ = u0;
    auto &u1_ = u1;
    auto &utest_ = utest;
    auto &bcctest_ = bcctest;
    auto &b1_ = b1;

    // Index bounds
    int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) { jl = js-1, ju = je+1; }
    if (three_d) { kl = ks-1, ku = ke+1; }

    // Estimate updated conserved variables and cell-centered fields
    par_for("FOFC-newu", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size.d_view(m).dx1;
      Real dtodx2 = beta_dt/size.d_view(m).dx2;
      Real dtodx3 = beta_dt/size.d_view(m).dx3;

      // Estimate conserved variables
      for (int n=0; n<nmhd_; ++n) {
        Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
        if (multi_d) {
          divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
        }
        if (three_d) {
          divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
        }
        utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
      }

      // Estimate updated cell-centered fields
      Real b1old = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      Real b2old = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      Real b3old = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));

      bcctest_(m,IBX,k,j,i) = gam0*bcc0_(m,IBX,k,j,i) + gam1*b1old;
      bcctest_(m,IBY,k,j,i) = gam0*bcc0_(m,IBY,k,j,i) + gam1*b2old;
      bcctest_(m,IBZ,k,j,i) = gam0*bcc0_(m,IBZ,k,j,i) + gam1*b3old;

      bcctest_(m,IBY,k,j,i) += dtodx1*(e3x1_(m,k,j,i+1) - e3x1_(m,k,j,i));
      bcctest_(m,IBZ,k,j,i) -= dtodx1*(e2x1_(m,k,j,i+1) - e2x1_(m,k,j,i));
      if (multi_d) {
        bcctest_(m,IBX,k,j,i) -= dtodx2*(e3x2_(m,k,j+1,i) - e3x2_(m,k,j,i));
        bcctest_(m,IBZ,k,j,i) += dtodx2*(e1x2_(m,k,j+1,i) - e1x2_(m,k,j,i));
      }
      if (three_d) {
        bcctest_(m,IBX,k,j,i) += dtodx3*(e2x3_(m,k+1,j,i) - e2x3_(m,k,j,i));
        bcctest_(m,IBY,k,j,i) -= dtodx3*(e1x3_(m,k+1,j,i) - e1x3_(m,k,j,i));
      }
    });

    // Test whether conversion to primitives requires floors
    // Note b0 and w0 passed to function, but not used/changed.
    peos->ConsToPrim(utest_, b0, w0, bcctest_, true, il, iu, jl, ju, kl, ku);
  }

  auto &coord = pmy_pack->pcoord->coord_data;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = peos->eos_data;
  auto &use_fofc_ = use_fofc;
  auto &use_sofc_ = use_sofc;
  auto fofc_ = fofc;
  auto &use_excise_ = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &w0_ = w0;
  auto &b0_ = b0;

  // Index bounds
  int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-1, ju = je+1; }
  if (three_d) { kl = ks-1, ku = ke+1; }

  // FOFC diagnostic: set conserved passive scalar to rho*s (s=1) at cells where FOFC triggered
  if ((use_fofc || use_sofc) && nscalars >= 1) {
    int &nmhd_ = nmhd;
    auto &u0_ = u0;
    auto fofc_scalar_ = fofc;
    par_for("FOFC-scalar", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (fofc_scalar_(m, k, j, i)) {
        u0_(m, nmhd_, k, j, i) = u0_(m, IDN, k, j, i);  // rho*s with s=1
      }
    });
  }

  // Replace fluxes with first-order or second-order (PLM) LLF fluxes at i,j,k faces
  // for any cell where FOFC/SOFC and/or excision is used (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_ || use_sofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Check for GR + excision
    bool fofc_excision = false;
    if (is_gr) {
      if (use_excise_) { fofc_excision = excision_flux_(m,k,j,i); }
    }

    // Apply FOFC (first-order)
    if (use_fofc_ && (fofc_flag || fofc_excision)) {
      // load W_{i-1} state
      MHDPrim1D wim1;
      wim1.d  = w0_(m,IDN,k,j,i-1);
      wim1.vx = w0_(m,IVX,k,j,i-1);
      wim1.vy = w0_(m,IVY,k,j,i-1);
      wim1.vz = w0_(m,IVZ,k,j,i-1);
      if (eos.is_ideal) {wim1.e  = w0_(m,IEN,k,j,i-1);}
      wim1.by = bcc0_(m,IBY,k,j,i-1);
      wim1.bz = bcc0_(m,IBZ,k,j,i-1);

      // load W_{i} state
      MHDPrim1D wi;
      wi.d  = w0_(m,IDN,k,j,i);
      wi.vx = w0_(m,IVX,k,j,i);
      wi.vy = w0_(m,IVY,k,j,i);
      wi.vz = w0_(m,IVZ,k,j,i);
      if (eos.is_ideal) {wi.e = w0_(m,IEN,k,j,i);}
      wi.by = bcc0_(m,IBY,k,j,i);
      wi.bz = bcc0_(m,IBZ,k,j,i);

      // compute new 1st-order LLF flux at i-face
      {
        Real bxi = b0_.x1f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = LeftEdgeX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wim1, wi, bxi, x1v, x2v, x3v, IVX, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wim1, wi, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wim1, wi, bxi, eos, flux);
        }

        // store 1st-order fluxes.
        flx1(m,IDN,k,j,i) = flux.d;
        flx1(m,IM1,k,j,i) = flux.mx;
        flx1(m,IM2,k,j,i) = flux.my;
        flx1(m,IM3,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx1(m,IEN,k,j,i) = flux.e;}
        e3x1_(m,k,j,i) = flux.by;
        e2x1_(m,k,j,i) = flux.bz;
      }

      if (multi_d) {
        // load W_{j-1} state, permutting components of vectors
        MHDPrim1D wjm1;
        wjm1.d  = w0_(m,IDN,k,j-1,i);
        wjm1.vx = w0_(m,IVY,k,j-1,i);
        wjm1.vy = w0_(m,IVZ,k,j-1,i);
        wjm1.vz = w0_(m,IVX,k,j-1,i);
        if (eos.is_ideal) {wjm1.e = w0_(m,IEN,k,j-1,i);}
        wjm1.by = bcc0_(m,IBZ,k,j-1,i);
        wjm1.bz = bcc0_(m,IBX,k,j-1,i);

        // load W_{j} state, permutting components of vectors
        MHDPrim1D wj;
        wj.d  = w0_(m,IDN,k,j,i);
        wj.vx = w0_(m,IVY,k,j,i);
        wj.vy = w0_(m,IVZ,k,j,i);
        wj.vz = w0_(m,IVX,k,j,i);
        if (eos.is_ideal) {wj.e = w0_(m,IEN,k,j,i);}
        wj.by = bcc0_(m,IBZ,k,j,i);
        wj.bz = bcc0_(m,IBX,k,j,i);

        // compute new first-order flux at j-face
        Real bxi = b0_.x2f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wjm1, wj, bxi, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wjm1, wj, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wjm1, wj, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx2(m,IDN,k,j,i) = flux.d;
        flx2(m,IM2,k,j,i) = flux.mx;
        flx2(m,IM3,k,j,i) = flux.my;
        flx2(m,IM1,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j,i) = flux.e;}
        e1x2_(m,k,j,i) = flux.by;
        e3x2_(m,k,j,i) = flux.bz;
      }

      if (three_d) {
        // load W_{k-1} state, permutting components of vectors
        MHDPrim1D wkm1;
        wkm1.d  = w0_(m,IDN,k-1,j,i);
        wkm1.vx = w0_(m,IVZ,k-1,j,i);
        wkm1.vy = w0_(m,IVX,k-1,j,i);
        wkm1.vz = w0_(m,IVY,k-1,j,i);
        if (eos.is_ideal) {wkm1.e = w0_(m,IEN,k-1,j,i);}
        wkm1.by = bcc0_(m,IBX,k-1,j,i);
        wkm1.bz = bcc0_(m,IBY,k-1,j,i);

        // load W_{k} state, permutting components of vectors
        MHDPrim1D wk;
        wk.d  = w0_(m,IDN,k,j,i);
        wk.vx = w0_(m,IVZ,k,j,i);
        wk.vy = w0_(m,IVX,k,j,i);
        wk.vz = w0_(m,IVY,k,j,i);
        if (eos.is_ideal) {wk.e = w0_(m,IEN,k,j,i);}
        wk.by = bcc0_(m,IBX,k,j,i);
        wk.bz = bcc0_(m,IBY,k,j,i);

        // compute new first-order flux at k-face
        Real bxi = b0_.x3f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wkm1, wk, bxi, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wkm1, wk, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wkm1, wk, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx3(m,IDN,k,j,i) = flux.d;
        flx3(m,IM3,k,j,i) = flux.mx;
        flx3(m,IM1,k,j,i) = flux.my;
        flx3(m,IM2,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k,j,i) = flux.e;}
        e2x3_(m,k,j,i) = flux.by;
        e1x3_(m,k,j,i) = flux.bz;
      }
    } else if (use_sofc_ && fofc_flag && !fofc_excision) {
      // Apply SOFC (PLM + LLF) when flag set and stencil available
      if (i >= is+1) {
        // PLM left/right states at i-face (ql_i from cell i-1, qr_i from cell i)
        MHDPrim1D wL, wR;
        Real ql_i, qr_im1, ql_ip1, qr_i;
        PLM(w0_(m,IDN,k,j,i-2), w0_(m,IDN,k,j,i-1), w0_(m,IDN,k,j,i), ql_i, qr_im1);
        wL.d = ql_i;
        PLM(w0_(m,IDN,k,j,i-1), w0_(m,IDN,k,j,i), w0_(m,IDN,k,j,i+1), ql_ip1, qr_i);
        wR.d = qr_i;
        PLM(w0_(m,IVX,k,j,i-2), w0_(m,IVX,k,j,i-1), w0_(m,IVX,k,j,i), ql_i, qr_im1);
        wL.vx = ql_i;
        PLM(w0_(m,IVX,k,j,i-1), w0_(m,IVX,k,j,i), w0_(m,IVX,k,j,i+1), ql_ip1, qr_i);
        wR.vx = qr_i;
        PLM(w0_(m,IVY,k,j,i-2), w0_(m,IVY,k,j,i-1), w0_(m,IVY,k,j,i), ql_i, qr_im1);
        wL.vy = ql_i;
        PLM(w0_(m,IVY,k,j,i-1), w0_(m,IVY,k,j,i), w0_(m,IVY,k,j,i+1), ql_ip1, qr_i);
        wR.vy = qr_i;
        PLM(w0_(m,IVZ,k,j,i-2), w0_(m,IVZ,k,j,i-1), w0_(m,IVZ,k,j,i), ql_i, qr_im1);
        wL.vz = ql_i;
        PLM(w0_(m,IVZ,k,j,i-1), w0_(m,IVZ,k,j,i), w0_(m,IVZ,k,j,i+1), ql_ip1, qr_i);
        wR.vz = qr_i;
        if (eos.is_ideal) {
          PLM(w0_(m,IEN,k,j,i-2), w0_(m,IEN,k,j,i-1), w0_(m,IEN,k,j,i), ql_i, qr_im1);
          wL.e = ql_i;
          PLM(w0_(m,IEN,k,j,i-1), w0_(m,IEN,k,j,i), w0_(m,IEN,k,j,i+1), ql_ip1, qr_i);
          wR.e = qr_i;
        }
        PLM(bcc0_(m,IBY,k,j,i-2), bcc0_(m,IBY,k,j,i-1), bcc0_(m,IBY,k,j,i), ql_i, qr_im1);
        wL.by = ql_i;
        PLM(bcc0_(m,IBY,k,j,i-1), bcc0_(m,IBY,k,j,i), bcc0_(m,IBY,k,j,i+1), ql_ip1, qr_i);
        wR.by = qr_i;
        PLM(bcc0_(m,IBZ,k,j,i-2), bcc0_(m,IBZ,k,j,i-1), bcc0_(m,IBZ,k,j,i), ql_i, qr_im1);
        wL.bz = ql_i;
        PLM(bcc0_(m,IBZ,k,j,i-1), bcc0_(m,IBZ,k,j,i), bcc0_(m,IBZ,k,j,i+1), ql_ip1, qr_i);
        wR.bz = qr_i;

        // compute new SOFC (PLM+LLF) flux at i-face
        {
          Real bxi = b0_.x1f(m,k,j,i);
          MHDCons1D flux;
          if (is_gr) {
            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            Real x1v = LeftEdgeX(i-is, nx1, x1min, x1max);
            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
            SingleStateLLF_GRMHD(wL, wR, bxi, x1v, x2v, x3v, IVX, coord, eos, flux);
          } else if (is_sr) {
            SingleStateLLF_SRMHD(wL, wR, bxi, eos, flux);
          } else {
            SingleStateLLF_MHD(wL, wR, bxi, eos, flux);
          }

          // store SOFC fluxes.
          flx1(m,IDN,k,j,i) = flux.d;
          flx1(m,IM1,k,j,i) = flux.mx;
          flx1(m,IM2,k,j,i) = flux.my;
          flx1(m,IM3,k,j,i) = flux.mz;
          if (eos.is_ideal) {flx1(m,IEN,k,j,i) = flux.e;}
          e3x1_(m,k,j,i) = flux.by;
          e2x1_(m,k,j,i) = flux.bz;
        }
      }

      if (multi_d && j >= js+1) {
        // PLM left/right states at j-face (permutting components for 1D solver)
        MHDPrim1D wL, wR;
        Real ql_j, qr_jm1, ql_jp1, qr_j;
        PLM(w0_(m,IDN,k,j-2,i), w0_(m,IDN,k,j-1,i), w0_(m,IDN,k,j,i), ql_j, qr_jm1);
        wL.d = ql_j;
        PLM(w0_(m,IDN,k,j-1,i), w0_(m,IDN,k,j,i), w0_(m,IDN,k,j+1,i), ql_jp1, qr_j);
        wR.d = qr_j;
        PLM(w0_(m,IVY,k,j-2,i), w0_(m,IVY,k,j-1,i), w0_(m,IVY,k,j,i), ql_j, qr_jm1);
        wL.vx = ql_j;
        PLM(w0_(m,IVY,k,j-1,i), w0_(m,IVY,k,j,i), w0_(m,IVY,k,j+1,i), ql_jp1, qr_j);
        wR.vx = qr_j;
        PLM(w0_(m,IVZ,k,j-2,i), w0_(m,IVZ,k,j-1,i), w0_(m,IVZ,k,j,i), ql_j, qr_jm1);
        wL.vy = ql_j;
        PLM(w0_(m,IVZ,k,j-1,i), w0_(m,IVZ,k,j,i), w0_(m,IVZ,k,j+1,i), ql_jp1, qr_j);
        wR.vy = qr_j;
        PLM(w0_(m,IVX,k,j-2,i), w0_(m,IVX,k,j-1,i), w0_(m,IVX,k,j,i), ql_j, qr_jm1);
        wL.vz = ql_j;
        PLM(w0_(m,IVX,k,j-1,i), w0_(m,IVX,k,j,i), w0_(m,IVX,k,j+1,i), ql_jp1, qr_j);
        wR.vz = qr_j;
        if (eos.is_ideal) {
          PLM(w0_(m,IEN,k,j-2,i), w0_(m,IEN,k,j-1,i), w0_(m,IEN,k,j,i), ql_j, qr_jm1);
          wL.e = ql_j;
          PLM(w0_(m,IEN,k,j-1,i), w0_(m,IEN,k,j,i), w0_(m,IEN,k,j+1,i), ql_jp1, qr_j);
          wR.e = qr_j;
        }
        PLM(bcc0_(m,IBZ,k,j-2,i), bcc0_(m,IBZ,k,j-1,i), bcc0_(m,IBZ,k,j,i), ql_j, qr_jm1);
        wL.by = ql_j;
        PLM(bcc0_(m,IBZ,k,j-1,i), bcc0_(m,IBZ,k,j,i), bcc0_(m,IBZ,k,j+1,i), ql_jp1, qr_j);
        wR.by = qr_j;
        PLM(bcc0_(m,IBX,k,j-2,i), bcc0_(m,IBX,k,j-1,i), bcc0_(m,IBX,k,j,i), ql_j, qr_jm1);
        wL.bz = ql_j;
        PLM(bcc0_(m,IBX,k,j-1,i), bcc0_(m,IBX,k,j,i), bcc0_(m,IBX,k,j+1,i), ql_jp1, qr_j);
        wR.bz = qr_j;

        // compute new SOFC (PLM+LLF) flux at j-face
        Real bxi = b0_.x2f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j-js, nx2, x2min, x2max);
          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wL, wR, bxi, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wL, wR, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wL, wR, bxi, eos, flux);
        }

        // store SOFC fluxes, permutting indices.
        flx2(m,IDN,k,j,i) = flux.d;
        flx2(m,IM2,k,j,i) = flux.mx;
        flx2(m,IM3,k,j,i) = flux.my;
        flx2(m,IM1,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j,i) = flux.e;}
        e1x2_(m,k,j,i) = flux.by;
        e3x2_(m,k,j,i) = flux.bz;
      }

      if (three_d && k >= ks+1) {
        // PLM left/right states at k-face (permutting components for 1D solver)
        MHDPrim1D wL, wR;
        Real ql_k, qr_km1, ql_kp1, qr_k;
        PLM(w0_(m,IDN,k-2,j,i), w0_(m,IDN,k-1,j,i), w0_(m,IDN,k,j,i), ql_k, qr_km1);
        wL.d = ql_k;
        PLM(w0_(m,IDN,k-1,j,i), w0_(m,IDN,k,j,i), w0_(m,IDN,k+1,j,i), ql_kp1, qr_k);
        wR.d = qr_k;
        PLM(w0_(m,IVZ,k-2,j,i), w0_(m,IVZ,k-1,j,i), w0_(m,IVZ,k,j,i), ql_k, qr_km1);
        wL.vx = ql_k;
        PLM(w0_(m,IVZ,k-1,j,i), w0_(m,IVZ,k,j,i), w0_(m,IVZ,k+1,j,i), ql_kp1, qr_k);
        wR.vx = qr_k;
        PLM(w0_(m,IVX,k-2,j,i), w0_(m,IVX,k-1,j,i), w0_(m,IVX,k,j,i), ql_k, qr_km1);
        wL.vy = ql_k;
        PLM(w0_(m,IVX,k-1,j,i), w0_(m,IVX,k,j,i), w0_(m,IVX,k+1,j,i), ql_kp1, qr_k);
        wR.vy = qr_k;
        PLM(w0_(m,IVY,k-2,j,i), w0_(m,IVY,k-1,j,i), w0_(m,IVY,k,j,i), ql_k, qr_km1);
        wL.vz = ql_k;
        PLM(w0_(m,IVY,k-1,j,i), w0_(m,IVY,k,j,i), w0_(m,IVY,k+1,j,i), ql_kp1, qr_k);
        wR.vz = qr_k;
        if (eos.is_ideal) {
          PLM(w0_(m,IEN,k-2,j,i), w0_(m,IEN,k-1,j,i), w0_(m,IEN,k,j,i), ql_k, qr_km1);
          wL.e = ql_k;
          PLM(w0_(m,IEN,k-1,j,i), w0_(m,IEN,k,j,i), w0_(m,IEN,k+1,j,i), ql_kp1, qr_k);
          wR.e = qr_k;
        }
        PLM(bcc0_(m,IBX,k-2,j,i), bcc0_(m,IBX,k-1,j,i), bcc0_(m,IBX,k,j,i), ql_k, qr_km1);
        wL.by = ql_k;
        PLM(bcc0_(m,IBX,k-1,j,i), bcc0_(m,IBX,k,j,i), bcc0_(m,IBX,k+1,j,i), ql_kp1, qr_k);
        wR.by = qr_k;
        PLM(bcc0_(m,IBY,k-2,j,i), bcc0_(m,IBY,k-1,j,i), bcc0_(m,IBY,k,j,i), ql_k, qr_km1);
        wL.bz = ql_k;
        PLM(bcc0_(m,IBY,k-1,j,i), bcc0_(m,IBY,k,j,i), bcc0_(m,IBY,k+1,j,i), ql_kp1, qr_k);
        wR.bz = qr_k;

        // compute new SOFC (PLM+LLF) flux at k-face
        Real bxi = b0_.x3f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wL, wR, bxi, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wL, wR, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wL, wR, bxi, eos, flux);
        }

        // store SOFC fluxes, permutting indices.
        flx3(m,IDN,k,j,i) = flux.d;
        flx3(m,IM3,k,j,i) = flux.mx;
        flx3(m,IM1,k,j,i) = flux.my;
        flx3(m,IM2,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k,j,i) = flux.e;}
        e2x3_(m,k,j,i) = flux.by;
        e1x3_(m,k,j,i) = flux.bz;
      }

    }
  });

  // Replace fluxes with first-order LLF fluxes at i+1,j+1,k+1 faces for any cell where
  // FOFC and/or excision is used (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_ || use_sofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Check for GR + excision
    bool fofc_excision = false;
    if (is_gr) {
      if (use_excise_) { fofc_excision = excision_flux_(m,k,j,i); }
    }

    // Apply FOFC (first-order)
    if (use_fofc_ && (fofc_flag || fofc_excision)) {
      // load W_{i} state
      MHDPrim1D wi;
      wi.d  = w0_(m,IDN,k,j,i);
      wi.vx = w0_(m,IVX,k,j,i);
      wi.vy = w0_(m,IVY,k,j,i);
      wi.vz = w0_(m,IVZ,k,j,i);
      if (eos.is_ideal) {wi.e = w0_(m,IEN,k,j,i);}
      wi.by = bcc0_(m,IBY,k,j,i);
      wi.bz = bcc0_(m,IBZ,k,j,i);

      // load W_{i+1} state
      MHDPrim1D wip1;
      wip1.d  = w0_(m,IDN,k,j,i+1);
      wip1.vx = w0_(m,IVX,k,j,i+1);
      wip1.vy = w0_(m,IVY,k,j,i+1);
      wip1.vz = w0_(m,IVZ,k,j,i+1);
      if (eos.is_ideal) {wip1.e = w0_(m,IEN,k,j,i+1);}
      wip1.by = bcc0_(m,IBY,k,j,i+1);
      wip1.bz = bcc0_(m,IBZ,k,j,i+1);

      // compute new 1st-order LLF flux at (i+1)-face
      {
        Real bxi = b0_.x1f(m,k,j,i+1);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = LeftEdgeX(i+1-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wi, wip1, bxi, x1v, x2v, x3v, IVX, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wi, wip1, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wi, wip1, bxi, eos, flux);
        }

        // store 1st-order fluxes.
        flx1(m,IDN,k,j,i+1) = flux.d;
        flx1(m,IM1,k,j,i+1) = flux.mx;
        flx1(m,IM2,k,j,i+1) = flux.my;
        flx1(m,IM3,k,j,i+1) = flux.mz;
        if (eos.is_ideal) {flx1(m,IEN,k,j,i+1) = flux.e;}
        e3x1_(m,k,j,i+1) = flux.by;
        e2x1_(m,k,j,i+1) = flux.bz;
      }

      if (multi_d) {
        // load W_{j} state, permutting components of vectors
        MHDPrim1D wj;
        wj.d  = w0_(m,IDN,k,j,i);
        wj.vx = w0_(m,IVY,k,j,i);
        wj.vy = w0_(m,IVZ,k,j,i);
        wj.vz = w0_(m,IVX,k,j,i);
        if (eos.is_ideal) {wj.e = w0_(m,IEN,k,j,i);}
        wj.by = bcc0_(m,IBZ,k,j,i);
        wj.bz = bcc0_(m,IBX,k,j,i);

        // load W_{j+1} state, permutting components of vectors
        MHDPrim1D wjp1;
        wjp1.d  = w0_(m,IDN,k,j+1,i);
        wjp1.vx = w0_(m,IVY,k,j+1,i);
        wjp1.vy = w0_(m,IVZ,k,j+1,i);
        wjp1.vz = w0_(m,IVX,k,j+1,i);
        if (eos.is_ideal) {wjp1.e = w0_(m,IEN,k,j+1,i);}
        wjp1.by = bcc0_(m,IBZ,k,j+1,i);
        wjp1.bz = bcc0_(m,IBX,k,j+1,i);

        // compute new first-order flux at (j+1)-face
        Real bxi = b0_.x2f(m,k,j+1,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j+1-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wj, wjp1, bxi, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wj, wjp1, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wj, wjp1, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx2(m,IDN,k,j+1,i) = flux.d;
        flx2(m,IM2,k,j+1,i) = flux.mx;
        flx2(m,IM3,k,j+1,i) = flux.my;
        flx2(m,IM1,k,j+1,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j+1,i) = flux.e;}
        e1x2_(m,k,j+1,i) = flux.by;
        e3x2_(m,k,j+1,i) = flux.bz;
      }

      if (three_d) {
        // load W_{k} state, permutting components of vectors
        MHDPrim1D wk;
        wk.d  = w0_(m,IDN,k,j,i);
        wk.vx = w0_(m,IVZ,k,j,i);
        wk.vy = w0_(m,IVX,k,j,i);
        wk.vz = w0_(m,IVY,k,j,i);
        if (eos.is_ideal) {wk.e = w0_(m,IEN,k,j,i);}
        wk.by = bcc0_(m,IBX,k,j,i);
        wk.bz = bcc0_(m,IBY,k,j,i);

        // load W_{k+1} state, permutting components of vectors
        MHDPrim1D wkp1;
        wkp1.d  = w0_(m,IDN,k+1,j,i);
        wkp1.vx = w0_(m,IVZ,k+1,j,i);
        wkp1.vy = w0_(m,IVX,k+1,j,i);
        wkp1.vz = w0_(m,IVY,k+1,j,i);
        if (eos.is_ideal) {wkp1.e = w0_(m,IEN,k+1,j,i);}
        wkp1.by = bcc0_(m,IBX,k+1,j,i);
        wkp1.bz = bcc0_(m,IBY,k+1,j,i);

        // compute new first-order flux at (k+1)-face
        Real bxi = b0_.x3f(m,k+1,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wk, wkp1, bxi, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wk, wkp1, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wk, wkp1, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx3(m,IDN,k+1,j,i) = flux.d;
        flx3(m,IM3,k+1,j,i) = flux.mx;
        flx3(m,IM1,k+1,j,i) = flux.my;
        flx3(m,IM2,k+1,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k+1,j,i) = flux.e;}
        e2x3_(m,k+1,j,i) = flux.by;
        e1x3_(m,k+1,j,i) = flux.bz;
      }
    } else if (use_sofc_ && fofc_flag && !fofc_excision) {
        // Apply SOFC (PLM + LLF) at i+1, j+1, k+1 faces
        if (i <= ie-1) {
          // PLM left/right states at (i+1)-face
          MHDPrim1D wL, wR;
          Real ql_ip1, qr_i, ql_ip2, qr_ip1;
          PLM(w0_(m,IDN,k,j,i-1), w0_(m,IDN,k,j,i), w0_(m,IDN,k,j,i+1), ql_ip1, qr_i);
          wL.d = ql_ip1;
          PLM(w0_(m,IDN,k,j,i), w0_(m,IDN,k,j,i+1), w0_(m,IDN,k,j,i+2), ql_ip2, qr_ip1);
          wR.d = qr_ip1;
          PLM(w0_(m,IVX,k,j,i-1), w0_(m,IVX,k,j,i), w0_(m,IVX,k,j,i+1), ql_ip1, qr_i);
          wL.vx = ql_ip1;
          PLM(w0_(m,IVX,k,j,i), w0_(m,IVX,k,j,i+1), w0_(m,IVX,k,j,i+2), ql_ip2, qr_ip1);
          wR.vx = qr_ip1;
          PLM(w0_(m,IVY,k,j,i-1), w0_(m,IVY,k,j,i), w0_(m,IVY,k,j,i+1), ql_ip1, qr_i);
          wL.vy = ql_ip1;
          PLM(w0_(m,IVY,k,j,i), w0_(m,IVY,k,j,i+1), w0_(m,IVY,k,j,i+2), ql_ip2, qr_ip1);
          wR.vy = qr_ip1;
          PLM(w0_(m,IVZ,k,j,i-1), w0_(m,IVZ,k,j,i), w0_(m,IVZ,k,j,i+1), ql_ip1, qr_i);
          wL.vz = ql_ip1;
          PLM(w0_(m,IVZ,k,j,i), w0_(m,IVZ,k,j,i+1), w0_(m,IVZ,k,j,i+2), ql_ip2, qr_ip1);
          wR.vz = qr_ip1;
          if (eos.is_ideal) {
            PLM(w0_(m,IEN,k,j,i-1), w0_(m,IEN,k,j,i), w0_(m,IEN,k,j,i+1), ql_ip1, qr_i);
            wL.e = ql_ip1;
            PLM(w0_(m,IEN,k,j,i), w0_(m,IEN,k,j,i+1), w0_(m,IEN,k,j,i+2), ql_ip2, qr_ip1);
            wR.e = qr_ip1;
          }
          PLM(bcc0_(m,IBY,k,j,i-1), bcc0_(m,IBY,k,j,i), bcc0_(m,IBY,k,j,i+1), ql_ip1, qr_i);
          wL.by = ql_ip1;
          PLM(bcc0_(m,IBY,k,j,i), bcc0_(m,IBY,k,j,i+1), bcc0_(m,IBY,k,j,i+2), ql_ip2, qr_ip1);
          wR.by = qr_ip1;
          PLM(bcc0_(m,IBZ,k,j,i-1), bcc0_(m,IBZ,k,j,i), bcc0_(m,IBZ,k,j,i+1), ql_ip1, qr_i);
          wL.bz = ql_ip1;
          PLM(bcc0_(m,IBZ,k,j,i), bcc0_(m,IBZ,k,j,i+1), bcc0_(m,IBZ,k,j,i+2), ql_ip2, qr_ip1);
          wR.bz = qr_ip1;

          // compute new SOFC (PLM+LLF) flux at (i+1)-face
          {
            Real bxi = b0_.x1f(m,k,j,i+1);
            MHDCons1D flux;
            if (is_gr) {
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              Real x1v = LeftEdgeX(i+1-is, nx1, x1min, x1max);
              Real &x2min = size.d_view(m).x2min;
              Real &x2max = size.d_view(m).x2max;
              Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
              Real &x3min = size.d_view(m).x3min;
              Real &x3max = size.d_view(m).x3max;
              Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
              SingleStateLLF_GRMHD(wL, wR, bxi, x1v, x2v, x3v, IVX, coord, eos, flux);
            } else if (is_sr) {
              SingleStateLLF_SRMHD(wL, wR, bxi, eos, flux);
            } else {
              SingleStateLLF_MHD(wL, wR, bxi, eos, flux);
            }

            // store SOFC fluxes.
            flx1(m,IDN,k,j,i+1) = flux.d;
            flx1(m,IM1,k,j,i+1) = flux.mx;
            flx1(m,IM2,k,j,i+1) = flux.my;
            flx1(m,IM3,k,j,i+1) = flux.mz;
            if (eos.is_ideal) {flx1(m,IEN,k,j,i+1) = flux.e;}
            e3x1_(m,k,j,i+1) = flux.by;
            e2x1_(m,k,j,i+1) = flux.bz;
          }
        }

        if (multi_d && j <= je-1) {
          // PLM left/right states at (j+1)-face (permutting components for 1D solver)
          MHDPrim1D wL, wR;
          Real ql_jp1, qr_j, ql_jp2, qr_jp1;
          PLM(w0_(m,IDN,k,j-1,i), w0_(m,IDN,k,j,i), w0_(m,IDN,k,j+1,i), ql_jp1, qr_j);
          wL.d = ql_jp1;
          PLM(w0_(m,IDN,k,j,i), w0_(m,IDN,k,j+1,i), w0_(m,IDN,k,j+2,i), ql_jp2, qr_jp1);
          wR.d = qr_jp1;
          PLM(w0_(m,IVY,k,j-1,i), w0_(m,IVY,k,j,i), w0_(m,IVY,k,j+1,i), ql_jp1, qr_j);
          wL.vx = ql_jp1;
          PLM(w0_(m,IVY,k,j,i), w0_(m,IVY,k,j+1,i), w0_(m,IVY,k,j+2,i), ql_jp2, qr_jp1);
          wR.vx = qr_jp1;
          PLM(w0_(m,IVZ,k,j-1,i), w0_(m,IVZ,k,j,i), w0_(m,IVZ,k,j+1,i), ql_jp1, qr_j);
          wL.vy = ql_jp1;
          PLM(w0_(m,IVZ,k,j,i), w0_(m,IVZ,k,j+1,i), w0_(m,IVZ,k,j+2,i), ql_jp2, qr_jp1);
          wR.vy = qr_jp1;
          PLM(w0_(m,IVX,k,j-1,i), w0_(m,IVX,k,j,i), w0_(m,IVX,k,j+1,i), ql_jp1, qr_j);
          wL.vz = ql_jp1;
          PLM(w0_(m,IVX,k,j,i), w0_(m,IVX,k,j+1,i), w0_(m,IVX,k,j+2,i), ql_jp2, qr_jp1);
          wR.vz = qr_jp1;
          if (eos.is_ideal) {
            PLM(w0_(m,IEN,k,j-1,i), w0_(m,IEN,k,j,i), w0_(m,IEN,k,j+1,i), ql_jp1, qr_j);
            wL.e = ql_jp1;
            PLM(w0_(m,IEN,k,j,i), w0_(m,IEN,k,j+1,i), w0_(m,IEN,k,j+2,i), ql_jp2, qr_jp1);
            wR.e = qr_jp1;
          }
          PLM(bcc0_(m,IBZ,k,j-1,i), bcc0_(m,IBZ,k,j,i), bcc0_(m,IBZ,k,j+1,i), ql_jp1, qr_j);
          wL.by = ql_jp1;
          PLM(bcc0_(m,IBZ,k,j,i), bcc0_(m,IBZ,k,j+1,i), bcc0_(m,IBZ,k,j+2,i), ql_jp2, qr_jp1);
          wR.by = qr_jp1;
          PLM(bcc0_(m,IBX,k,j-1,i), bcc0_(m,IBX,k,j,i), bcc0_(m,IBX,k,j+1,i), ql_jp1, qr_j);
          wL.bz = ql_jp1;
          PLM(bcc0_(m,IBX,k,j,i), bcc0_(m,IBX,k,j+1,i), bcc0_(m,IBX,k,j+2,i), ql_jp2, qr_jp1);
          wR.bz = qr_jp1;

          // compute new SOFC (PLM+LLF) flux at (j+1)-face
          Real bxi = b0_.x2f(m,k,j+1,i);
          MHDCons1D flux;
          if (is_gr) {
            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            Real x2v = LeftEdgeX(j+1-js, nx2, x2min, x2max);
            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
            SingleStateLLF_GRMHD(wL, wR, bxi, x1v, x2v, x3v, IVY, coord, eos, flux);
          } else if (is_sr) {
            SingleStateLLF_SRMHD(wL, wR, bxi, eos, flux);
          } else {
            SingleStateLLF_MHD(wL, wR, bxi, eos, flux);
          }

          // store SOFC fluxes, permutting indices.
          flx2(m,IDN,k,j+1,i) = flux.d;
          flx2(m,IM2,k,j+1,i) = flux.mx;
          flx2(m,IM3,k,j+1,i) = flux.my;
          flx2(m,IM1,k,j+1,i) = flux.mz;
          if (eos.is_ideal) {flx2(m,IEN,k,j+1,i) = flux.e;}
          e1x2_(m,k,j+1,i) = flux.by;
          e3x2_(m,k,j+1,i) = flux.bz;
        }

        if (three_d && k <= ke-1) {
          // PLM left/right states at (k+1)-face (permutting components for 1D solver)
          MHDPrim1D wL, wR;
          Real ql_kp1, qr_k, ql_kp2, qr_kp1;
          PLM(w0_(m,IDN,k-1,j,i), w0_(m,IDN,k,j,i), w0_(m,IDN,k+1,j,i), ql_kp1, qr_k);
          wL.d = ql_kp1;
          PLM(w0_(m,IDN,k,j,i), w0_(m,IDN,k+1,j,i), w0_(m,IDN,k+2,j,i), ql_kp2, qr_kp1);
          wR.d = qr_kp1;
          PLM(w0_(m,IVZ,k-1,j,i), w0_(m,IVZ,k,j,i), w0_(m,IVZ,k+1,j,i), ql_kp1, qr_k);
          wL.vx = ql_kp1;
          PLM(w0_(m,IVZ,k,j,i), w0_(m,IVZ,k+1,j,i), w0_(m,IVZ,k+2,j,i), ql_kp2, qr_kp1);
          wR.vx = qr_kp1;
          PLM(w0_(m,IVX,k-1,j,i), w0_(m,IVX,k,j,i), w0_(m,IVX,k+1,j,i), ql_kp1, qr_k);
          wL.vy = ql_kp1;
          PLM(w0_(m,IVX,k,j,i), w0_(m,IVX,k+1,j,i), w0_(m,IVX,k+2,j,i), ql_kp2, qr_kp1);
          wR.vy = qr_kp1;
          PLM(w0_(m,IVY,k-1,j,i), w0_(m,IVY,k,j,i), w0_(m,IVY,k+1,j,i), ql_kp1, qr_k);
          wL.vz = ql_kp1;
          PLM(w0_(m,IVY,k,j,i), w0_(m,IVY,k+1,j,i), w0_(m,IVY,k+2,j,i), ql_kp2, qr_kp1);
          wR.vz = qr_kp1;
          if (eos.is_ideal) {
            PLM(w0_(m,IEN,k-1,j,i), w0_(m,IEN,k,j,i), w0_(m,IEN,k+1,j,i), ql_kp1, qr_k);
            wL.e = ql_kp1;
            PLM(w0_(m,IEN,k,j,i), w0_(m,IEN,k+1,j,i), w0_(m,IEN,k+2,j,i), ql_kp2, qr_kp1);
            wR.e = qr_kp1;
          }
          PLM(bcc0_(m,IBX,k-1,j,i), bcc0_(m,IBX,k,j,i), bcc0_(m,IBX,k+1,j,i), ql_kp1, qr_k);
          wL.by = ql_kp1;
          PLM(bcc0_(m,IBX,k,j,i), bcc0_(m,IBX,k+1,j,i), bcc0_(m,IBX,k+2,j,i), ql_kp2, qr_kp1);
          wR.by = qr_kp1;
          PLM(bcc0_(m,IBY,k-1,j,i), bcc0_(m,IBY,k,j,i), bcc0_(m,IBY,k+1,j,i), ql_kp1, qr_k);
          wL.bz = ql_kp1;
          PLM(bcc0_(m,IBY,k,j,i), bcc0_(m,IBY,k+1,j,i), bcc0_(m,IBY,k+2,j,i), ql_kp2, qr_kp1);
          wR.bz = qr_kp1;

          // compute new SOFC (PLM+LLF) flux at (k+1)-face
          Real bxi = b0_.x3f(m,k+1,j,i);
          MHDCons1D flux;
          if (is_gr) {
            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            Real x3v = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
            SingleStateLLF_GRMHD(wL, wR, bxi, x1v, x2v, x3v, IVZ, coord, eos, flux);
          } else if (is_sr) {
            SingleStateLLF_SRMHD(wL, wR, bxi, eos, flux);
          } else {
            SingleStateLLF_MHD(wL, wR, bxi, eos, flux);
          }

          // store SOFC fluxes, permutting indices.
          flx3(m,IDN,k+1,j,i) = flux.d;
          flx3(m,IM3,k+1,j,i) = flux.mx;
          flx3(m,IM1,k+1,j,i) = flux.my;
          flx3(m,IM2,k+1,j,i) = flux.mz;
          if (eos.is_ideal) {flx3(m,IEN,k+1,j,i) = flux.e;}
          e2x3_(m,k+1,j,i) = flux.by;
          e1x3_(m,k+1,j,i) = flux.bz;
        }

    }
  });

  // reset FOFC/SOFC flag (do not reset excision flag)
  if (use_fofc_ || use_sofc_) {
    Kokkos::deep_copy(fofc, false);
  }

  return;
}

} // namespace mhd
