//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic mhd

#include <float.h>

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

#include "radiation/radiation.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRMHD::IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("mhd","gamma_max",(FLT_MAX));  // gamma ceiling
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.
//! Operates over range of cells given in argument list.

void IdealGRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                            DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                            const bool only_testfloors, const bool temperature_fix,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto &customize_fofc_ = pmy_pack->pmhd->customize_fofc;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;

  // flags and variables for ad hoc fixes
  auto &c2p_flag_ = pmy_pack->pmhd->c2p_flag;
  auto &smooth_flag_ = pmy_pack->pmhd->smooth_flag;
  auto &w0_old_ = pmy_pack->pmhd->w0_old;
  auto &is_radiation_enabled_ = pmy_pack->pmhd->is_radiation_enabled;
  DvceArray4D<Real> tgas_radsource_;
  if (is_radiation_enabled_) tgas_radsource_ = pmy_pack->prad->tgas_radsource;
  bool &cellavg_fix_turn_on_ = pmy_pack->pmhd->cellavg_fix_turn_on;

  // flags and variables for entropy fix
  auto &entropy_fix_ = pmy_pack->pmhd->entropy_fix;
  auto &entropy_fix_turnoff_ = pmy_pack->pmhd->entropy_fix_turnoff;
  int entropyIdx = (entropy_fix_) ? nmhd+nscal-1 : -1;
  auto c2p_test_ = pmy_pack->pmhd->c2p_test;
  auto &sigma_cold_cut_ = pmy_pack->pmhd->sigma_cold_cut;
  auto &r_tfix_cut_ = pmy_pack->pmhd->r_tfix_cut;

  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nceilv_=0, nfail_=0, maxit_=0;
  Kokkos::parallel_reduce("grmhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumv, int &sumf, int &max_it) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if ((only_testfloors) || (c2p_test_)) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rv = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    bool vceiling_used=false, c2p_failure=false;
    int iter_used=0;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        w.d = dexcise_;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        w.e = pexcise_/gm1;
        excised = true;
      }
      if (only_testfloors) {
        if (excision_flux_(m,k,j,i)) {
          excised = true;
        }
      }
    }

    if (excised) {
      c2p_flag_(m,k,j,i) = true;
      smooth_flag_(m,k,j,i) = false;
    }

    if (!(excised)) {
      // calculate SR conserved quantities
      MHDCons1D u_sr;
      Real s2, b2, rpar;
      TransformToSRMHD(u,glower,gupper,s2,b2,rpar,u_sr);

      // call c2p function
      // (inline function in ideal_c2p_mhd.hpp file)
      SingleC2P_IdealSRMHD(u_sr, eos, s2, b2, rpar, w,
                           dfloor_used, efloor_used, c2p_failure, iter_used);

      HydPrim1D w_old;
      w_old.d  = w0_old_(m,IDN,k,j,i);
      w_old.vx = w0_old_(m,IVX,k,j,i);
      w_old.vy = w0_old_(m,IVY,k,j,i);
      w_old.vz = w0_old_(m,IVZ,k,j,i);
      w_old.e  = w0_old_(m,IEN,k,j,i);
      // SingleC2P_IdealSRMHD_NH(u_sr, eos, s2, b2, rpar, w, w_old,
      //                         dfloor_used, efloor_used, c2p_failure, iter_used);

      // apply old value fix
      if (c2p_failure) {
        w.d  = w_old.d;
        w.e  = w_old.e;
        w.vx = w_old.vx;
        w.vy = w_old.vy;
        w.vz = w_old.vz;
      }

      // compute sigma_cold (2*pmag/rho) to decide whether turn on the fixes
      Real sigma_cold = 0.0;
      if (!c2p_failure) {
        Real qq = glower[1][1]*w.vx*w.vx +2.0*glower[1][2]*w.vx*w.vy +2.0*glower[1][3]*w.vx*w.vz
                + glower[2][2]*w.vy*w.vy +2.0*glower[2][3]*w.vy*w.vz
                + glower[3][3]*w.vz*w.vz;
        Real alpha = sqrt(-1.0/gupper[0][0]);
        Real u0_norm = sqrt(1.0 + qq);
        Real u0 = u0_norm / alpha;
        Real u1 = w.vx - alpha * u0_norm * gupper[0][1];
        Real u2 = w.vy - alpha * u0_norm * gupper[0][2];
        Real u3 = w.vz - alpha * u0_norm * gupper[0][3];

        // lower vector indices
        Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
        Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
        Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

        // calculate 4-magnetic field
        Real b0_ = u_1*u.bx + u_2*u.by + u_3*u.bz;
        Real b1_ = (u.bx + b0_ * u1) / u0;
        Real b2_ = (u.by + b0_ * u2) / u0;
        Real b3_ = (u.bz + b0_ * u3) / u0;

        // lower vector indices
        Real b_0 = glower[0][0]*b0_ + glower[0][1]*b1_ + glower[0][2]*b2_ + glower[0][3]*b3_;
        Real b_1 = glower[1][0]*b0_ + glower[1][1]*b1_ + glower[1][2]*b2_ + glower[1][3]*b3_;
        Real b_2 = glower[2][0]*b0_ + glower[2][1]*b1_ + glower[2][2]*b2_ + glower[2][3]*b3_;
        Real b_3 = glower[3][0]*b0_ + glower[3][1]*b1_ + glower[3][2]*b2_ + glower[3][3]*b3_;
        Real b_sq = b0_*b_0 + b1_*b_1 + b2_*b_2 + b3_*b_3;

        sigma_cold = b_sq/w.d;
      }

      // apply temperature fix
      if (is_radiation_enabled_ && temperature_fix) {
        if ((sigma_cold > sigma_cold_cut_) && (rv < r_tfix_cut_)) { // different criterion can be used here
          Real log10_sfloor_local = log10(eos.sfloor1) + (log10(w.d)-log10(eos.rho1)) * (log10(eos.sfloor2)-log10(eos.sfloor1))/(log10(eos.rho2)-log10(eos.rho1));
          Real sfloor_local = pow(10.0, log10_sfloor_local);
          Real sfloor_ = fmax(eos.sfloor, sfloor_local);
          Real pgas_ = w.d*tgas_radsource_(m,k,j,i);
          Real pgas_min = fmax(eos.pfloor, sfloor_*pow(w.d, eos.gamma));
          if (pgas_ < pgas_min) {
            pgas_ = pgas_min;
            efloor_used = true;
          }
          w.e = pgas_/gm1;
        }
      } // endif temperature fix

      // apply entropy fix
      if (entropy_fix_ && !entropy_fix_turnoff_) {
        // fix the prim in strongly magnetized region or cells that fail the variable inversion
        if (c2p_failure || (sigma_cold > sigma_cold_cut_)) {
          // compute the entropy fix
          bool dfloor_used_in_fix=false, efloor_used_in_fix=false;
          bool c2p_failure_in_fix=c2p_failure;
          int iter_used_in_fix=0;
          HydPrim1D w_fix;
          w_fix.d  = w.d;
          w_fix.vx = w.vx;
          w_fix.vy = w.vy;
          w_fix.vz = w.vz;
          w_fix.e  = w.e;
          Real &s_tot = cons(m,entropyIdx,k,j,i);
          SingleC2P_IdealSRMHD_EntropyFix(u_sr, s_tot, eos, s2, b2, rpar, w_fix, w_old,
                                          dfloor_used_in_fix, efloor_used_in_fix,
                                          c2p_failure_in_fix, iter_used_in_fix);
          // entropy fix
          if (!c2p_failure_in_fix) {
            // successful entropy-fixed c2p
            w.d  = w_fix.d;
            w.e  = w_fix.e;
            w.vx = w_fix.vx;
            w.vy = w_fix.vy;
            w.vz = w_fix.vz;
            dfloor_used = dfloor_used_in_fix;
            efloor_used = efloor_used_in_fix;
            c2p_failure = c2p_failure_in_fix;
            iter_used_in_fix = iter_used;
          }
        } // endif (c2p_failure || (sigma_cold > sigma_cold_cut_))
      } // endif entropy_fix_

      // flag the cell if c2p succeeds or fails
      c2p_flag_(m,k,j,i) = !c2p_failure;
      smooth_flag_(m,k,j,i) = false;
      // try different strategy here to smooth the checkerboarding issue
      // if ((sigma_cold > sigma_cold_cut_) && (efloor_used || dfloor_used)) {
      // if (sigma_cold > sigma_cold_cut_) {
      //   // if ((sigma_cold > sigma_cold_cut_) && (rv > r_tfix_cut_)) {
      //   smooth_flag_(m,k,j,i) = true;
      // }
      if (customize_fofc_ && (sigma_cold > sigma_cold_cut_)) fofc_(m,k,j,i) = true;

      // apply velocity ceiling if necessary
      Real tmp = glower[1][1]*SQR(w.vx)
               + glower[2][2]*SQR(w.vy)
               + glower[3][3]*SQR(w.vz)
               + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
               + 2.0*glower[2][3]*w.vy*w.vz;
      Real lor = sqrt(1.0+tmp);
      if (lor > eos.gamma_max) {
        vceiling_used = true;
        Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (vceiling_used) {sumv++;}
      if (c2p_failure) {sumf++;}
      max_it = (iter_used > max_it) ? iter_used : max_it;

      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;

      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;

      // reset conserved variables if floor, ceiling, failure, or excision encountered
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure || excised) {
        MHDPrim1D w_in;
        w_in.d  = w.d;
        w_in.vx = w.vx;
        w_in.vy = w.vy;
        w_in.vz = w.vz;
        w_in.e  = w.e;
        w_in.bx = u.bx;
        w_in.by = u.by;
        w_in.bz = u.bz;

        HydCons1D u_out;
        SingleP2C_IdealGRMHD(glower, gupper, w_in, eos.gamma, u_out);
        cons(m,IDN,k,j,i) = u_out.d;
        cons(m,IM1,k,j,i) = u_out.mx;
        cons(m,IM2,k,j,i) = u_out.my;
        cons(m,IM3,k,j,i) = u_out.mz;
        cons(m,IEN,k,j,i) = u_out.e;
        u.d = u_out.d;  // (needed if there are scalars below)
      }

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nceilv_),
     Kokkos::Sum<int>(nfail_), Kokkos::Max<int>(maxit_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_vceil  += nceilv_;
    pmy_pack->pmesh->ecounter.neos_fail   += nfail_;
    pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;
  }

  // fallback for the failure of variable inversion that uses the average of the valid primitives in adjacent cells
  if (cellavg_fix_turn_on_) {
    par_for("adjacent_cellavg_fix", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Check if the cell is in excised region
      bool excised = false;
      if (use_excise) {
        if (excision_floor_(m,k,j,i)) {
          excised = true;
        }
        if (only_testfloors) {
          if (excision_flux_(m,k,j,i)) {
            excised = true;
          }
        }
      }

      // Assign fallback state if inversion fails
      if ((!c2p_flag_(m,k,j,i) || smooth_flag_(m,k,j,i)) && !(excised)) {
        // Set indices around the problematic cell
        int km1 = (k-1 < kl) ? kl : k-1;
        int kp1 = (k+1 > ku) ? ku : k+1;
        int jm1 = (j-1 < jl) ? jl : j-1;
        int jp1 = (j+1 > ju) ? ju : j+1;
        int im1 = (i-1 < il) ? il : i-1;
        int ip1 = (i+1 > iu) ? iu : i+1;

        // try to identify checkboard region
        if (smooth_flag_(m,k,j,i)) {
          Real diff_large = 1.e4;
          Real diff_small = 1.e1;

          Real diff_u = fabs(prim(m,IEN,kp1,j,i)/prim(m,IEN,k,j,i) - 1);
          Real diff_d = fabs(prim(m,IEN,km1,j,i)/prim(m,IEN,k,j,i) - 1);
          Real diff_r = fabs(prim(m,IEN,k,jp1,i)/prim(m,IEN,k,j,i) - 1);
          Real diff_l = fabs(prim(m,IEN,k,jm1,i)/prim(m,IEN,k,j,i) - 1);
          Real diff_f = fabs(prim(m,IEN,k,j,ip1)/prim(m,IEN,k,j,i) - 1);
          Real diff_b = fabs(prim(m,IEN,k,j,im1)/prim(m,IEN,k,j,i) - 1);

          Real diff_ur = fabs(prim(m,IEN,kp1,jp1,i)/prim(m,IEN,k,j,i) - 1);
          Real diff_ul = fabs(prim(m,IEN,kp1,jm1,i)/prim(m,IEN,k,j,i) - 1);
          Real diff_uf = fabs(prim(m,IEN,kp1,j,ip1)/prim(m,IEN,k,j,i) - 1);
          Real diff_ub = fabs(prim(m,IEN,kp1,j,im1)/prim(m,IEN,k,j,i) - 1);
          Real diff_dr = fabs(prim(m,IEN,km1,jp1,i)/prim(m,IEN,k,j,i) - 1);
          Real diff_dl = fabs(prim(m,IEN,km1,jm1,i)/prim(m,IEN,k,j,i) - 1);
          Real diff_df = fabs(prim(m,IEN,km1,j,ip1)/prim(m,IEN,k,j,i) - 1);
          Real diff_db = fabs(prim(m,IEN,km1,j,im1)/prim(m,IEN,k,j,i) - 1);
          Real diff_rf = fabs(prim(m,IEN,k,jp1,ip1)/prim(m,IEN,k,j,i) - 1);
          Real diff_rb = fabs(prim(m,IEN,k,jp1,im1)/prim(m,IEN,k,j,i) - 1);
          Real diff_lf = fabs(prim(m,IEN,k,jm1,ip1)/prim(m,IEN,k,j,i) - 1);
          Real diff_lb = fabs(prim(m,IEN,k,jm1,im1)/prim(m,IEN,k,j,i) - 1);

          Real diff_urf = fabs(prim(m,IEN,kp1,jp1,ip1)/prim(m,IEN,k,j,i) - 1);
          Real diff_urb = fabs(prim(m,IEN,kp1,jp1,im1)/prim(m,IEN,k,j,i) - 1);
          Real diff_ulf = fabs(prim(m,IEN,kp1,jm1,ip1)/prim(m,IEN,k,j,i) - 1);
          Real diff_ulb = fabs(prim(m,IEN,kp1,jm1,im1)/prim(m,IEN,k,j,i) - 1);
          Real diff_dlb = fabs(prim(m,IEN,km1,jm1,im1)/prim(m,IEN,k,j,i) - 1);
          Real diff_dlf = fabs(prim(m,IEN,km1,jm1,ip1)/prim(m,IEN,k,j,i) - 1);
          Real diff_drb = fabs(prim(m,IEN,km1,jp1,im1)/prim(m,IEN,k,j,i) - 1);
          Real diff_drf = fabs(prim(m,IEN,km1,jp1,ip1)/prim(m,IEN,k,j,i) - 1);

          bool is_checkboard1 = (diff_u > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_d > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_r > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_l > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_f > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_b > diff_large);

          is_checkboard1 = is_checkboard1 && (diff_urf > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_urb > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_ulf > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_ulb > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_dlb > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_dlf > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_drb > diff_large);
          is_checkboard1 = is_checkboard1 && (diff_drf > diff_large);

          is_checkboard1 = is_checkboard1 && (diff_ur < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_ul < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_uf < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_ub < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_dr < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_dl < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_df < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_db < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_rf < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_rb < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_lf < diff_small);
          is_checkboard1 = is_checkboard1 && (diff_lb < diff_small);

          bool is_checkboard2 = (diff_u < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_d < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_r < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_l < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_f < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_b < diff_small);

          is_checkboard2 = is_checkboard2 && (diff_urf < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_urb < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_ulf < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_ulb < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_dlb < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_dlf < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_drb < diff_small);
          is_checkboard2 = is_checkboard2 && (diff_drf < diff_small);

          is_checkboard2 = is_checkboard2 && (diff_ur > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_ul > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_uf > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_ub > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_dr > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_dl > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_df > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_db > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_rf > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_rb > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_lf > diff_large);
          is_checkboard2 = is_checkboard2 && (diff_lb > diff_large);

          if (is_checkboard1 || is_checkboard2) {
            smooth_flag_(m,k,j,i) = true;
          } else {
            smooth_flag_(m,k,j,i) = false;
          }
        }

        // initialize primitive fallback
        MHDPrim1D w;
        w.d = 0.0; w.vx = 0.0; w.vy = 0.0; w.vz = 0.0; w.e = 0.0;
        // Load cell-centered fields
        if ((only_testfloors) || (c2p_test_)) {
          // use input CC fields if only testing floors with FOFC
          w.bx = bcc(m,IBX,k,j,i);
          w.by = bcc(m,IBY,k,j,i);
          w.bz = bcc(m,IBZ,k,j,i);
        } else {
          // else use simple linear average of face-centered fields
          w.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
          w.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
          w.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
        }

        // Add the primitives of valid adjacent cells
        int n_count = 0;
        for (int kk=km1; kk<=kp1; ++kk) {
          for (int jj=jm1; jj<=jp1; ++jj) {
            for (int ii=im1; ii<=ip1; ++ii) {
              if (c2p_flag_(m,kk,jj,ii) && !(excised)) {
                w.d  = w.d  + prim(m,IDN,kk,jj,ii);
                w.vx = w.vx + prim(m,IVX,kk,jj,ii);
                w.vy = w.vy + prim(m,IVY,kk,jj,ii);
                w.vz = w.vz + prim(m,IVZ,kk,jj,ii);
                w.e  = w.e  + prim(m,IEN,kk,jj,ii);
                n_count += 1;
              } // endif c2p_flag_(m,kk,jj,ii)
            } // endfor ii
          } // endfor jj
        } // endfor kk

        // Assign the fallback state
        if (n_count == 0) {
          w.d  = w0_old_(m,IDN,k,j,i);
          w.vx = w0_old_(m,IVX,k,j,i);
          w.vy = w0_old_(m,IVY,k,j,i);
          w.vz = w0_old_(m,IVZ,k,j,i);
          w.e  = w0_old_(m,IEN,k,j,i);
        } else {
          w.d  = w.d/n_count;
          w.vx = w.vx/n_count;
          w.vy = w.vy/n_count;
          w.vz = w.vz/n_count;
          w.e  = w.e/n_count;
        }

        if (!c2p_flag_(m,k,j,i)) { // if variable inversion fails
          prim(m,IDN,k,j,i) = w.d;
          prim(m,IVX,k,j,i) = w.vx;
          prim(m,IVY,k,j,i) = w.vy;
          prim(m,IVZ,k,j,i) = w.vz;
          prim(m,IEN,k,j,i) = w.e;
        } else if (smooth_flag_(m,k,j,i)) { // if extra smooth is needed
          prim(m,IDN,k,j,i) = w.d;
          prim(m,IVX,k,j,i) = w.vx;
          prim(m,IVY,k,j,i) = w.vy;
          prim(m,IVZ,k,j,i) = w.vz;
          prim(m,IEN,k,j,i) = w.e;
        }

        // Extract components of metric
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
        HydCons1D u;
        SingleP2C_IdealGRMHD(glower, gupper, w, eos.gamma, u);
        cons(m,IDN,k,j,i) = u.d;
        cons(m,IM1,k,j,i) = u.mx;
        cons(m,IM2,k,j,i) = u.my;
        cons(m,IM3,k,j,i) = u.mz;
        cons(m,IEN,k,j,i) = u.e;
      } // endif (!c2p_flag_(m,k,j,i) && !(excised))
    }); // end_par_for 'adjacent_cellavg_fix'
  } // endif (cellavg_fix_turn_on_)

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  Operates over range of cells
//! given in argument list.

void IdealGRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                            DvceArray5D<Real> &cons, const int il, const int iu,
                            const int jl, const int ju, const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;
  auto &entropy_fix_ = pmy_pack->pmhd->entropy_fix;
  int n_var = (entropy_fix_) ? nmhd+nscal-1 : nmhd+nscal;

  par_for("grmhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Extract components of metric
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

    // Load single state of primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);

    // store conserved quantities in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nmhd; n<n_var; ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
