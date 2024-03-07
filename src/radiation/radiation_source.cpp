//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_source.cpp

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "units/units.hpp"
#include "radiation.hpp"

#include "radiation/radiation_tetrad.hpp"
#include "radiation/radiation_opacities.hpp"

namespace radiation {

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root);

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::AddRadiationSourceTerm(Driver *pdriver, int stage)
// \brief Add implicit radiation source term.  Based off of @c-white and @yanfeij's gr_rad
// branch, radiation/coupling/emission.cpp commit be7f84565b.

TaskStatus Radiation::AddRadiationSourceTerm(Driver *pdriver, int stage) {
  // Return if radiation source term disabled
  if (!(rad_source)) {
    return TaskStatus::complete;
  }

  // Extract indices, size data, hydro/mhd/units flags, and coupling flags
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = prgeo->nangles - 1;
  auto &size = pmy_pack->pmb->mb_size;
  bool &is_hydro_enabled_ = is_hydro_enabled;
  bool &is_mhd_enabled_ = is_mhd_enabled;
  bool &are_units_enabled_ = are_units_enabled;
  bool &is_compton_enabled_ = is_compton_enabled;
  bool &fixed_fluid_ = fixed_fluid;
  bool &affect_fluid_ = affect_fluid;

  // Extract variables and flags for ad hoc fixes
  bool &update_vel_in_rad_source_ = update_vel_in_rad_source;
  bool &correct_vel_in_rad_source_ = correct_vel_in_rad_source;
  bool &use_old_coupling_in_rad_source_ = use_old_coupling_in_rad_source;
  bool &compton_second_order_correction_ = compton_second_order_correction;
  bool &compton_use_artificial_mask_ = compton_use_artificial_mask;
  bool &temperature_fix_turn_on_ = temperature_fix_turn_on;
  // auto &cellavg_rad_source_ = cellavg_rad_source;
  auto &tgas_radsource_ = tgas_radsource; // for saving final gas temperature
  Real sigma_cold_cut_ = (is_mhd_enabled_) ? pmy_pack->pmhd->sigma_cold_cut : 1.e1;
  // bool turn_on_sao_radsrc_ = false;
  auto &correct_radsrc_opacity_ = correct_radsrc_opacity;
  Real &dfloor_opacity_ = dfloor_opacity;
  Real &floor_planck_ = floor_planck;

  // Extract coordinate/excision data
  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;
  bool &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = n_0_floor;

  // Extract radiation constant and units
  Real &arad_ = arad;
  Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
  Real mean_mol_weight_ = 1.0;
  Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
  Real inv_t_electron_ = 1.0;
  Real telec_cgs_ = pmy_pack->punit->electron_rest_mass_energy_cgs;
  if (are_units_enabled_) {
    density_scale_ = pmy_pack->punit->density_cgs();
    temperature_scale_ = pmy_pack->punit->temperature_cgs();
    length_scale_ = pmy_pack->punit->length_cgs();
    mean_mol_weight_ = pmy_pack->punit->mu();
    rosseland_coef_ = pmy_pack->punit->rosseland_coef_cgs;
    planck_minus_rosseland_coef_ = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
    inv_t_electron_ = temperature_scale_/pmy_pack->punit->electron_rest_mass_energy_cgs;
  }

  // Extract adiabatic index and velocity ceiling
  Real gm1, dfloor, pfloor;
  Real v_sq_max = 1. - 1.e-12;
  if (is_hydro_enabled_) {
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
    v_sq_max = 1. - 1./SQR(pmy_pack->phydro->peos->eos_data.gamma_max);
    dfloor = pmy_pack->phydro->peos->eos_data.dfloor;
    pfloor = pmy_pack->phydro->peos->eos_data.pfloor;
  } else if (is_mhd_enabled_) {
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
    v_sq_max = 1. - 1./SQR(pmy_pack->pmhd->peos->eos_data.gamma_max);
    dfloor = pmy_pack->pmhd->peos->eos_data.dfloor;
    pfloor = pmy_pack->pmhd->peos->eos_data.pfloor;
  }

  // Extract flag and index for entropy fix
  bool entropy_fix_ = false;
  int entropyIdx = -1;
  if (is_mhd_enabled_) {
    // note that entropy fix can only be turned on in GRMHD
    entropy_fix_ = pmy_pack->pmhd->entropy_fix;
    int nmhd = pmy_pack->pmhd->nmhd;
    int nscal = pmy_pack->pmhd->nscalars;
    if (entropy_fix_) entropyIdx = nmhd+nscal-1;
  }

  // Extract radiation, radiation frame, and radiation angular mesh data
  auto &i0_ = i0;
  Real &kappa_a_ = kappa_a;
  Real &kappa_s_ = kappa_s;
  Real &kappa_p_ = kappa_p;
  bool &power_opacity_ = power_opacity;
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &solid_angles_ = prgeo->solid_angles;

  // Extract hydro/mhd quantities
  DvceArray5D<Real> u0_, w0_, w_noupdate_;
  if (is_hydro_enabled_) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
    w_noupdate_ = pmy_pack->phydro->w0;
  } else if (is_mhd_enabled_) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
    w_noupdate_ = update_vel_in_rad_source_ ? pmy_pack->pmhd->w0 : w_noupdate;
  }

  // Extract timestep
  Real dt_ = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Call ConsToPrim over active zones prior to source term application
  DvceArray5D<Real> bcc0_;
  if (!(fixed_fluid_)) {
    if (is_hydro_enabled_) {
      pmy_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is,ie,js,je,ks,ke);
    } else if (is_mhd_enabled_) {
      auto &b0_ = pmy_pack->pmhd->b0;
      bcc0_ = pmy_pack->pmhd->bcc0;
      pmy_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,false,is,ie,js,je,ks,ke);
    }
  }

  // compute implicit source term
  par_for("radiation_source",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

    // fluid state
    Real wdn = w0_(m,IDN,k,j,i);
    Real &wvx = update_vel_in_rad_source_ ? w0_(m,IVX,k,j,i) : w_noupdate_(m,IVX,k,j,i);
    Real &wvy = update_vel_in_rad_source_ ? w0_(m,IVY,k,j,i) : w_noupdate_(m,IVY,k,j,i);
    Real &wvz = update_vel_in_rad_source_ ? w0_(m,IVZ,k,j,i) : w_noupdate_(m,IVZ,k,j,i);
    Real wen = w0_(m,IEN,k,j,i);

    // compute sigma_cold
    Real sigma_cold = 0.0;
    if (is_mhd_enabled_) {
      Real qq = glower[1][1]*wvx*wvx +2.0*glower[1][2]*wvx*wvy +2.0*glower[1][3]*wvx*wvz
              + glower[2][2]*wvy*wvy +2.0*glower[2][3]*wvy*wvz
              + glower[3][3]*wvz*wvz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real u0_norm = sqrt(1.0 + qq);
      Real u0 = u0_norm / alpha;
      Real u1 = wvx - alpha * u0_norm * gupper[0][1];
      Real u2 = wvy - alpha * u0_norm * gupper[0][2];
      Real u3 = wvz - alpha * u0_norm * gupper[0][3];

      // lower vector indices
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // calculate 4-magnetic field
      auto &bccx = bcc0_(m,IBX,k,j,i);
      auto &bccy = bcc0_(m,IBY,k,j,i);
      auto &bccz = bcc0_(m,IBZ,k,j,i);
      Real b0_ = u_1*bccx + u_2*bccy + u_3*bccz;
      Real b1_ = (bccx + b0_ * u1) / u0;
      Real b2_ = (bccy + b0_ * u2) / u0;
      Real b3_ = (bccz + b0_ * u3) / u0;

      // lower vector indices
      Real b_0 = glower[0][0]*b0_ + glower[0][1]*b1_ + glower[0][2]*b2_ + glower[0][3]*b3_;
      Real b_1 = glower[1][0]*b0_ + glower[1][1]*b1_ + glower[1][2]*b2_ + glower[1][3]*b3_;
      Real b_2 = glower[2][0]*b0_ + glower[2][1]*b1_ + glower[2][2]*b2_ + glower[2][3]*b3_;
      Real b_3 = glower[3][0]*b0_ + glower[3][1]*b1_ + glower[3][2]*b2_ + glower[3][3]*b3_;
      Real b_sq = b0_*b_0 + b1_*b_1 + b2_*b_2 + b3_*b_3;

      sigma_cold = b_sq/wdn;
    } // endif (is_mhd_enabled_)

    // apply cell-averaged profile to compute the radiation source terms
    // if (cellavg_rad_source_ && !rad_mask_(m,k,j,i)) {
    //   if (sigma_cold > sigma_cold_cut_) {
    //     int km1 = (k-1 < ks) ? ks : k-1;
    //     int kp1 = (k+1 > ke) ? ke : k+1;
    //     int jm1 = (j-1 < js) ? js : j-1;
    //     int jp1 = (j+1 > je) ? je : j+1;
    //     int im1 = (i-1 < is) ? is : i-1;
    //     int ip1 = (i+1 > ie) ? ie : i+1;
    //     // averaging adjecent cells
    //     Real wdn_avg = 0.0;
    //     Real wen_avg = 0.0;
    //     int n_count = 0;
    //     for (int kk=km1; kk<=kp1; ++kk) {
    //       for (int jj=jm1; jj<=jp1; ++jj) {
    //         for (int ii=im1; ii<=ip1; ++ii) {
    //           if (!rad_mask_(m,k,j,i)) {
    //             wdn_avg += w0_(m,IDN,kk,jj,ii);
    //             wen_avg += w0_(m,IEN,kk,jj,ii);
    //             n_count += 1;
    //           } // endif c2p_flag_(m,kk,jj,ii)
    //         } // endfor ii
    //       } // endfor jj
    //     } // endfor kk
    //     if (n_count == 0) {
    //       wdn_avg = w0_(m,IDN,k,j,i);
    //       wen_avg = w0_(m,IEN,k,j,i);
    //     } else {
    //       wdn_avg = wdn_avg/n_count;
    //       wen_avg = wen_avg/n_count;
    //     }
    //     // assign cell-averaged quantity for source term calculation
    //     // wdn = wdn_avg;
    //     // wen = wen_avg;
    //   } // endif (sigma_cold > sigma_cold_cut_)
    // } // endif (cellavg_rad_source_ && !rad_mask_(m,k,j,i))

    // derived quantities
    Real pgas = gm1*wen;
    Real tgas = pgas/wdn;
    Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
           + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz
           + glower[3][3]*wvz*wvz;
    Real gamma = sqrt(1.0 + q);
    Real u0 = gamma/alpha;

    // radiation internal energy change
    // Real delta_erad_f = 0.0;

    // set opacities
    Real sigma_a, sigma_s, sigma_p;
    Real wdn_opacity = fmax(wdn-dfloor, dfloor_opacity_);
    if (correct_radsrc_opacity_ && (sigma_cold > sigma_cold_cut_)) {
      wdn_opacity = dfloor_opacity_;
    }
    OpacityFunction(wdn_opacity, density_scale_,
                    tgas, temperature_scale_,
                    length_scale_, gm1, mean_mol_weight_,
                    power_opacity_, rosseland_coef_, planck_minus_rosseland_coef_,
                    kappa_a_, kappa_s_, kappa_p_,
                    sigma_a, sigma_s, sigma_p);
    // sigma_p = fmax(sigma_p, floor_planck_-sigma_a);
    Real dtcsiga = dt_*sigma_a;
    Real dtcsigs = dt_*sigma_s;
    Real dtcsigp = dt_*sigma_p;
    Real dtaucsiga = dtcsiga/u0;
    Real dtaucsigs = dtcsigs/u0;
    Real dtaucsigp = dtcsigp/u0;

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

    // coordinate component n^0
    Real n0 = tt(m,0,0,k,j,i);

    // Correct velocity in radiation-dominated regime
    if (correct_vel_in_rad_source_) {
      // NOTE(@lzhang): In radiation-dominated regime, the gas-radiation momentum
      // coupling without accounting the sharp change of the gas velocity can
      // result in overestimated high gas temperature because the overestimated
      // velocity leads to the low density. This turns out to be extremely dangerous
      // because it can generate enormous radiation through the thermal coupling or
      // Compton process.

      // calculate radiation energy density in fluid frame
      Real erad_f_ = 0.0;
      Real omega_hat_tot = 0.0; Real omega_cm_tot = 0.0;
      for (int n=0; n<=nang1; ++n) {
        // compute coordinate normal components
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);

        // compute quantites in fluid frame
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));

        Real i0_cm = i0_(m,n,k,j,i)/(n0*n_0)*SQR(SQR(n0_cm));
        Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
        omega_hat_tot += solid_angles_.d_view(n);
        omega_cm_tot += omega_cm;
        erad_f_ += i0_cm*omega_cm;
      }
      erad_f_ *= omega_hat_tot/omega_cm_tot;

      // apply velocity correction if radiation is dominated
      if (erad_f_ > wdn + wen) {
        // compute radiation moments in terad frame
        Real rr_tet00 = 0.0;
        Real rr_tet01 = 0.0; Real rr_tet02 = 0.0; Real rr_tet03 = 0.0;
        Real rr_tet11 = 0.0; Real rr_tet22 = 0.0; Real rr_tet33 = 0.0;
        Real rr_tet12 = 0.0; Real rr_tet13 = 0.0; Real rr_tet23 = 0.0;
        for (int n=0; n<=nang1; ++n) {
          // tetrad normal components
          Real nh0 = nh_c_.d_view(n,0);
          Real nh1 = nh_c_.d_view(n,1);
          Real nh2 = nh_c_.d_view(n,2);
          Real nh3 = nh_c_.d_view(n,3);

          // coordinate normal components
          Real n_0 = tc(m,0,0,k,j,i)*nh0 + tc(m,1,0,k,j,i)*nh1 + tc(m,2,0,k,j,i)*nh2 + tc(m,3,0,k,j,i)*nh3;
          Real n_1 = tc(m,0,1,k,j,i)*nh0 + tc(m,1,1,k,j,i)*nh1 + tc(m,2,1,k,j,i)*nh2 + tc(m,3,1,k,j,i)*nh3;
          Real n_2 = tc(m,0,2,k,j,i)*nh0 + tc(m,1,2,k,j,i)*nh1 + tc(m,2,2,k,j,i)*nh2 + tc(m,3,2,k,j,i)*nh3;
          Real n_3 = tc(m,0,3,k,j,i)*nh0 + tc(m,1,3,k,j,i)*nh1 + tc(m,2,3,k,j,i)*nh2 + tc(m,3,3,k,j,i)*nh3;

          // radiation moments in terad frame
          rr_tet00 += (        i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet01 += (    nh1*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet02 += (    nh2*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet03 += (    nh3*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet11 += (nh1*nh1*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet22 += (nh2*nh2*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet33 += (nh3*nh3*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet12 += (nh1*nh2*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet13 += (nh1*nh3*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet23 += (nh2*nh3*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
        } // endfor n

        // calculate radiation velocity in tetrad frame
        Real vrad_tet1 = rr_tet01 / rr_tet00;
        Real vrad_tet2 = rr_tet02 / rr_tet00;
        Real vrad_tet3 = rr_tet03 / rr_tet00;
        Real vrad_sq = SQR(vrad_tet1) + SQR(vrad_tet2) + SQR(vrad_tet3);
        if (vrad_sq > v_sq_max) {
          Real ratio = sqrt(v_sq_max / vrad_sq);
          vrad_tet1 *= ratio;
          vrad_tet2 *= ratio;
          vrad_tet3 *= ratio;
          vrad_sq = v_sq_max;
        }
        Real urad_tet0 = 1.0 / sqrt(1.0 - vrad_sq);
        Real urad_tet1 = urad_tet0 * vrad_tet1;
        Real urad_tet2 = urad_tet0 * vrad_tet2;
        Real urad_tet3 = urad_tet0 * vrad_tet3;

        // calculate current fluid momentum
        Real wgas = wdn + wen + pgas;
        Real mgas_tet1 = wgas * u_tet[0] * u_tet[1];
        Real mgas_tet2 = wgas * u_tet[0] * u_tet[2];
        Real mgas_tet3 = wgas * u_tet[0] * u_tet[3];

        // calculate fluid momentum if accelerated to radiation frame
        Real mgas_rad_tet1 = wgas * urad_tet0 * urad_tet1;
        Real mgas_rad_tet2 = wgas * urad_tet0 * urad_tet2;
        Real mgas_rad_tet3 = wgas * urad_tet0 * urad_tet3;

        // calculate the gas-radiation coupling force
        Real chi_p = wdn * (kappa_p_ + kappa_a_);
        Real chi_s = wdn * kappa_s_;
        Real chi_a = wdn * (kappa_a_ + kappa_s_);
        Real emissivity = chi_p*arad_*SQR(SQR(tgas)) + chi_s*erad_f_;
        if (is_compton_enabled_) {
          Real trad = sqrt(sqrt(erad_f_/arad_));
          emissivity += chi_s*4*(tgas-trad)*inv_t_electron_*erad_f_;
          if (compton_second_order_correction_) emissivity += chi_s*16*SQR(tgas*inv_t_electron_)*erad_f_;
        }
        Real gg_tet1 = -emissivity*u_tet[1] - chi_a*(-u_tet[0]*rr_tet01 + u_tet[1]*rr_tet11 + u_tet[2]*rr_tet12 + u_tet[3]*rr_tet13);
        Real gg_tet2 = -emissivity*u_tet[2] - chi_a*(-u_tet[0]*rr_tet02 + u_tet[1]*rr_tet12 + u_tet[2]*rr_tet22 + u_tet[3]*rr_tet23);
        Real gg_tet3 = -emissivity*u_tet[3] - chi_a*(-u_tet[0]*rr_tet03 + u_tet[1]*rr_tet13 + u_tet[2]*rr_tet23 + u_tet[3]*rr_tet33);

        // estimate change in fluid momentum from source terms
        Real dmgas_tet1 = gg_tet1 * dt_ / u_tet[0];
        Real dmgas_tet2 = gg_tet2 * dt_ / u_tet[0];
        Real dmgas_tet3 = gg_tet3 * dt_ / u_tet[0];

        // estimate new fluid velocity
        Real frac1 = (mgas_rad_tet1==mgas_tet1) ? 0.0 : dmgas_tet1 / (mgas_rad_tet1 - mgas_tet1);
        Real frac2 = (mgas_rad_tet2==mgas_tet2) ? 0.0 : dmgas_tet2 / (mgas_rad_tet2 - mgas_tet2);
        Real frac3 = (mgas_rad_tet3==mgas_tet3) ? 0.0 : dmgas_tet3 / (mgas_rad_tet3 - mgas_tet3);
        frac1 = fmin(fmax(frac1, 0.0), 1.0);
        frac2 = fmin(fmax(frac2, 0.0), 1.0);
        frac3 = fmin(fmax(frac3, 0.0), 1.0);
        u_tet[1] = (1.0-frac1)*u_tet[1] + frac1*urad_tet1;
        u_tet[2] = (1.0-frac2)*u_tet[2] + frac2*urad_tet2;
        u_tet[3] = (1.0-frac3)*u_tet[3] + frac3*urad_tet3;
        u_tet[0] = sqrt(1.0 + SQR(u_tet[1]) + SQR(u_tet[2]) + SQR(u_tet[3]));
      } // endif (erad_f_ > wdn + wen)
    } // endif (correct_vel_in_rad_source_)

    // Calculate polynomial coefficients
    Real wght_sum = 0.0;
    Real suma1 = 0.0;
    Real suma2 = 0.0;
    Real erad_f_ = 0.0;
    for (int n=0; n<=nang1; ++n) {
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                 tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
      Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                    u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
      Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
      Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
      Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
      Real vncsigma2 = n0_cm*vncsigma;
      Real ir_weight = intensity_cm*omega_cm;
      wght_sum += omega_cm;
      suma1 += omega_cm*vncsigma2;
      suma2 += ir_weight*n0*vncsigma;
      erad_f_ += ir_weight;
    }
    suma1 /= wght_sum;
    suma2 /= wght_sum;
    Real suma3 = suma1*(dtcsigs - dtcsigp);
    suma1 *= (dtcsiga + dtcsigp);
    erad_f_ /= wght_sum;

    // compute coefficients
    Real coef[2];
    if (use_old_coupling_in_rad_source_) {
      coef[1] = (dtaucsiga+dtaucsigp-(dtaucsiga+dtaucsigp)*suma1/(1.0-suma3))*arad_*gm1/wdn;
      coef[0] = -tgas-(dtaucsiga+dtaucsigp)*suma2*gm1/(wdn*(1.0-suma3));
    } else {
      coef[1] = gm1/wdn * suma1/(1-suma3) * arad_;
      coef[0] = gm1/wdn*(suma2/(1-suma3)-erad_f_) - tgas;
    }

    // Calculate new gas temperature
    Real tgasnew = tgas;
    bool badcell = false;
    if (fabs(coef[1]) > 1.0e-20) {
      bool flag = FourthPolyRoot(coef[1], coef[0], tgasnew);
      if (!(flag) || !(isfinite(tgasnew))) {
        badcell = true;
        tgasnew = tgas;
      }
    } else {
      tgasnew = -coef[0];
    }

    // Update the specific intensity
    if (!(badcell)) {
      // Calculate emission coefficient and updated jr_cm
      Real emission = arad_*SQR(SQR(tgasnew));
      Real jr_cm = (suma1*emission + suma2)/(1.0 - suma3);
      Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
      for (int n=0; n<=nang1; ++n) {
        // compute coordinate normal components
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
        Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
        Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);

        // compute moments before coupling
        m_old[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
        m_old[1] += (n_1*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
        m_old[2] += (n_2*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
        m_old[3] += (n_3*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));

        // update intensity
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
        Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        Real di_cm = ( ((dtcsigs-dtcsigp)*jr_cm
                      + (dtcsiga+dtcsigp)*emission
                      - (dtcsigs+dtcsiga)*intensity_cm)*vncsigma2 );
        i0_(m,n,k,j,i) = n0*n_0*fmax(i0_(m,n,k,j,i)/(n0*n_0) +
                                     di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);

        // compute moments after coupling
        m_new[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
        m_new[1] += (n_1*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
        m_new[2] += (n_2*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
        m_new[3] += (n_3*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));

        // handle excision
        // NOTE(@pdmullen): The below zeroes all intensities within rks <= r_excision and
        // zeroes intensities within angles where n_0 is about zero. When Compton is
        // enabled, we delay the n_0_floor excision so that intensites updated via
        // absorption and scattering inform the Compton update
        if (excise) {
          bool apply_excision = (rad_mask_(m,k,j,i) ||
                                 (!(is_compton_enabled_) && fabs(n_0) < n_0_floor_));
          if (apply_excision) { i0_(m,n,k,j,i) = 0.0; }
        }
      }

      // update conserved fluid variables
      if (affect_fluid_) {
        u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
        u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
        u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
        u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
      }

      // update total entropy if entropy fix is enabled
      if (entropy_fix_) {
        // total entropy must be assigned as the last passive scalar
        Real src0 = m_new[0] - m_old[0];
        Real src1 = m_new[1] - m_old[1];
        Real src2 = m_new[2] - m_old[2];
        Real src3 = m_new[3] - m_old[3];

        // compute coord-frame 4-velocity
        Real u1  = wvx - alpha * gamma * gupper[0][1];;
        Real u2  = wvy - alpha * gamma * gupper[0][2];;
        Real u3  = wvz - alpha * gamma * gupper[0][3];;
        Real rho = wdn;

        // add source term to total entropy
        Real src = gm1 * (u0*src0+u1*src1+u2*src2+u3*src3) / std::pow(rho, gm1);
        u0_(m,entropyIdx,k,j,i) += src;
      } // endif entropy_fix_

    } // endif !badcell

    // record radiation internal energy change
    // if (turn_on_sao_radsrc_) delta_erad_f = -wdn*(tgasnew-tgas)/gm1;

    // compton scattering
    if (is_compton_enabled_) {
      // use partially updated gas temperature
      tgas = tgasnew;

      // artificial mask for applying compton term
      Real scale_fac = 1.0;
      if (compton_use_artificial_mask_) {
        // scale_fac = 1./(1.+exp(-10.*(log10(wdn)+4.5)));
        scale_fac = 1. - 1./(1.+exp(-10.*(log10(tgas*temperature_scale_)-log10(telec_cgs_))));
      }

      // compute polynomial coefficients using partially updated gas temp and intensity
      suma1 = 0.0;
      Real jr_cm = 0.0;
      for (int n=0; n<=nang1; ++n) {
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                   tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real wght_cm = solid_angles_.d_view(n)/SQR(n0_cm)/wght_sum;
        Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
        Real ir_weight = intensity_cm*wght_cm;
        jr_cm += ir_weight; // note this is actually er_cm
        suma1 += (n0_cm/n0)*4.0*dtcsigs*inv_t_electron_*wght_cm;
      }
      suma2 = (use_old_coupling_in_rad_source_) ? 4.0*dtaucsigs*inv_t_electron_*gm1/wdn : suma1*gm1/wdn;
      suma1 *= scale_fac;
      suma2 *= scale_fac;

      Real sumb3 = 1.0;
      if (compton_second_order_correction_) {
        Real sumb1 = suma1/inv_t_electron_;
        Real sumb2 = sumb1/suma2;
        Real sumb3 = wdn*tgas/(gm1*sumb1*jr_cm) + sqrt(sqrt(jr_cm/arad_))*inv_t_electron_;
        sumb3 = 1.0 + 4*sumb3 / SQR(sumb2/(sumb1*jr_cm) + 1.0);
        sumb3 = 1.0 + 0.5*(fmax(sqrt(sumb3), 1.0) - 1.0);
      }

      // compute partially updated radiation temperature
      Real trad = sqrt(sqrt(jr_cm/arad_));
      const bool temp_equil = (fabs(trad - tgas) < 1.0e-12);

      // Calculate new gas temperature due to Compton
      Real tradnew = trad;
      badcell = false;
      if (!(temp_equil)) {
        if (compton_second_order_correction_) {
          Real a2_jr = suma2*jr_cm;
          coef[1] = sumb3*(1.0 + suma2*jr_cm)/(suma1*jr_cm)*arad_;
          coef[0] = -sumb3*(1.0 + a2_jr)/suma1 - tgas*(1.0 + (sumb3-1)*(1.0+1./a2_jr));
        } else {
          coef[1] = (1.0 + suma2*jr_cm)/(suma1*jr_cm)*arad_;
          coef[0] = -(1.0 + suma2*jr_cm)/suma1 - tgas;
        }
        bool flag = FourthPolyRoot(coef[1], coef[0], tradnew);
        if (!(flag) || !(isfinite(tradnew))) {
          badcell = true;
        }

        // if (fabs(coef[1]) > 1.0e-20) {
        //   bool flag = FourthPolyRoot(coef[1], coef[0], tradnew);
        //   if (!(flag) || !(isfinite(tradnew))) {
        //     badcell = true;
        //     tradnew = trad;
        //   }
        // } else {
        //   tradnew = -coef[0];
        // }
      }

      // Update the specific intensity
      if (!(badcell) && !(temp_equil)) {
        // Compute updated gas temperature
        if (use_old_coupling_in_rad_source_) tgasnew = (arad_*SQR(SQR(tradnew)) - jr_cm)/(suma1*jr_cm) + tradnew;
        else tgasnew = -(arad_*SQR(SQR(tradnew)) - jr_cm)*gm1/wdn + tgas;

        Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
        for (int n=0; n<=nang1; ++n) {
          // compute coordinate normal components
          Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
          Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
          Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);

          // compute moments before coupling
          m_old[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
          m_old[1] += (n_1*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
          m_old[2] += (n_2*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
          m_old[3] += (n_3*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));

          // update intensity
          Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                        u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
          Real di_cm = (n0_cm/n0)*dtcsigs*4.0*jr_cm*inv_t_electron_*(tgasnew - tradnew);
          if (compton_second_order_correction_) di_cm += (n0_cm/n0)*dtcsigs*jr_cm*16.0*SQR(tgasnew*inv_t_electron_);
          di_cm *= scale_fac;

          i0_(m,n,k,j,i) = n0*n_0*fmax(i0_(m,n,k,j,i)/(n0*n_0) +
                                       di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);

          // compute moments after coupling
          m_new[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
          m_new[1] += (n_1*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
          m_new[2] += (n_2*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
          m_new[3] += (n_3*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));

          // try the ad hoc fix for the regime of high tgas + high radiation pressure
          // if (tgas*inv_t_electron_ > 1) {
          //   Real part1 = suma1/inv_t_electron_ * (arad_*SQR(SQR(tradnew)));
          //   Real part2 = wdn/gm1/inv_t_electron_;
          //   Real frac = part1/(part1+part2);
          //   tgasnew = frac*tradnew + (1-frac)*tgas;
          // }

          // record radiation internal energy change
          // w0_(m,entropyIdx,k,j,i) = tgasnew; // temporarily for sanity check, REMOVE LATER!!!
          // if (turn_on_sao_radsrc_) {
          //   Real tgas_update = tgasnew;
          //   if (wdn < 1e-5) { // only modify it for low-density region
          //     delta_erad_f += -wdn*(tgasnew-tgas)/gm1;
          //     Real erad_f_new = arad_*SQR(SQR(tradnew));
          //     if (erad_f_new == 0) {
          //       tgas_update = sqrt(sqrt(u_tet[0]*delta_erad_f/(arad_*(dtcsigp+dtcsiga))));
          //     } else { // erad_f_new > 0
          //       Real chi_ratio = (sigma_a+sigma_p)/sigma_s;
          //       Real coeff4 = 0.25*chi_ratio*arad_/(inv_t_electron_*erad_f_new);
          //       Real coeff0 = -(tradnew + 0.25*chi_ratio/inv_t_electron_);
          //       coeff0 += -0.25*u_tet[0]/dtcsigs * delta_erad_f/erad_f_new / inv_t_electron_;
          //       if (fabs(coeff4) > 1.0e-20) {
          //         bool flag = FourthPolyRoot(coeff4, coeff0, tgas_update);
          //         if (!(flag) || !(isfinite(tgas_update))) tgas_update = tgasnew;
          //       } else {
          //         tgas_update = -coeff0;
          //       }
          //     } // endelse
          //     if (wdn*tgas_update < pfloor) tgas_update = tgasnew;
          //     // assign the tgas update
          //     tgasnew = tgas_update;
          //   } // endif (wdn < 1e-5)
          // } // endif turn_on_sao_radsrc_

          // handle excision (see notes above)
          if (excise) {
            if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0_(m,n,k,j,i) = 0.0; }
          }
        }

        // feedback on fluid
        if (affect_fluid_) {
          u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
          u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
          u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
          u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
        }

        // update total entropy if entropy fix is enabled
        if (entropy_fix_) {
          // total entropy must be assigned as the last passive scalar
          Real src0 = m_new[0] - m_old[0];
          Real src1 = m_new[1] - m_old[1];
          Real src2 = m_new[2] - m_old[2];
          Real src3 = m_new[3] - m_old[3];

          // compute coord-frame 4-velocity
          Real u1  = wvx - alpha * gamma * gupper[0][1];;
          Real u2  = wvy - alpha * gamma * gupper[0][2];;
          Real u3  = wvz - alpha * gamma * gupper[0][3];;
          Real rho = wdn;

          // add source term to total entropy
          Real src = gm1 * (u0*src0+u1*src1+u2*src2+u3*src3) / std::pow(rho, gm1);
          u0_(m,entropyIdx,k,j,i) += src;
        }
      } else {
        // NOTE(@pdmullen): At this point, it is possible that excision has not been
        // entirely applied if Compton is enabled and a badcell or temperature equilibrium
        // was encountered.. apply excision
        if (excise) {
          for (int n=0; n<=nang1; ++n) {
            Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+
                       tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)+
                       tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+
                       tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
            if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0_(m,n,k,j,i) = 0.0; }
          }
        }
      }
    } // endif is_compton_enabled_

    // Save updated gas temperature
    tgas_radsource_(m,k,j,i) = tgasnew;
  }); // end par_for 'radiation_source'

  // Call ConsToPrim over active zones with temperature fix
  if (temperature_fix_turn_on_) {
    // temperature fix currently only works with MHD
    if (!(fixed_fluid_) && is_mhd_enabled_) {
        auto &b0_ = pmy_pack->pmhd->b0;
        auto &bcc0_ = pmy_pack->pmhd->bcc0;
        pmy_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,true,is,ie,js,je,ks,ke);
    }
  } // endif temperature_fix_turn_on

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  bool FourthPolyRoot
//  \brief Exact solution for fourth order polynomial of
//  the form coef4 * x^4 + x + tconst = 0.

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root) {
  // Calculate real root of z^3 - 4*tconst/coef4 * z - 1/coef4^2 = 0
  Real ccubic = tconst * tconst * tconst;
  Real delta1 = 0.25 - 64.0 * ccubic * coef4 / 27.0;
  if (delta1 < 0.0) {
    return false;
  }
  delta1 = sqrt(delta1);
  if (delta1 < 0.5) {
    return false;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = pow(delta1, -2.0/3.0) / 3.0;
  } else {
    zroot = pow(0.5 + delta1, 1.0/3.0) - pow(-0.5 + delta1, 1.0/3.0);
  }
  if (zroot < 0.0) {
    return false;
  }
  zroot *= pow(coef4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coef4 * rcoef);
  if (delta2 < 0.0) {
    return false;
  }
  delta2 = sqrt(delta2);
  root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return false;
  }
  return true;
}

} // namespace radiation
