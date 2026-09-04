//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_source.cpp

#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "units/units.hpp"
#include "dyn_radiation.hpp"

#include "dyn_radiation/dyn_radiation_tetrad.hpp"
#include "dyn_radiation/dyn_radiation_opacities.hpp"

namespace dyn_radiation {

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root);

KOKKOS_INLINE_FUNCTION
bool OpacityDensityScale(const Real wdn, const Real dfloor, const Real dfloor_opacity,
                         const Real dens_trunc_max, const Real tau_truncation,
                         const Real sigmoid_residual, const Real kappa_s,
                         const Real delta_l, const Real sigma_cold,
                         const bool use_excision_density, Real &scale) {
  scale = 1.0;
  if (!(wdn > 0.0) || !(dfloor > 0.0) || !(dfloor_opacity > 0.0)) {
    return false;
  }
  if (use_excision_density) {
    scale = dfloor_opacity/wdn;
    return Kokkos::isfinite(scale);
  }
  if (!(delta_l > 0.0) || !(Kokkos::isfinite(delta_l))) {
    return false;
  }

  Real dtrunc = dfloor;
  if (kappa_s > 0.0 && tau_truncation > 0.0 && sigma_cold > 0.0) {
    dtrunc = sigma_cold*tau_truncation/(kappa_s*delta_l);
    if (!(Kokkos::isfinite(dtrunc)) || dtrunc <= 0.0) {
      return false;
    }
    dtrunc = fmin(dens_trunc_max, fmax(dfloor, dtrunc));
  }

  const Real fac_trunc = dtrunc/dfloor;
  const Real wdn_real = fmax(wdn - dfloor, dfloor_opacity);
  if (!(fac_trunc > 0.0) || !(wdn_real > 0.0) || !(Kokkos::isfinite(fac_trunc))) {
    return false;
  }

  Real wdn_opacity = wdn_real;
  if (fabs(fac_trunc - 1.0) > 1.0e-12) {
    const Real denom = log(1.0/sigmoid_residual - 1.0);
    if (!(denom > 0.0)) { return false; }
    const Real wid_trunc = 0.5*log10(fac_trunc)/denom;
    if (!(wid_trunc > 0.0) || !(Kokkos::isfinite(wid_trunc))) {
      return false;
    }
    const Real center = log10(dfloor) + 0.5*log10(fac_trunc);
    const Real fac_inv = 1.0 + exp(-(log10(wdn_real) - center)/wid_trunc);
    if (!(fac_inv > 0.0) || !(Kokkos::isfinite(fac_inv))) {
      return false;
    }
    const Real del_reduce = log10(dfloor) - log10(dfloor_opacity);
    wdn_opacity = pow(10.0, log10(wdn_real) - (1.0 - 1.0/fac_inv)*del_reduce);
  }

  scale = wdn_opacity/wdn;
  return (scale >= 0.0 && Kokkos::isfinite(scale));
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus DynRadiation::AddTmunu(Driver *pdriver, int stage)
//! \brief Radiation stress-energy is intentionally metric-passive for now.

TaskStatus DynRadiation::AddTmunu(Driver *pdriver, int stage) {
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus DynRadiation::RadFluidCoupling(Driver *pdriver, int stage)
//! \brief Add implicit dyn_radiation-fluid source terms.  Based on @c-white and @yanfeij's
//! gr_rad branch, dyn_radiation/coupling/emission.cpp commit be7f84565b.

TaskStatus DynRadiation::RadFluidCoupling(Driver *pdriver, int stage) {
  // Return if dyn_radiation source term disabled
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
  int &source_max_iter_ = source_max_iter;
  Real &source_tolerance_ = source_tolerance;
  bool &correct_radsrc_velocity_ = correct_radsrc_velocity;
  bool &correct_radsrc_opacity_ = correct_radsrc_opacity;
  Real &dfloor_opacity_ = dfloor_opacity;
  Real &dens_trunc_max_ = dens_trunc_max;
  Real &tau_truncation_ = tau_truncation;
  Real &sigmoid_residual_ = sigmoid_residual;
  bool use_dyn_grmhd_ = (pmy_pack->pdyngr != nullptr);

  // Extract coordinate/excision data
  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;
  bool &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  Real &n_0_floor_ = n_0_floor;

  // Extract dyn_radiation constant and units
  Real &arad_ = arad;
  Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
  Real mean_mol_weight_ = 1.0;
  Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
  Real inv_t_electron_ = 1.0;
  if (are_units_enabled_) {
    density_scale_ = pmy_pack->punit->density_cgs();
    temperature_scale_ = pmy_pack->punit->temperature_cgs();
    length_scale_ = pmy_pack->punit->length_cgs();
    mean_mol_weight_ = pmy_pack->punit->mu();
    rosseland_coef_ = pmy_pack->punit->rosseland_coef_cgs;
    planck_minus_rosseland_coef_ = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
    inv_t_electron_ = temperature_scale_/pmy_pack->punit->electron_rest_mass_energy_cgs;
  }

  // Extract adiabatic index
  Real gm1 = 0.0, dfloor = FLT_MIN, v_sq_max = 1.0 - 1.0e-14;
  if (is_hydro_enabled_) {
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
    dfloor = pmy_pack->phydro->peos->eos_data.dfloor;
    v_sq_max = 1.0 - 1.0/SQR(pmy_pack->phydro->peos->eos_data.gamma_max);
  } else if (is_mhd_enabled_) {
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
    dfloor = pmy_pack->pmhd->peos->eos_data.dfloor;
    v_sq_max = 1.0 - 1.0/SQR(pmy_pack->pmhd->peos->eos_data.gamma_max);
  }

  // Extract dyn_radiation, dyn_radiation frame, and dyn_radiation angular mesh data
  auto &i0_ = i0;
  Real &kappa_a_ = kappa_a;
  Real &kappa_s_ = kappa_s;
  Real &kappa_p_ = kappa_p;
  bool &power_opacity_ = power_opacity;
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &sqrt_detg_c_ = sqrt_detg_c;
  auto &adm_alpha_c_ = adm_alpha_c;
  auto &adm_beta_u_c_ = adm_beta_u_c;
  auto &adm_g_dd_c_ = adm_g_dd_c;
  auto &solid_angles_ = prgeo->solid_angles;
  bool use_adm_geometry_ = use_adm_geometry;

  // Extract hydro/mhd quantities
  DvceArray5D<Real> u0_, w0_, bcc0_;
  if (is_hydro_enabled_) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (is_mhd_enabled_) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
    bcc0_ = pmy_pack->pmhd->bcc0;
  }

  // Extract timestep
  Real dt_ = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  const Real coupling_floor = std::numeric_limits<Real>::min();

  // Call ConsToPrim over active zones prior to source term application
  if (!(fixed_fluid_)) {
    if (use_dyn_grmhd_) {
      (void) pmy_pack->pdyngr->ConToPrim(pdriver, stage);
    } else if (is_hydro_enabled_) {
      pmy_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is,ie,js,je,ks,ke);
    } else if (is_mhd_enabled_) {
      auto &b0_ = pmy_pack->pmhd->b0;
      pmy_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,is,ie,js,je,ks,ke);
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

    // Compute the metric needed by the fluid primitive convention.  In ADM mode,
    // radiation quantities are evolved as U=sqrt(gamma) I and the tetrad is the
    // Eulerian ADM tetrad cached once per stage.
    Real glower[4][4], gupper[4][4];
    Real alpha;
    if (use_adm_geometry_) {
      alpha = adm_alpha_c_(m,k,j,i);
      for (int a=0; a<4; ++a) {
        for (int b=0; b<4; ++b) {
          glower[a][b] = 0.0;
          gupper[a][b] = 0.0;
        }
      }
      for (int a=0; a<3; ++a) {
        for (int b=0; b<3; ++b) {
          glower[a+1][b+1] = adm_g_dd_c_(m,a,b,k,j,i);
        }
      }
      Real beta_d[3] = {0.0, 0.0, 0.0};
      for (int a=0; a<3; ++a) {
        for (int b=0; b<3; ++b) {
          beta_d[a] += adm_g_dd_c_(m,a,b,k,j,i)*adm_beta_u_c_(m,b,k,j,i);
        }
      }
      Real beta_sq = 0.0;
      for (int a=0; a<3; ++a) {
        beta_sq += beta_d[a]*adm_beta_u_c_(m,a,k,j,i);
        glower[0][a+1] = beta_d[a];
        glower[a+1][0] = beta_d[a];
      }
      glower[0][0] = -SQR(alpha) + beta_sq;
    } else {
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
      alpha = sqrt(-1.0/gupper[0][0]);
    }

    // fluid state
    Real &wdn = w0_(m,IDN,k,j,i);
    Real &wvx = w0_(m,IVX,k,j,i);
    Real &wvy = w0_(m,IVY,k,j,i);
    Real &wvz = w0_(m,IVZ,k,j,i);
    Real &wen = w0_(m,IEN,k,j,i);

    // derived quantities
    Real pgas = use_dyn_grmhd_ ? wen : gm1*wen;
    Real tgas = pgas/wdn;
    Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
           + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz
           + glower[3][3]*wvz*wvz;
    Real gamma = sqrt(1.0 + q);
    Real u0 = gamma/alpha;

    Real sigma_cold = 0.0;
    if (correct_radsrc_opacity_ && is_mhd_enabled_ && wdn > 0.0 &&
        alpha > 0.0 && u0 > 0.0 && Kokkos::isfinite(u0)) {
      Real u1, u2, u3;
      if (use_adm_geometry_) {
        u1 = wvx - gamma*adm_beta_u_c_(m,0,k,j,i)/alpha;
        u2 = wvy - gamma*adm_beta_u_c_(m,1,k,j,i)/alpha;
        u3 = wvz - gamma*adm_beta_u_c_(m,2,k,j,i)/alpha;
      } else {
        u1 = wvx - alpha*gamma*gupper[0][1];
        u2 = wvy - alpha*gamma*gupper[0][2];
        u3 = wvz - alpha*gamma*gupper[0][3];
      }

      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      Real bccx = bcc0_(m,IBX,k,j,i);
      Real bccy = bcc0_(m,IBY,k,j,i);
      Real bccz = bcc0_(m,IBZ,k,j,i);
      Real b0 = u_1*bccx + u_2*bccy + u_3*bccz;
      Real b1 = (bccx + b0*u1)/u0;
      Real b2 = (bccy + b0*u2)/u0;
      Real b3 = (bccz + b0*u3)/u0;

      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
      if (b_sq > 0.0 && Kokkos::isfinite(b_sq)) {
        sigma_cold = b_sq/wdn;
      }
    }

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

    // Coordinate component n^0 of the photon direction normalized to unit
    // Eulerian/tetrad frequency.
    Real n0 = use_adm_geometry_ ? 1.0/alpha : tt(m,0,0,k,j,i);
    Real sqrtg = use_adm_geometry_ ? sqrt_detg_c_(m,k,j,i) : 1.0;

    Real sigma_a = 0.0, sigma_s = 0.0, sigma_p = 0.0;
    Real dtcsiga = 0.0, dtcsigs = 0.0, dtcsigp = 0.0;
    Real dtaucsiga = 0.0, dtaucsigs = 0.0, dtaucsigp = 0.0;
    Real wght_sum = 0.0;
    Real suma1 = 0.0;
    Real suma2 = 0.0;
    Real suma3 = 0.0;
    Real coef[2] = {0.0, 0.0};
    Real tgasnew = tgas;
    bool badcell = false;

    // Iterate the local nonlinear solve so temperature-dependent opacities and
    // stiff radiation-matter exchange use mutually consistent coefficients.
    bool radsrc_velocity_applied = false;
    for (int iter=0; iter<source_max_iter_; ++iter) {
      const Real opacity_tgas = fmax(tgasnew, coupling_floor);
      OpacityFunction(wdn, density_scale_,
                      opacity_tgas, temperature_scale_,
                      length_scale_, gm1, mean_mol_weight_,
                      power_opacity_, rosseland_coef_, planck_minus_rosseland_coef_,
                      kappa_a_, kappa_s_, kappa_p_,
                      sigma_a, sigma_s, sigma_p);
      if (correct_radsrc_opacity_) {
        Real delta_l = fmax(fmax(size.d_view(m).dx1, size.d_view(m).dx2),
                            size.d_view(m).dx3);
        if (use_adm_geometry_) {
          Real dl1 = sqrt(fmax(adm_g_dd_c_(m,0,0,k,j,i), 0.0))*size.d_view(m).dx1;
          Real dl2 = sqrt(fmax(adm_g_dd_c_(m,1,1,k,j,i), 0.0))*size.d_view(m).dx2;
          Real dl3 = sqrt(fmax(adm_g_dd_c_(m,2,2,k,j,i), 0.0))*size.d_view(m).dx3;
          Real proper_l = fmax(fmax(dl1, dl2), dl3);
          if (proper_l > 0.0 && Kokkos::isfinite(proper_l)) {
            delta_l = proper_l;
          }
        }
        Real opacity_scale = 1.0;
        bool scale_ok = OpacityDensityScale(wdn, dfloor, dfloor_opacity_, dens_trunc_max_,
                                            tau_truncation_, sigmoid_residual_, kappa_s_,
                                            delta_l, sigma_cold, excision_flux_(m,k,j,i),
                                            opacity_scale);
        if (scale_ok) {
          // Optional opacity-density regularization for floor, high-magnetization,
          // and flux-excised cells.
          sigma_a *= opacity_scale;
          sigma_s *= opacity_scale;
          sigma_p *= opacity_scale;
        }
      }
      dtcsiga = dt_*sigma_a;
      dtcsigs = dt_*sigma_s;
      dtcsigp = dt_*sigma_p;
      dtaucsiga = dtcsiga/u0;
      dtaucsigs = dtcsigs/u0;
      dtaucsigp = dtcsigp/u0;

      if (correct_radsrc_velocity_ && !(radsrc_velocity_applied)) {
        radsrc_velocity_applied = true;
        // Optional stabilization for radiation-dominated cells.  This changes only
        // the velocity used by this local source solve.
        bool velocity_ok = (wdn > 0.0 && u_tet[0] > 0.0 && arad_ > 0.0 &&
                            n0 != 0.0 && sqrtg > 0.0 && Kokkos::isfinite(n0) &&
                            Kokkos::isfinite(sqrtg));
        Real erad_f = 0.0;
        Real omega_hat_tot = 0.0;
        Real omega_cm_tot = 0.0;
        for (int n=0; n<=nang1; ++n) {
          Real n_0 = 1.0;
          if (!(use_adm_geometry_)) {
            n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
                  tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                  tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
                  tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          }
          Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                        u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
          Real norm = use_adm_geometry_ ? sqrtg : n0*n_0;
          if (norm == 0.0 || n0_cm == 0.0 || !(Kokkos::isfinite(norm)) ||
              !(Kokkos::isfinite(n0_cm))) {
            velocity_ok = false;
            break;
          }
          Real intensity = i0_(m,n,k,j,i)/norm;
          Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
          if (!(Kokkos::isfinite(intensity)) || !(Kokkos::isfinite(omega_cm))) {
            velocity_ok = false;
            break;
          }
          omega_hat_tot += solid_angles_.d_view(n);
          omega_cm_tot += omega_cm;
          erad_f += intensity*SQR(SQR(n0_cm))*omega_cm;
        }
        if (velocity_ok && omega_cm_tot > 0.0 && Kokkos::isfinite(omega_cm_tot)) {
          erad_f *= omega_hat_tot/omega_cm_tot;
        } else {
          velocity_ok = false;
        }

        Real egas = use_dyn_grmhd_ && gm1 > 0.0 ? pgas/gm1 : wen;
        if (velocity_ok && erad_f > wdn + egas && Kokkos::isfinite(erad_f)) {
          Real rr_tet00 = 0.0;
          Real rr_tet01 = 0.0, rr_tet02 = 0.0, rr_tet03 = 0.0;
          Real rr_tet11 = 0.0, rr_tet22 = 0.0, rr_tet33 = 0.0;
          Real rr_tet12 = 0.0, rr_tet13 = 0.0, rr_tet23 = 0.0;
          for (int n=0; n<=nang1; ++n) {
            Real nh0 = nh_c_.d_view(n,0);
            Real nh1 = nh_c_.d_view(n,1);
            Real nh2 = nh_c_.d_view(n,2);
            Real nh3 = nh_c_.d_view(n,3);
            Real n_0 = 1.0;
            if (!(use_adm_geometry_)) {
              n_0 = tc(m,0,0,k,j,i)*nh0 + tc(m,1,0,k,j,i)*nh1 +
                    tc(m,2,0,k,j,i)*nh2 + tc(m,3,0,k,j,i)*nh3;
            }
            Real norm = use_adm_geometry_ ? sqrtg : n0*n_0;
            if (norm == 0.0 || !(Kokkos::isfinite(norm))) {
              velocity_ok = false;
              break;
            }
            Real intensity = i0_(m,n,k,j,i)/norm;
            if (!(Kokkos::isfinite(intensity))) {
              velocity_ok = false;
              break;
            }
            Real weight = intensity*solid_angles_.d_view(n);
            rr_tet00 += weight;
            rr_tet01 += nh1*weight;
            rr_tet02 += nh2*weight;
            rr_tet03 += nh3*weight;
            rr_tet11 += nh1*nh1*weight;
            rr_tet22 += nh2*nh2*weight;
            rr_tet33 += nh3*nh3*weight;
            rr_tet12 += nh1*nh2*weight;
            rr_tet13 += nh1*nh3*weight;
            rr_tet23 += nh2*nh3*weight;
          }

          const Real local_v_sq_max = fmin(v_sq_max, 1.0 - 1.0e-14);
          if (velocity_ok && rr_tet00 > 0.0 && local_v_sq_max > 0.0 &&
              Kokkos::isfinite(rr_tet00)) {
            Real vrad_tet1 = rr_tet01/rr_tet00;
            Real vrad_tet2 = rr_tet02/rr_tet00;
            Real vrad_tet3 = rr_tet03/rr_tet00;
            Real vrad_sq = SQR(vrad_tet1) + SQR(vrad_tet2) + SQR(vrad_tet3);
            if (vrad_sq > local_v_sq_max) {
              Real ratio = sqrt(local_v_sq_max/vrad_sq);
              vrad_tet1 *= ratio;
              vrad_tet2 *= ratio;
              vrad_tet3 *= ratio;
              vrad_sq = local_v_sq_max;
            }
            velocity_ok = (vrad_sq >= 0.0 && vrad_sq < 1.0 && Kokkos::isfinite(vrad_sq));
            if (velocity_ok) {
              Real urad_tet0 = 1.0/sqrt(1.0 - vrad_sq);
              Real urad_tet1 = urad_tet0*vrad_tet1;
              Real urad_tet2 = urad_tet0*vrad_tet2;
              Real urad_tet3 = urad_tet0*vrad_tet3;

              Real wgas = wdn + egas + pgas;
              Real mgas_tet1 = wgas*u_tet[0]*u_tet[1];
              Real mgas_tet2 = wgas*u_tet[0]*u_tet[2];
              Real mgas_tet3 = wgas*u_tet[0]*u_tet[3];
              Real mgas_rad_tet1 = wgas*urad_tet0*urad_tet1;
              Real mgas_rad_tet2 = wgas*urad_tet0*urad_tet2;
              Real mgas_rad_tet3 = wgas*urad_tet0*urad_tet3;

              Real chi_p = sigma_p + sigma_a;
              Real chi_s = sigma_s;
              Real chi_a = sigma_a + sigma_s;
              Real emissivity = chi_p*arad_*SQR(SQR(tgas)) + chi_s*erad_f;
              if (is_compton_enabled_ && erad_f > 0.0) {
                Real trad = sqrt(sqrt(erad_f/arad_));
                if (Kokkos::isfinite(trad)) {
                  emissivity += chi_s*4.0*(tgas - trad)*inv_t_electron_*erad_f;
                }
              }

              Real gg_tet1 = -emissivity*u_tet[1] -
                             chi_a*(-u_tet[0]*rr_tet01 + u_tet[1]*rr_tet11 +
                                    u_tet[2]*rr_tet12 + u_tet[3]*rr_tet13);
              Real gg_tet2 = -emissivity*u_tet[2] -
                             chi_a*(-u_tet[0]*rr_tet02 + u_tet[1]*rr_tet12 +
                                    u_tet[2]*rr_tet22 + u_tet[3]*rr_tet23);
              Real gg_tet3 = -emissivity*u_tet[3] -
                             chi_a*(-u_tet[0]*rr_tet03 + u_tet[1]*rr_tet13 +
                                    u_tet[2]*rr_tet23 + u_tet[3]*rr_tet33);
              Real dt_over_ut0 = dt_/u_tet[0];
              Real dmgas_tet1 = gg_tet1*dt_over_ut0;
              Real dmgas_tet2 = gg_tet2*dt_over_ut0;
              Real dmgas_tet3 = gg_tet3*dt_over_ut0;

              Real denom1 = mgas_rad_tet1 - mgas_tet1;
              Real denom2 = mgas_rad_tet2 - mgas_tet2;
              Real denom3 = mgas_rad_tet3 - mgas_tet3;
              Real frac1 = (fabs(denom1) > 0.0 && Kokkos::isfinite(denom1)) ?
                           dmgas_tet1/denom1 : 0.0;
              Real frac2 = (fabs(denom2) > 0.0 && Kokkos::isfinite(denom2)) ?
                           dmgas_tet2/denom2 : 0.0;
              Real frac3 = (fabs(denom3) > 0.0 && Kokkos::isfinite(denom3)) ?
                           dmgas_tet3/denom3 : 0.0;
              frac1 = Kokkos::isfinite(frac1) ? fmin(fmax(frac1, 0.0), 1.0) : 0.0;
              frac2 = Kokkos::isfinite(frac2) ? fmin(fmax(frac2, 0.0), 1.0) : 0.0;
              frac3 = Kokkos::isfinite(frac3) ? fmin(fmax(frac3, 0.0), 1.0) : 0.0;
              Real u1_new = (1.0 - frac1)*u_tet[1] + frac1*urad_tet1;
              Real u2_new = (1.0 - frac2)*u_tet[2] + frac2*urad_tet2;
              Real u3_new = (1.0 - frac3)*u_tet[3] + frac3*urad_tet3;
              Real u0_new = sqrt(1.0 + SQR(u1_new) + SQR(u2_new) + SQR(u3_new));
              if (Kokkos::isfinite(u0_new) && Kokkos::isfinite(u1_new) &&
                  Kokkos::isfinite(u2_new) && Kokkos::isfinite(u3_new)) {
                u_tet[0] = u0_new;
                u_tet[1] = u1_new;
                u_tet[2] = u2_new;
                u_tet[3] = u3_new;
              }
            }
          }
        }
      }

      wght_sum = 0.0;
      suma1 = 0.0;
      suma2 = 0.0;
      for (int n=0; n<=nang1; ++n) {
        Real n_0 = 1.0;
        if (!(use_adm_geometry_)) {
          n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
                tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
                tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        }
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
        Real intensity = use_adm_geometry_ ? i0_(m,n,k,j,i)/sqrtg :
                                             i0_(m,n,k,j,i)/(n0*n_0);
        Real intensity_cm = 4.0*M_PI*intensity*SQR(SQR(n0_cm));
        // In ADM mode n0=1/alpha, so this denominator is n0 times
        // (1 + alpha*dt*sigma*D); the coordinate step therefore uses the
        // comoving optical-depth factor alpha*D*dt.
        Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        Real ir_weight = intensity_cm*omega_cm;
        wght_sum += omega_cm;
        suma1 += omega_cm*vncsigma2;
        suma2 += ir_weight*n0*vncsigma;
      }

      if (wght_sum <= coupling_floor || !(Kokkos::isfinite(wght_sum))) {
        badcell = true;
        break;
      }
      suma1 /= wght_sum;
      suma2 /= wght_sum;
      suma3 = suma1*(dtcsigs - dtcsigp);
      suma1 *= (dtcsiga + dtcsigp);
      const Real denom = 1.0 - suma3;
      if (fabs(denom) <= coupling_floor || !(Kokkos::isfinite(denom))) {
        badcell = true;
        break;
      }

      coef[1] = (dtaucsiga+dtaucsigp-(dtaucsiga+dtaucsigp)*suma1/denom)
                *arad_*gm1/wdn;
      coef[0] = -tgas-(dtaucsiga+dtaucsigp)*suma2*gm1/(wdn*denom);

      Real tnext = tgasnew;
      bool flag = true;
      if (fabs(coef[1]) > 1.0e-20) {
        flag = FourthPolyRoot(coef[1], coef[0], tnext);
      } else {
        tnext = -coef[0];
      }
      if (!(flag) || !(Kokkos::isfinite(tnext)) || tnext < 0.0) {
        badcell = true;
        break;
      }

      const Real tupdated = power_opacity_ ? (0.75*tgasnew + 0.25*tnext) : tnext;
      const Real rel = fabs(tupdated - tgasnew)/
                       fmax(fmax(fabs(tupdated), fabs(tgasnew)), coupling_floor);
      tgasnew = tupdated;
      if (!(power_opacity_) || rel <= source_tolerance_) { break; }
    }
    if (badcell) { tgasnew = tgas; }

    // Update the specific intensity
    if (!(badcell)) {
      // Calculate emission coefficient and updated jr_cm
      Real emission = arad_*SQR(SQR(tgasnew));
      Real jr_cm = (suma1*emission + suma2)/(1.0 - suma3);
      Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
      for (int n=0; n<=nang1; ++n) {
        Real n_0 = 1.0;
        Real mom[3] = {0.0, 0.0, 0.0};
        if (use_adm_geometry_) {
          Real s[3] = {0.0, 0.0, 0.0};
          for (int a=0; a<3; ++a) {
            for (int d=0; d<3; ++d) {
              s[d] += tt(m,a+1,d+1,k,j,i)*nh_c_.d_view(n,a+1);
            }
          }
          for (int a=0; a<3; ++a) {
            for (int b=0; b<3; ++b) {
              mom[a] += adm_g_dd_c_(m,a,b,k,j,i)*s[b];
            }
          }
        } else {
          n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
                tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
                tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0) +
                     tc(m,1,1,k,j,i)*nh_c_.d_view(n,1) +
                     tc(m,2,1,k,j,i)*nh_c_.d_view(n,2) +
                     tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
          Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0) +
                     tc(m,1,2,k,j,i)*nh_c_.d_view(n,1) +
                     tc(m,2,2,k,j,i)*nh_c_.d_view(n,2) +
                     tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
          Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0) +
                     tc(m,1,3,k,j,i)*nh_c_.d_view(n,1) +
                     tc(m,2,3,k,j,i)*nh_c_.d_view(n,2) +
                     tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);
          mom[0] = n_1/n_0;
          mom[1] = n_2/n_0;
          mom[2] = n_3/n_0;
        }
        // compute moments before coupling
        m_old[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
        m_old[1] += (mom[0]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
        m_old[2] += (mom[1]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
        m_old[3] += (mom[2]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
      }

      for (int n=0; n<=nang1; ++n) {
        Real n_0 = 1.0;
        if (!(use_adm_geometry_)) {
          n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
                tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
                tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        }
        Real conserved_norm = use_adm_geometry_ ? sqrtg : n0*n_0;
        // update intensity
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/conserved_norm)*SQR(SQR(n0_cm));
        Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        Real di_cm = ( ((dtcsigs-dtcsigp)*jr_cm
                      + (dtcsiga+dtcsigp)*emission
                      - (dtcsigs+dtcsiga)*intensity_cm)*vncsigma2 );
        i0_(m,n,k,j,i) = conserved_norm*(i0_(m,n,k,j,i)/conserved_norm +
                                         di_cm/(4.0*M_PI*SQR(SQR(n0_cm))));
      }

      // handle excision
      // NOTE(@pdmullen): The below zeroes all intensities within rks <= r_excision and
      // zeroes intensities within angles where n_0 is about zero. When Compton is
      // enabled, we delay the n_0_floor excision so that intensites updated via
      // absorption and scattering inform the Compton update
      if (excise) {
        for (int n=0; n<=nang1; ++n) {
          Real n_0 = 1.0;
          if (!(use_adm_geometry_)) {
            n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
                  tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                  tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
                  tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          }
          bool apply_excision = (rad_mask_(m,k,j,i) ||
                                 (!(use_adm_geometry_) && !(is_compton_enabled_) &&
                                  fabs(n_0) < n_0_floor_));
          if (apply_excision) { i0_(m,n,k,j,i) = 0.0; }
        }
      }
      if (use_adm_geometry_) {
        ConservativeAngularFloor(i0_, solid_angles_, m, k, j, i, nang1);
      } else {
        ConservativePrimitiveAngularFloor(i0_, solid_angles_, tt, tc, nh_c_,
                                          m, k, j, i, nang1);
      }

      for (int n=0; n<=nang1; ++n) {
        Real n_0 = 1.0;
        Real mom[3] = {0.0, 0.0, 0.0};
        if (use_adm_geometry_) {
          Real s[3] = {0.0, 0.0, 0.0};
          for (int a=0; a<3; ++a) {
            for (int d=0; d<3; ++d) {
              s[d] += tt(m,a+1,d+1,k,j,i)*nh_c_.d_view(n,a+1);
            }
          }
          for (int a=0; a<3; ++a) {
            for (int b=0; b<3; ++b) {
              mom[a] += adm_g_dd_c_(m,a,b,k,j,i)*s[b];
            }
          }
        } else {
          n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
                tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
                tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0) +
                     tc(m,1,1,k,j,i)*nh_c_.d_view(n,1) +
                     tc(m,2,1,k,j,i)*nh_c_.d_view(n,2) +
                     tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
          Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0) +
                     tc(m,1,2,k,j,i)*nh_c_.d_view(n,1) +
                     tc(m,2,2,k,j,i)*nh_c_.d_view(n,2) +
                     tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
          Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0) +
                     tc(m,1,3,k,j,i)*nh_c_.d_view(n,1) +
                     tc(m,2,3,k,j,i)*nh_c_.d_view(n,2) +
                     tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);
          mom[0] = n_1/n_0;
          mom[1] = n_2/n_0;
          mom[2] = n_3/n_0;
        }
        m_new[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
        m_new[1] += (mom[0]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
        m_new[2] += (mom[1]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
        m_new[3] += (mom[2]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
      }
      // update conserved fluid variables
      if (affect_fluid_) {
        u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
        u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
        u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
        u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
      }
    }

    // compton scattering
    if (is_compton_enabled_ && wght_sum > coupling_floor && Kokkos::isfinite(wght_sum)) {
      // use partially updated gas temperature
      tgas = tgasnew;

      // compute polynomial coefficients using partially updated gas temp and intensity
      suma1 = 0.0;
      Real jr_cm = 0.0;
      for (int n=0; n<=nang1; ++n) {
        Real n_0 = 1.0;
        if (!(use_adm_geometry_)) {
          n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) +
                tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) +
                tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        }
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real wght_cm = solid_angles_.d_view(n)/SQR(n0_cm)/wght_sum;
        Real intensity = use_adm_geometry_ ? i0_(m,n,k,j,i)/sqrtg :
                                             i0_(m,n,k,j,i)/(n0*n_0);
        Real intensity_cm = 4.0*M_PI*intensity*SQR(SQR(n0_cm));
        Real ir_weight = intensity_cm*wght_cm;
        jr_cm += ir_weight;
        suma1 += (n0_cm/n0)*4.0*dtcsigs*inv_t_electron_*wght_cm;
      }
      suma2 = 4.0*dtaucsigs*inv_t_electron_*gm1/wdn;

      // compute partially updated dyn_radiation temperature
      const bool compton_well_defined = (jr_cm > coupling_floor &&
                                         fabs(suma1) > coupling_floor &&
                                         arad_ > coupling_floor &&
                                         Kokkos::isfinite(jr_cm) && Kokkos::isfinite(suma1));
      Real trad = compton_well_defined ? sqrt(sqrt(jr_cm/arad_)) : tgas;
      const bool temp_equil = (fabs(trad - tgas) < 1.0e-12);

      // Calculate new gas temperature due to Compton
      Real tradnew = trad;
      badcell = false;
      if (!(compton_well_defined)) {
        badcell = true;
      } else if (!(temp_equil)) {
        coef[1] = (1.0 + suma2*jr_cm)/(suma1*jr_cm)*arad_;
        coef[0] = -(1.0 + suma2*jr_cm)/suma1 - tgas;
        bool flag = FourthPolyRoot(coef[1], coef[0], tradnew);
        if (!(flag) || !(Kokkos::isfinite(tradnew))) {
          badcell = true;
        }
      }

      // Update the specific intensity
      if (!(badcell) && !(temp_equil)) {
        // Compute updated gas temperature
        tgasnew = (arad_*SQR(SQR(tradnew)) - jr_cm)/(suma1*jr_cm) + tradnew;
        Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
        for (int n=0; n<=nang1; ++n) {
          Real n_0 = 1.0;
          Real mom[3] = {0.0, 0.0, 0.0};
          if (use_adm_geometry_) {
            Real s[3] = {0.0, 0.0, 0.0};
            for (int a=0; a<3; ++a) {
              for (int d=0; d<3; ++d) {
                s[d] += tt(m,a+1,d+1,k,j,i)*nh_c_.d_view(n,a+1);
              }
            }
            for (int a=0; a<3; ++a) {
              for (int b=0; b<3; ++b) {
                mom[a] += adm_g_dd_c_(m,a,b,k,j,i)*s[b];
              }
            }
          } else {
            n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 +tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
            Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                     + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
            Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                     + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
            Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                     + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);
            mom[0] = n_1/n_0;
            mom[1] = n_2/n_0;
            mom[2] = n_3/n_0;
          }

          // compute moments before coupling
          m_old[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
          m_old[1] += (mom[0]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
          m_old[2] += (mom[1]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
          m_old[3] += (mom[2]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
        }

        for (int n=0; n<=nang1; ++n) {
          Real n_0 = 1.0;
          if (!(use_adm_geometry_)) {
            n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 +tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          }
          Real conserved_norm = use_adm_geometry_ ? sqrtg : n0*n_0;
          // update intensity
          Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                        u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
          Real di_cm = (n0_cm/n0)*dtcsigs*4.0*jr_cm*inv_t_electron_*(tgasnew - tradnew);
          i0_(m,n,k,j,i) = conserved_norm*(i0_(m,n,k,j,i)/conserved_norm +
                                           di_cm/(4.0*M_PI*SQR(SQR(n0_cm))));

          // handle excision (see notes above)
          if (excise) {
            if (rad_mask_(m,k,j,i) || (!(use_adm_geometry_) && fabs(n_0) < n_0_floor_)) {
              i0_(m,n,k,j,i) = 0.0;
            }
          }
        }
        if (use_adm_geometry_) {
          ConservativeAngularFloor(i0_, solid_angles_, m, k, j, i, nang1);
        } else {
          ConservativePrimitiveAngularFloor(i0_, solid_angles_, tt, tc, nh_c_,
                                            m, k, j, i, nang1);
        }

        for (int n=0; n<=nang1; ++n) {
          Real n_0 = 1.0;
          Real mom[3] = {0.0, 0.0, 0.0};
          if (use_adm_geometry_) {
            Real s[3] = {0.0, 0.0, 0.0};
            for (int a=0; a<3; ++a) {
              for (int d=0; d<3; ++d) {
                s[d] += tt(m,a+1,d+1,k,j,i)*nh_c_.d_view(n,a+1);
              }
            }
            for (int a=0; a<3; ++a) {
              for (int b=0; b<3; ++b) {
                mom[a] += adm_g_dd_c_(m,a,b,k,j,i)*s[b];
              }
            }
          } else {
            n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 +tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
            Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                     + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
            Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                     + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
            Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                     + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);
            mom[0] = n_1/n_0;
            mom[1] = n_2/n_0;
            mom[2] = n_3/n_0;
          }
          m_new[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
          m_new[1] += (mom[0]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
          m_new[2] += (mom[1]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
          m_new[3] += (mom[2]*i0_(m,n,k,j,i)*solid_angles_.d_view(n));
        }

        // feedback on fluid
        if (affect_fluid_) {
          u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
          u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
          u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
          u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
        }
      } else {
        // NOTE(@pdmullen): At this point, it is possible that excision has not been
        // entirely applied if Compton is enabled and a badcell or temperature equilibrium
        // was encountered.. apply excision
        if (excise) {
          for (int n=0; n<=nang1; ++n) {
            Real n_0 = 1.0;
            if (!(use_adm_geometry_)) {
              n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+
                    tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)+
                    tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+
                    tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
            }
            if (rad_mask_(m,k,j,i) || (!(use_adm_geometry_) && fabs(n_0) < n_0_floor_)) {
              i0_(m,n,k,j,i) = 0.0;
            }
          }
        }
      }
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  bool FourthPolyRoot
//  \brief Bracketed monotone solve for fourth order polynomial of
//  the form coef4 * x^4 + x + tconst = 0.

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root) {
  if (!(Kokkos::isfinite(coef4)) || !(Kokkos::isfinite(tconst)) || coef4 < 0.0) {
    return false;
  }
  if (fabs(coef4) <= 1.0e-300) {
    root = -tconst;
    return (root >= 0.0 && Kokkos::isfinite(root));
  }

  // For coef4 >= 0, f(x)=coef4*x^4+x+tconst is monotone on x >= 0.
  // A positive root exists only when f(0) <= 0.
  if (tconst > 0.0) { return false; }

  Real lo = 0.0;
  Real hi = fmax(1.0, root);
  if (!(Kokkos::isfinite(hi)) || hi <= 0.0) { hi = 1.0; }
  bool bracketed = false;
  for (int it=0; it<128; ++it) {
    Real fhi = coef4*SQR(SQR(hi)) + hi + tconst;
    if (!(Kokkos::isfinite(fhi))) { return false; }
    if (fhi >= 0.0) {
      bracketed = true;
      break;
    }
    hi *= 2.0;
    if (!(Kokkos::isfinite(hi))) { return false; }
  }
  if (!(bracketed)) { return false; }

  Real x = fmin(fmax(root, lo), hi);
  if (x <= lo || x >= hi) { x = 0.5*(lo + hi); }
  const Real ftol = 1.0e-13*(1.0 + fabs(tconst));
  for (int it=0; it<80; ++it) {
    const Real f = coef4*SQR(SQR(x)) + x + tconst;
    if (!(Kokkos::isfinite(f))) { return false; }
    if (fabs(f) <= ftol) {
      root = x;
      return true;
    }
    if (f > 0.0) {
      hi = x;
    } else {
      lo = x;
    }

    const Real df = 4.0*coef4*x*x*x + 1.0;
    Real xnew = x - f/df;
    if (!(Kokkos::isfinite(xnew)) || xnew <= lo || xnew >= hi) {
      xnew = 0.5*(lo + hi);
    }
    x = xnew;
  }

  root = x;
  return (root >= 0.0 && Kokkos::isfinite(root));
}

} // namespace dyn_radiation
