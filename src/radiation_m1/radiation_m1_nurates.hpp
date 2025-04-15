#ifndef RADIATION_M1_NURATES_HPP
#define RADIATION_M1_NURATES_HPP

#include "athena.hpp"
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"

namespace radiationm1 {
KOKKOS_INLINE_FUNCTION
Real AverageBaryonMass() { return 1; }

KOKKOS_INLINE_FUNCTION
Real NeutrinoDens2_cgs(Real rho, Real temp, Real ye, Real &n_nue, Real n_nua, Real n_nux, Real en_nue, Real en_nua, Real en_nux){
    /*
    int NeuDens_cgs = 0;

    Real lrho  = Kokkos::log10(rho);
    Real ltemp = Kokkos::log10(temp);

    Real mass_fact_cgs = mass_fact * mev_to_erg / (clight*clight);
    Real nb = rho / mass_fact_cgs;

    Real mu_n = wkLinearInterpolation3d(lrho, ltemp, ye, MU_N);
    Real mu_p = wkLinearInterpolation3d(lrho, ltemp, ye, MU_P);
    Real mu_e = wkLinearInterpolation3d(lrho, ltemp, ye, MU_E);

    Real eta_nue = (mu_p + mu_e - mu_n) / temp;
    Real eta_nua = -eta_nue;
    Real eta_nux = 0.0;

    Real n_nue = 4.0 * Kokkos::numbers::pi / hc_mevcm**3 * temp**3 * FERMI2(eta_nue);
    Real n_nua = 4.0 * Kokkos::numbers::pi / hc_mevcm**3 * temp**3 * FERMI2(eta_nua);
    Real n_nux = 16.0 * Kokkos::numbers::pi / hc_mevcm**3 * temp**3 * FERMI2(eta_nux);

    Real en_nue = 4.0 * Kokkos::numbers::pi / hc_mevcm**3 * temp**4 * FERMI3(eta_nue);
    Real en_nua = 4.0 * Kokkos::numbers::pi / hc_mevcm**3 * temp**4 * FERMI3(eta_nua);
    Real en_nux = 16.0 * Kokkos::numbers::pi / hc_mevcm**3 * temp**4 * FERMI3(eta_nux);

    return NeuDens_cgs; */
}
KOKKOS_INLINE_FUNCTION
Real NeutrinoDensity(Real rho, Real temp, Real ye, Real &num_nue, Real &num_nua,
                     Real &num_nux, Real &ene_nue, Real &ene_nua, Real &ene_nux) {
  int NuDens = 0;

  Real rho_cgs = rho * 1;  // fix conversion factor
  Real temp0 = temp;
  Real ye0 = ye;

  if ((rho_cgs < 1) || (temp0 < 1)) {
    num_nue = 0;
    num_nua = 0;
    num_nux = 0;
    ene_nue = 0;
    ene_nua = 0;
    ene_nux = 0;
    return NuDens;
  }
  /*
  auto boundsErr = enforceTableBounds(rho_cgs, temp0, ye0);

  if (boundsErr == -1) {
    NuDens = -1;
    return NuDens;
  }

  auto ierr = NeutrinoDens_cgs(rho_cgs, temp0, ye0, &num_nue, num_nua, num_nux, &ene_nue,
                               ene_nua, ene_nux);

  if (ierr != 0) {
    NuDens = -1;
  }

  num_nue = num_nue / (cgs2cactusLength * *3 * normfact);
  num_nua = num_nua / (cgs2cactusLength * *3 * normfact);
  num_nux = num_nux / (cgs2cactusLength * *3 * normfact);

  ene_nue = ene_nue * (mev_to_erg * cgs2cactusEnergy / cgs2cactusLength * *3);
  ene_nua = ene_nua * (mev_to_erg * cgs2cactusEnergy / cgs2cactusLength * *3);
  ene_nux = ene_nux * (mev_to_erg * cgs2cactusEnergy / cgs2cactusLength * *3);

  return NuDens; */
}

KOKKOS_INLINE_FUNCTION
Real NucleiAbar(Real rho, Real temp, Real ye, Real &abar) {
  int NucleiAbar = 0;
  /*
  Real rho_cgs = rho * 1; //fix this factor
  Real temp0 = temp;
  Real ye0 = ye;

  auto boundsErr = enforceTableBounds(rho_cgs, temp0, ye0);

  if (boundsErr == 1) {
    NucleiAbar = 1;
  }

  Real lrho0 = Kokkos::log10(rho_cgs);
  Real ltemp0 = Kokkos::log10(temp0);

  abar = wkLinearInterpolation3d(lrho,ltem0,ye0,ABAR?);
  return NucleiAbar;*/
}

KOKKOS_INLINE_FUNCTION
void bns_nurates(const Real &nb, const Real &temp, const Real &ye, const Real &mu_n,
                 const Real &mu_p, const Real &mu_e, const Real &n_nue, const Real &j_nue,
                 const Real &chi_nue, const Real &n_nua, const Real &j_nua,
                 const Real &chi_nua, const Real &n_nux, const Real &j_nux,
                 const Real &chi_nux, Real &R_nue, Real &R_nua, Real &R_nux, Real &Q_nue,
                 Real &Q_nua, Real &Q_nux, Real &sigma_0_nue, Real &sigma_0_nua,
                 Real &sigma_0_nux, Real &sigma_1_nue, Real &sigma_1_nua,
                 Real &sigma_1_nux, Real &scat_0_nue, Real &scat_0_nua, Real &scat_0_nux,
                 Real &scat_1_nue, Real &scat_1_nua, Real &scat_1_nux,
                 const NuratesParams nurates_params) {
  // opacity params structure
  GreyOpacityParams my_grey_opacity_params = {0};

  // reaction flags
  my_grey_opacity_params.opacity_flags = opacity_flags_default_none;
  my_grey_opacity_params.opacity_flags.use_abs_em = nurates_params.use_abs_em;
  my_grey_opacity_params.opacity_flags.use_brem = nurates_params.use_brem;
  my_grey_opacity_params.opacity_flags.use_pair = nurates_params.use_pair;
  my_grey_opacity_params.opacity_flags.use_iso = nurates_params.use_iso_scat;

  // other flags
  my_grey_opacity_params.opacity_pars.use_WM_ab = nurates_params.use_WM_ab;
  my_grey_opacity_params.opacity_pars.use_WM_sc = nurates_params.use_WM_sc;
  my_grey_opacity_params.opacity_pars.use_dU = nurates_params.use_dU;

  // populate EOS quantities
  my_grey_opacity_params.eos_pars.mu_e = mu_e;
  my_grey_opacity_params.eos_pars.mu_p = mu_p;
  my_grey_opacity_params.eos_pars.mu_n = mu_n;
  my_grey_opacity_params.eos_pars.temp = temp;
  my_grey_opacity_params.eos_pars.yp = ye;
  my_grey_opacity_params.eos_pars.yn = 1 - ye;
  my_grey_opacity_params.eos_pars.nb = nb;

  // populate M1 quantities
  my_grey_opacity_params.m1_pars.n[id_nue] = n_nue;
  my_grey_opacity_params.m1_pars.J[id_nue] = j_nue;
  my_grey_opacity_params.m1_pars.chi[id_nue] = chi_nue;
  my_grey_opacity_params.m1_pars.n[id_anue] = n_nua;
  my_grey_opacity_params.m1_pars.J[id_anue] = j_nua;
  my_grey_opacity_params.m1_pars.chi[id_anue] = chi_nua;
  // note that bns_nurates take nux to be one of the nux's
  my_grey_opacity_params.m1_pars.n[id_nux] = n_nux / 4;
  my_grey_opacity_params.m1_pars.J[id_nux] = j_nux / 4;
  my_grey_opacity_params.m1_pars.chi[id_nux] = chi_nux;

  if (!nurates_params.use_equilibrium_distribution) {
    // reconstruct distribution function
    my_grey_opacity_params.distr_pars = CalculateDistrParamsFromM1(
        &my_grey_opacity_params.m1_pars, &my_grey_opacity_params.eos_pars);
  } else {
    // reconstruct distribution function
    my_grey_opacity_params.distr_pars =
        NuEquilibriumParams(&my_grey_opacity_params.eos_pars);

    // compute n and j
    ComputeM1DensitiesEq(&my_grey_opacity_params.eos_pars,
                         &my_grey_opacity_params.distr_pars,
                         &my_grey_opacity_params.m1_pars);

    for (int idx = 0; idx < total_num_species; idx++) {
      my_grey_opacity_params.m1_pars.chi[idx] = 1. / 3.;
      my_grey_opacity_params.m1_pars.J[idx] =
          my_grey_opacity_params.m1_pars.J[idx] *
          kBS_MeV;  //@TODO: check that this is sensible
    }
  }

  // compute opacities
  M1Opacities opacities =
      ComputeM1Opacities(&nurates_params.my_quadrature_1d,
                         &nurates_params.my_quadrature_2d, &my_grey_opacity_params);

  // extract emissivities
  R_nue = opacities.eta_0[id_nue];
  R_nua = opacities.eta_0[id_anue];
  R_nux = 4.0 * opacities.eta_0[id_nux];
  Q_nue = opacities.eta[id_nue];
  Q_nua = opacities.eta[id_anue];
  Q_nux = 4.0 * opacities.eta[id_nux];

  // extract absorption inverse mean-free path
  sigma_0_nue = opacities.kappa_0_a[id_nue];
  sigma_0_nua = opacities.kappa_0_a[id_anue];
  sigma_0_nux = opacities.kappa_0_a[id_nux];
  sigma_1_nue = opacities.kappa_a[id_nue];
  sigma_1_nua = opacities.kappa_a[id_anue];
  sigma_1_nux = opacities.kappa_a[id_nux];

  // extract scattering inverse mean-free path
  scat_0_nue = 0;
  scat_0_nua = 0;
  scat_0_nux = 0;
  scat_1_nue = opacities.kappa_s[id_nue];
  scat_1_nua = opacities.kappa_s[id_anue];
  scat_1_nux = opacities.kappa_s[id_nux];
}
}  // namespace radiationm1

#endif  // RADIATION_M1_NURATES_HPP
