//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_nurates.cpp
//! \brief functions for opacity computation using bns-nurates

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_calc_closure.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

namespace radiationm1 {

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