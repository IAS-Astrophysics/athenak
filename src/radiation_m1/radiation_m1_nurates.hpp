#ifndef RADIATION_M1_NURATES_HPP
#define RADIATION_M1_NURATES_HPP

#include "athena.hpp"
#include "radiation_m1_params.hpp"

#include "eos/primitive-solver/eos.hpp"

#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"

namespace radiationm1 {

KOKKOS_INLINE_FUNCTION
void bns_nurates(Real &nb, Real &temp, Real &ye, Real &mu_n, Real &mu_p, Real &mu_e,
                 Real &n_nue, Real &j_nue, Real &chi_nue,
                 Real &n_anue, Real &j_anue, Real &chi_anue,
                 Real &n_nux, Real &j_nux, Real &chi_nux,
                 Real &n_anux, Real &j_anux, Real &chi_anux, 
                 Real &R_nue, Real &R_anue, Real &R_nux, Real &R_anux,
                 Real &Q_nue, Real &Q_anue, Real &Q_nux, Real &Q_anux,
                 Real &sigma_0_nue, Real &sigma_0_anue, Real &sigma_0_nux, Real &sigma_0_anux,
                 Real &sigma_1_nue, Real &sigma_1_anue, Real &sigma_1_nux, Real &sigma_1_anux,
                 Real &scat_0_nue, Real &scat_0_anue, Real &scat_0_nux, Real &scat_0_anux,
                 Real &scat_1_nue, Real &scat_1_anue, Real &scat_1_nux, Real &scat_1_anux,
                 const NuratesParams nurates_params) {

  // TODO convert units
    
  // opacity params structure
  GreyOpacityParams my_grey_opacity_params = {0};

  // reaction flags
  my_grey_opacity_params.opacity_flags = opacity_flags_default_none;
  my_grey_opacity_params.opacity_flags.use_abs_em = nurates_params.use_abs_em;
  my_grey_opacity_params.opacity_flags.use_brem = nurates_params.use_brem;
  my_grey_opacity_params.opacity_flags.use_pair = nurates_params.use_pair;
  my_grey_opacity_params.opacity_flags.use_iso = nurates_params.use_iso;
  my_grey_opacity_params.opacity_flags.use_inelastic_scatt =
      nurates_params.use_inelastic_scatt;

  // other flags
  my_grey_opacity_params.opacity_pars = opacity_params_default_none;
  my_grey_opacity_params.opacity_pars.use_WM_ab = nurates_params.use_WM_ab;
  my_grey_opacity_params.opacity_pars.use_WM_sc = nurates_params.use_WM_sc;
  my_grey_opacity_params.opacity_pars.use_dU = nurates_params.use_dU;
  my_grey_opacity_params.opacity_pars.use_dm_eff = nurates_params.use_dm_eff;

  // populate EOS quantities
  my_grey_opacity_params.eos_pars.mu_e = mu_e;
  my_grey_opacity_params.eos_pars.mu_p = mu_p;
  my_grey_opacity_params.eos_pars.mu_n = mu_n;
  my_grey_opacity_params.eos_pars.temp = temp;
  my_grey_opacity_params.eos_pars.yp = ye;
  my_grey_opacity_params.eos_pars.yn = 1 - ye;
  my_grey_opacity_params.eos_pars.nb = nb;

  // populate M1 quantities
  // The factors of 1/2 come from the fact that bns_nurates and THC weight the
  // heavy neutrinos differently. THC weights them with a factor of 2 (because
  // "nux" means "mu AND tau"), bns_nurates with a factor of 1 (because "nux"
  // means "mu OR tau").
  my_grey_opacity_params.m1_pars.n[id_nue] = n_nue;
  my_grey_opacity_params.m1_pars.J[id_nue] = j_nue;
  my_grey_opacity_params.m1_pars.chi[id_nue] = chi_nue;
  my_grey_opacity_params.m1_pars.n[id_anue] = n_anue;
  my_grey_opacity_params.m1_pars.J[id_anue] = j_anue;
  my_grey_opacity_params.m1_pars.chi[id_anue] = chi_anue;
  my_grey_opacity_params.m1_pars.n[id_nux] = n_nux * 0.5;
  my_grey_opacity_params.m1_pars.J[id_nux] = j_nux * 0.5;
  my_grey_opacity_params.m1_pars.chi[id_nux] = chi_nux;
  my_grey_opacity_params.m1_pars.n[id_anux] = n_anux * 0.5;
  my_grey_opacity_params.m1_pars.J[id_anux] = j_anux * 0.5;
  my_grey_opacity_params.m1_pars.chi[id_anux] = chi_anux;

  // reconstruct distribution function
  if (!nurates_params.use_equilibrium_distribution) {
    my_grey_opacity_params.distr_pars = CalculateDistrParamsFromM1(
        &my_grey_opacity_params.m1_pars, &my_grey_opacity_params.eos_pars);
  } else {
    my_grey_opacity_params.distr_pars =
        NuEquilibriumParams(&my_grey_opacity_params.eos_pars);

    // compute neutrino number and energy densities
    ComputeM1DensitiesEq(&my_grey_opacity_params.eos_pars,
                         &my_grey_opacity_params.distr_pars,
                         &my_grey_opacity_params.m1_pars);

    // populate M1 quantities
    my_grey_opacity_params.m1_pars.chi[id_nue] = 0.333333333333333333333333333;
    my_grey_opacity_params.m1_pars.chi[id_anue] = 0.333333333333333333333333333;
    my_grey_opacity_params.m1_pars.chi[id_nux] = 0.333333333333333333333333333;
    my_grey_opacity_params.m1_pars.chi[id_anux] = 0.333333333333333333333333333;

    // convert neutrino energy density to mixed MeV and cgs as requested by bns_nurates
    my_grey_opacity_params.m1_pars.J[id_nue] *= kBS_MeV;
    my_grey_opacity_params.m1_pars.J[id_anue] *= kBS_MeV;
    my_grey_opacity_params.m1_pars.J[id_nux] *= kBS_MeV;
    my_grey_opacity_params.m1_pars.J[id_anux] *= kBS_MeV;
  }
  // compute opacities
  M1Opacities opacities =
      ComputeM1Opacities(&nurates_params.my_quadrature_1d,
                         &nurates_params.my_quadrature_1d, &my_grey_opacity_params);

  // Similar to the comment above, the factors of 2 come from the fact that
  // bns_nurates and THC weight the heavy neutrinos differently. THC weights
  // them with a factor of 2 (because "nux" means "mu AND tau"), bns_nurates
  // with a factor of 1 (because "nux" means "mu OR tau").

  // extract emissivities
  R_nue = opacities.eta_0[id_nue];
  R_anue = opacities.eta_0[id_anue];
  R_nux = opacities.eta_0[id_nux] * 2.;
  R_anux = opacities.eta_0[id_anux] * 2.;
  Q_nue = opacities.eta[id_nue];
  Q_anue = opacities.eta[id_anue];
  Q_nux = opacities.eta[id_nux] * 2.;
  Q_anux = opacities.eta[id_anux] * 2.;

  // extract absorption inverse mean-free path
  sigma_0_nue = opacities.kappa_0_a[id_nue];
  sigma_0_anue = opacities.kappa_0_a[id_anue];
  sigma_0_nux = opacities.kappa_0_a[id_nux] * 2.;
  sigma_0_anux = opacities.kappa_0_a[id_anux] * 2.;
  sigma_1_nue = opacities.kappa_a[id_nue];
  sigma_1_anue = opacities.kappa_a[id_anue];
  sigma_1_nux = opacities.kappa_a[id_nux] * 2.;
  sigma_1_anux = opacities.kappa_a[id_anux] * 2.;

  // extract scattering inverse mean-free path
  scat_0_nue = 0;
  scat_0_anue = 0;
  scat_0_nux = 0;
  scat_0_anux = 0;
  scat_1_nue = opacities.kappa_s[id_nue];
  scat_1_anue = opacities.kappa_s[id_anue];
  scat_1_nux = opacities.kappa_s[id_nux] * 2.;
  scat_1_anux = opacities.kappa_s[id_anux] * 2.;

  // TODO: convert units

  // Check for NaNs/Infs
  assert(isfinite(R_nue));
  assert(isfinite(R_anue));
  assert(isfinite(R_nux));
  assert(isfinite(R_anux));
  assert(isfinite(Q_nue));
  assert(isfinite(Q_anue));
  assert(isfinite(Q_nux));
  assert(isfinite(Q_anux));
  assert(isfinite(sigma_0_nue));
  assert(isfinite(sigma_0_anue));
  assert(isfinite(sigma_0_nux));
  assert(isfinite(sigma_0_anux));
  assert(isfinite(sigma_1_nue));
  assert(isfinite(sigma_1_anue));
  assert(isfinite(sigma_1_nux));
  assert(isfinite(sigma_1_anux));
  assert(isfinite(scat_0_nue));
  assert(isfinite(scat_0_anue));
  assert(isfinite(scat_0_nux));
  assert(isfinite(scat_0_anux));
  assert(isfinite(scat_1_nue));
  assert(isfinite(scat_1_anue));
  assert(isfinite(scat_1_nux));
  assert(isfinite(scat_1_anux));
}

}  // namespace radiationm1

#endif  // RADIATION_M1_NURATES_HPP
