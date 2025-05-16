#ifndef RADIATION_M1_NURATES_HPP
#define RADIATION_M1_NURATES_HPP
#ifdef ENABLE_NURATES

#include "athena.hpp"
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "radiation_m1_fermi.hpp"
#include "radiation_m1_params.hpp"

namespace radiationm1 {

struct NuratesParams {
  int nurates_quad_nx;     // no. of quadrature points for 1d integration (bns_nurates)
  int nurates_quad_ny;     // no. of quadrature points for 2d integration (bns_nurates)
  Real opacity_tau_trap;   // incl. effects of neutrino trapping above this optical depth
  Real opacity_tau_delta;  // range of optical depths over which trapping is introduced
  Real opacity_corr_fac_max;  // maximum correction factor for optically thin regime
  Real rho_min_cgs;
  Real temp_min_mev;

  bool use_abs_em;
  bool use_pair;
  bool use_brem;
  bool use_iso;
  bool use_inelastic_scatt;
  bool use_WM_ab;
  bool use_WM_sc;
  bool use_dU;
  bool use_dm_eff;
  bool use_equilibrium_distribution;
  bool use_kirchhoff_law;

  MyQuadrature my_quadrature_1d;
  MyQuadrature my_quadrature_2d;
};

//! \fn void bns_nurates(Real &nb, Real &temp, Real &ye, Real &mu_n, Real &mu_p,
//!                      Real &mu_e, Real &n_nue, Real &j_nue, Real &chi_nue,
//!                      Real &n_anue, Real &j_anue, Real &chi_anue, Real &n_nux,
//!                      Real &j_nux, Real &chi_nux, Real &n_anux, Real &j_anux,
//!                      Real &chi_anux, Real &R_nue, Real &R_anue, Real &R_nux,
//!                      Real &R_anux, Real &Q_nue, Real &Q_anue, Real &Q_nux,
//!                      Real &Q_anux, Real &sigma_0_nue, Real &sigma_0_anue,
//!                      Real &sigma_0_nux, Real &sigma_0_anux, Real &sigma_1_nue,
//!                      Real &sigma_1_anue, Real &sigma_1_nux, Real &sigma_1_anux,
//!                      Real &scat_0_nue, Real &scat_0_anue, Real &scat_0_nux,
//!                      Real &scat_0_anux, Real &scat_1_nue, Real &scat_1_anue,
//!                      Real &scat_1_nux, Real &scat_1_anux,
//!                      const NuratesParams nurates_params)
//   \brief Computes the rates given the M1 quantities
//
//   \note  All input and output quantities are in code units, except temperature (MeV)
//
//   \param[in] nb              baryon number density
//   \param[in] temp            temperature (MeV)
//   \param[in] ye              electron fraction
//   \param[in] mu_n            neutron chemical potential
//   \param[in] mu_p            proton chemical potential
//   \param[in] mu_e            electron chemical potential
//   \param[in] n_nue           number density electron neutrinos
//   \param[in] j_nue           energy density electron neutrinos
//   \param[in] chi_nue         eddington factor electron neutrinos
//   \param[in] n_anue          number density electron anti-neutrinos
//   \param[in] j_anue          energy density electron anti-neutrinos
//   \param[in] chi_anue        eddington factor electron anti-neutrinos
//   \param[in] n_nux           number density mu/tau neutrinos
//   \param[in] j_nux           energy density mu/tau neutrinos
//   \param[in] chi_nux         eddington factor mu/tau neutrinos
//   \param[in] n_anux          number density mu/tau neutrinos
//   \param[in] j_anux          energy density mu/tau neutrinos
//   \param[in] chi_anux        eddington factor mu/tau neutrinos
//
//   \param[out] R_nue          number emissivity electron neutrinos
//   \param[out] R_anue         number emissivity electron anti-neutrinos
//   \param[out] R_nux          number emissivity mu/tau neutrinos
//   \param[out] R_anux         number emissivity mu/tau anti-neutrinos
//   \param[out] Q_nue          energy emissivity electron neutrinos
//   \param[out] Q_anue         energy emissivity electron anti-neutrinos
//   \param[out] Q_nux          energy emissivity mu/tau neutrinos
//   \param[out] Q_anux         energy emissivity mu/tau anti-neutrinos
//   \param[out] sigma_0_nue    number inv mean-free path electron neutrinos
//   \param[out] sigma_0_anue   number inv mean-free path electron anti-neutrinos
//   \param[out] sigma_0_nux    number inv mean-free path mu/tau neutrinos
//   \param[out] sigma_0_anux   number inv mean-free path mu/tau anti-neutrinos
//   \param[out] sigma_1_nue    energy inv mean-free path electron neutrinos
//   \param[out] sigma_1_anue   energy inv mean-free path electron anti-neutrinos
//   \param[out] sigma_1_nux    energy inv mean-free path mu/tau neutrinos
//   \param[out] sigma_1_anux   energy inv mean-free path mu/tau neutrinos
//   \param[out] scat_0_nue     number scatt coeff electron neutrinos
//   \param[out] scat_0_anue    number scatt coeff electron anti-neutrinos
//   \param[out] scat_0_nux     number scatt coeff mu/tau neutrinos
//   \param[out] scat_0_anux    number scatt coeff mu/tau anti-neutrinos
//   \param[out] scat_1_nue     energy scatt coeff electron neutrinos
//   \param[out] scat_1_anue    energy scatt coeff electron ant-neutrinos
//   \param[out] scat_1_nux     energy scatt coeff mu/tau neutrinos
//   \param[out] scat_1_anux    energy scatt coeff mu/tau anti-neutrinos
//   \param[in]  nurates_params params for nurates

KOKKOS_INLINE_FUNCTION
void bns_nurates(Real &nb, Real &temp, Real &ye, Real &mu_n, Real &mu_p, Real &mu_e,
                 Real &n_nue, Real &j_nue, Real &chi_nue, Real &n_anue, Real &j_anue,
                 Real &chi_anue, Real &n_nux, Real &j_nux, Real &chi_nux, Real &n_anux,
                 Real &j_anux, Real &chi_anux, Real &R_nue, Real &R_anue, Real &R_nux,
                 Real &R_anux, Real &Q_nue, Real &Q_anue, Real &Q_nux, Real &Q_anux,
                 Real &sigma_0_nue, Real &sigma_0_anue, Real &sigma_0_nux,
                 Real &sigma_0_anux, Real &sigma_1_nue, Real &sigma_1_anue,
                 Real &sigma_1_nux, Real &sigma_1_anux, Real &scat_0_nue,
                 Real &scat_0_anue, Real &scat_0_nux, Real &scat_0_anux, Real &scat_1_nue,
                 Real &scat_1_anue, Real &scat_1_nux, Real &scat_1_anux,
                 const NuratesParams nurates_params, const RadiationM1Units units) {
  // unit conversion factors
  // from cm^-3 to code
  const Real n_cgs2code = 1. / Kokkos::pow(units.cgs2code_length, 3);
  // from MeV cm^-3 to code
  const Real j_cgs2code = units.cgs2code_energy / Kokkos::pow(units.cgs2code_length, 3);
  // from cm^-3 s^-1 to code
  const Real r_cgs2code =
      1. / (units.cgs2code_time * Kokkos::pow(units.cgs2code_length, 3));
  // from MeV cm^-3 s^-1 to code
  const Real q_cgs2code = units.mev_to_erg * units.cgs2code_energy /
                          (units.cgs2code_time * Kokkos::pow(units.cgs2code_length, 3));
  // from cm^-1 to code
  const Real kappa_cgs2code = 1. / (units.cgs2code_length);

  if (nb < nurates_params.rho_min_cgs || temp < nurates_params.temp_min_mev) {
    R_nue = 0.;
    R_anue = 0.;
    R_nux = 0.;
    R_anux = 0.;
    Q_nue = 0.;
    Q_anue = 0.;
    Q_nux = 0.;
    Q_anux = 0.;
    sigma_0_nue = 0.;
    sigma_0_anue = 0.;
    sigma_0_nux = 0.;
    sigma_0_anux = 0.;
    sigma_1_nue = 0.;
    sigma_1_anue = 0.;
    sigma_1_nux = 0.;
    sigma_1_anux = 0.;
    scat_0_nue = 0.;
    scat_0_anue = 0.;
    scat_0_nux = 0.;
    scat_0_anux = 0.;
    scat_1_nue = 0.;
    scat_1_anue = 0.;
    scat_1_nux = 0.;
    scat_1_anux = 0.;
    return;
  }

  // convert neutrino quantities to cgs
  const Real n_nue_cgs = n_nue / (n_cgs2code/units.normfact) * 1e-21;
  const Real n_anue_cgs = n_anue / (n_cgs2code/units.normfact) * 1e-21;
  const Real n_nux_cgs = n_nux / (n_cgs2code/units.normfact) * 1e-21;
  const Real n_anux_cgs = n_anux / (n_cgs2code/units.normfact) * 1e-21;
  const Real j_nue_cgs = j_nue / j_cgs2code;
  const Real j_anue_cgs = j_anue / j_cgs2code;
  const Real j_nux_cgs = j_nux / j_cgs2code;
  const Real j_anux_cgs = j_anux / j_cgs2code;
  const Real nb_cgs = nb; //@TODO: fix!

  // opacity params structure
  GreyOpacityParams my_grey_opacity_params{};

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

  //@TODO: logic for nurates_params.use_dU not implemented from bns_nurates_wrap.cpp

  // populate EOS quantities
  my_grey_opacity_params.eos_pars.mu_e = mu_e;
  my_grey_opacity_params.eos_pars.mu_p = mu_p;
  my_grey_opacity_params.eos_pars.mu_n = mu_n;
  my_grey_opacity_params.eos_pars.temp = temp;
  my_grey_opacity_params.eos_pars.yp = ye;
  my_grey_opacity_params.eos_pars.yn = 1 - ye;
  my_grey_opacity_params.eos_pars.nb = nb_cgs;

  // populate M1 quantities
  // The factors of 1/2 come from the fact that bns_nurates and THC weight the
  // heavy neutrinos differently. THC weights them with a factor of 2 (because
  // "nux" means "mu AND tau"), bns_nurates with a factor of 1 (because "nux"
  // means "mu OR tau").
  my_grey_opacity_params.m1_pars.n[id_nue] = n_nue_cgs;
  my_grey_opacity_params.m1_pars.J[id_nue] = j_nue_cgs;
  my_grey_opacity_params.m1_pars.chi[id_nue] = chi_nue;
  my_grey_opacity_params.m1_pars.n[id_anue] = n_anue_cgs;
  my_grey_opacity_params.m1_pars.J[id_anue] = j_anue_cgs;
  my_grey_opacity_params.m1_pars.chi[id_anue] = chi_anue;
  my_grey_opacity_params.m1_pars.n[id_nux] = n_nux_cgs * 0.5;
  my_grey_opacity_params.m1_pars.J[id_nux] = j_nux_cgs * 0.5;
  my_grey_opacity_params.m1_pars.chi[id_nux] = chi_nux;
  my_grey_opacity_params.m1_pars.n[id_anux] = n_anux_cgs * 0.5;
  my_grey_opacity_params.m1_pars.J[id_anux] = j_anux_cgs * 0.5;
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

  // convert to code units
   R_nue = R_nue * (r_cgs2code/units.normfact) * 1e21;
   R_anue = R_anue * (r_cgs2code/units.normfact) * 1e21;
   R_nux = R_nux * (r_cgs2code/units.normfact) * 1e21;
   R_anux = R_anux * (r_cgs2code/units.normfact) * 1e21;
   Q_nue = Q_nue * q_cgs2code * 1e21;
   Q_anue = Q_anue * q_cgs2code * 1e21;
   Q_nux = Q_nux * q_cgs2code * 1e21;
   Q_anux = Q_anux * q_cgs2code * 1e21;
   sigma_0_nue = sigma_0_nue * kappa_cgs2code * 1e7;
   sigma_0_anue = sigma_0_anue * kappa_cgs2code * 1e7;
   sigma_0_nux = sigma_0_nux * kappa_cgs2code * 1e7;
   sigma_0_anux = sigma_0_anux * kappa_cgs2code * 1e7;
   sigma_1_nue = sigma_1_nue * kappa_cgs2code * 1e7;
   sigma_1_anue = sigma_1_anue * kappa_cgs2code * 1e7;
   sigma_1_nux = sigma_1_nux * kappa_cgs2code * 1e7;
   sigma_1_anux = sigma_1_anux * kappa_cgs2code * 1e7;
   scat_0_nue = scat_0_nue * kappa_cgs2code * 1e7;
   scat_0_anue = scat_0_anue * kappa_cgs2code * 1e7;
   scat_0_nux = scat_0_nux * kappa_cgs2code * 1e7;
   scat_0_anux = scat_0_anux * kappa_cgs2code * 1e7;
   scat_1_nue = scat_1_nue * kappa_cgs2code * 1e7;
   scat_1_anue = scat_1_anue * kappa_cgs2code * 1e7;
   scat_1_nux = scat_1_nux * kappa_cgs2code * 1e7;
   scat_1_anux = scat_1_anux * kappa_cgs2code * 1e7;
}

//! \fn void NeutrinoDens(Real mu_n, Real mu_p, Real mu_e, Real nb, Real temp,
//!                       Real &n_nue, Real &n_anue, Real &n_nux, Real &en_nue,
//!                       Real &en_anue, Real &en_nux, NuratesParams nurates_params)
//
//   \brief Computes the neutrino number and energy density
//
//   \note  All input and output quantities are in code units, except temperature (MeV)
//
//   \param[in]  mu_n            neutron chemical potential
//   \param[in]  mu_p            proton chemical potential
//   \param[in]  mu_e            electron chemical potential
//   \param[in]  nb              baryon number density
//   \param[in]  temp            temperature (MeV).
//   \param[out] n_nue           number density electron neutrinos
//   \param[out] n_anue          number density electron anti-neutrinos
//   \param[out] n_nux           number density mu/tau neutrinos
//   \param[out] en_nue          energy density electron neutrinos
//   \param[out] en_anue         energy density electron anti-neutrinos
//   \param[out] en_nux          energy density mu/tau neutrinos
//   \param[in]  nurates_params  struct for nurates parameters
KOKKOS_INLINE_FUNCTION
void NeutrinoDens(Real mu_n, Real mu_p, Real mu_e, Real nb, Real temp, Real &n_nue,
                  Real &n_anue, Real &n_nux, Real &en_nue, Real &en_anue, Real &en_nux,
                  NuratesParams nurates_params, RadiationM1Units units) {
  if ((nb < nurates_params.rho_min_cgs) || (temp < nurates_params.temp_min_mev)) {
    n_nue = 0.;
    n_anue = 0.;
    n_nux = 0.;
    en_nue = 0.;
    en_anue = 0.;
    en_nux = 0.;
    return;
  }

  Real eta_nue = (mu_p + mu_e - mu_n) / temp;
  Real eta_anue = -eta_nue;
  Real eta_nux = 0.0;

  n_nue = 4.0 * M_PI / Kokkos::pow(HC_MEVCM, 3) * Kokkos::pow(temp, 3) *
          Fermi::fermi2(eta_nue);
  n_anue = 4.0 * M_PI / Kokkos::pow(HC_MEVCM, 3) * Kokkos::pow(temp, 3) *
           Fermi::fermi2(eta_anue);
  n_nux = 16.0 * M_PI / Kokkos::pow(HC_MEVCM, 3) * Kokkos::pow(temp, 3) *
          Fermi::fermi2(eta_nux);

  en_nue = 4.0 * M_PI / Kokkos::pow(HC_MEVCM, 3) * Kokkos::pow(temp, 4) *
           Fermi::fermi3(eta_nue);
  en_anue = 4.0 * M_PI / Kokkos::pow(HC_MEVCM, 3) * Kokkos::pow(temp, 4) *
            Fermi::fermi3(eta_anue);
  en_nux = 16.0 * M_PI / Kokkos::pow(HC_MEVCM, 3) * Kokkos::pow(temp, 4) *
           Fermi::fermi3(eta_nux);

  assert(isfinite(n_nue));
  assert(isfinite(n_nua));
  assert(isfinite(n_nux));
  assert(isfinite(en_nue));
  assert(isfinite(en_nua));
  assert(isfinite(en_nux));

  const Real fact1 = Kokkos::pow(units.cgs2code_length, 3) * units.normfact;
  n_nue = n_nue / fact1;
  n_anue = n_anue / fact1;
  n_nux = n_nux / fact1;

  const Real fact2 =
      units.mev_to_erg * units.cgs2code_energy / Kokkos::pow(units.cgs2code_length, 3);
  en_nue = en_nue / fact2;
  en_anue = en_anue / fact2;
  en_nux = en_nux / fact2;
}

}  // namespace radiationm1
#endif  // ENABLE_NURATES
#endif  // RADIATION_M1_NURATES_HPP
