#ifndef RADIATION_M1_NURATES_HPP
#define RADIATION_M1_NURATES_HPP

#include "config.hpp"

#if ENABLE_NURATES

#include "athena.hpp"
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1_fermi.hpp"
#include "radiation_m1_params.hpp"

namespace radiationm1 {

struct NuratesParams {
  Real opacity_tau_trap;   // incl. effects of neutrino trapping above this optical depth
  Real opacity_tau_delta;  // range of optical depths over which trapping is introduced
  Real opacity_corr_fac_max;  // maximum correction factor for optically thin regime
  Real nb_min;
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
  bool use_NN_medium_corr;
  bool neglect_blocking;
  bool use_decay;
  bool use_BRT_brem;

  int quad_nx;  // no. of quadrature points for 1d integration (bns_nurates)
  MyQuadrature quadrature;
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
//
//   \brief Computes the rates given the M1 quantities
//
//   \param[in] nb              baryon number density (fm^-3)
//   \param[in] temp            temperature (MeV)
//   \param[in] ye              electron fraction (dimensionless)
//   \param[in] mu_n            neutron chemical potential (MeV)
//   \param[in] mu_p            proton chemical potential (MeV)
//   \param[in] mu_e            electron chemical potential (MeV)
//   \param[in] n_nue           number density e- neutrinos (code units)
//   \param[in] j_nue           energy density e- neutrinos (code units)
//   \param[in] chi_nue         eddington factor e- neutrinos (dimensionless)
//   \param[in] n_anue          number density e- anti-neutrinos (code units)
//   \param[in] j_anue          energy density e- anti-neutrinos (code units)
//   \param[in] chi_anue        eddington factor e- anti-neutrinos (dimensionless)
//   \param[in] n_nux           number density mu/tau neutrinos (code units)
//   \param[in] j_nux           energy density mu/tau neutrinos (code units)
//   \param[in] chi_nux         eddington factor mu/tau neutrinos (dimensionless)
//   \param[in] n_anux          number density mu/tau neutrinos (code units)
//   \param[in] j_anux          energy density mu/tau neutrinos (code units)
//   \param[in] chi_anux        eddington factor mu/tau neutrinos (dimensionless)
//
//   \param[out] R_nue          number emissivity e- neutrinos (code units)
//   \param[out] R_anue         number emissivity e- anti-neutrinos (code units)
//   \param[out] R_nux          number emissivity mu/tau neutrinos (code units)
//   \param[out] R_anux         number emissivity mu/tau anti-neutrinos (code units)
//   \param[out] Q_nue          energy emissivity e- neutrinos (code units)
//   \param[out] Q_anue         energy emissivity e- anti-neutrinos (code units)
//   \param[out] Q_nux          energy emissivity mu/tau neutrinos (code units)
//   \param[out] Q_anux         energy emissivity mu/tau anti-neutrinos (code units)
//   \param[out] sigma_0_nue    number inv mean-free path e- neutrinos (code units)
//   \param[out] sigma_0_anue   number inv mean-free path e- anti-neutrinos (code units)
//   \param[out] sigma_0_nux    number inv mean-free path mu/tau neutrinos (code units)
//   \param[out] sigma_0_anux   num inv mean-free path mu/tau anti-neutrinos (code units)
//   \param[out] sigma_1_nue    energy inv mean-free path e- neutrinos (code units)
//   \param[out] sigma_1_anue   energy inv mean-free path e- anti-neutrinos (code units)
//   \param[out] sigma_1_nux    energy inv mean-free path mu/tau neutrinos (code units)
//   \param[out] sigma_1_anux   energy inv mean-free path mu/tau neutrinos (code units)
//   \param[out] scat_0_nue     number scatt coeff e- neutrinos (code units)
//   \param[out] scat_0_anue    number scatt coeff e- anti-neutrinos (code units)
//   \param[out] scat_0_nux     number scatt coeff mu/tau neutrinos (code units)
//   \param[out] scat_0_anux    number scatt coeff mu/tau anti-neutrinos (code units)
//   \param[out] scat_1_nue     energy scatt coeff e- neutrinos (code units)
//   \param[out] scat_1_anue    energy scatt coeff e- ant-neutrinos (code units)
//   \param[out] scat_1_nux     energy scatt coeff mu/tau neutrinos (code units)
//   \param[out] scat_1_anux    energy scatt coeff mu/tau anti-neutrinos (code units)
//   \param[in]  nurates_params params for nurates
//   \param[in]  code_units     code units
//   \param[in]  eos_units      eos units

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
                 NuratesParams const &nurates_params,
                 Primitive::UnitSystem const &code_units,
                 Primitive::UnitSystem const &eos_units) {
  // compute unit conversion factors
  Primitive::UnitSystem nurates_units = Primitive::MakeNGS();

  Real const unit_length = code_units.LengthConversion(nurates_units);
  Real const unit_time = code_units.VolumeConversion(nurates_units);
  // Note that the number densities are always in EOS units
  Real const unit_num_dens = eos_units.NumberDensityConversion(nurates_units);
  Real const unit_ene_dens = code_units.EnergyDensityConversion(nurates_units);

  if ((nb < nurates_params.nb_min) || (temp < nurates_params.temp_min_mev)) {
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

  // populate opacity params
  GreyOpacityParams grey_op_params = {0};

  // reaction flags
  grey_op_params.opacity_flags.use_abs_em = nurates_params.use_abs_em;
  grey_op_params.opacity_flags.use_brem = nurates_params.use_brem;
  grey_op_params.opacity_flags.use_pair = nurates_params.use_pair;
  grey_op_params.opacity_flags.use_iso = nurates_params.use_iso;
  grey_op_params.opacity_flags.use_inelastic_scatt = nurates_params.use_inelastic_scatt;

  // other flags
  grey_op_params.opacity_pars.use_WM_ab = nurates_params.use_WM_ab;
  grey_op_params.opacity_pars.use_WM_sc = nurates_params.use_WM_sc;
  grey_op_params.opacity_pars.use_dU = nurates_params.use_dU;
  grey_op_params.opacity_pars.use_dm_eff = nurates_params.use_dm_eff;
  grey_op_params.opacity_pars.use_NN_medium_corr = nurates_params.use_NN_medium_corr;
  grey_op_params.opacity_pars.neglect_blocking = nurates_params.neglect_blocking;
  grey_op_params.opacity_pars.use_decay = nurates_params.use_decay;
  grey_op_params.opacity_pars.use_BRT_brem = nurates_params.use_BRT_brem;

  // populate EOS quantities
  grey_op_params.eos_pars.nb = nb * unit_num_dens;  // [baryon/nm^3]
  grey_op_params.eos_pars.temp = temp;              // [MeV]
  grey_op_params.eos_pars.yp = ye;                  // [dimensionless]
  grey_op_params.eos_pars.yn = 1 - ye;              // [dimensionless]
  grey_op_params.eos_pars.mu_e = mu_e;              // [MeV]
  grey_op_params.eos_pars.mu_p = mu_p;              // [MeV]
  grey_op_params.eos_pars.mu_n = mu_n;              // [MeV]

  // @TODO: add these quantities!
  grey_op_params.eos_pars.dU = 0;      // [MeV]
  grey_op_params.eos_pars.dm_eff = 0;  // [MeV]

  // reconstruct distribution function
  if (!nurates_params.use_equilibrium_distribution) {
    // populate M1 quantities
    // Note: factor 1/2 comes because in M1 "nux" means "mu & tau" and in bns_nurates
    // "nux" means "mu or tau"
    grey_op_params.m1_pars.n[id_nue] = n_nue * unit_num_dens;          // [nm^-3]
    grey_op_params.m1_pars.n[id_anue] = n_anue * unit_num_dens;        // [nm^-3]
    grey_op_params.m1_pars.n[id_nux] = 0.5 * n_nux * unit_num_dens;    // [nm^-3]
    grey_op_params.m1_pars.n[id_anux] = 0.5 * n_anux * unit_num_dens;  // [nm^-3]

    grey_op_params.m1_pars.J[id_nue] = j_nue * unit_ene_dens;         // [MeV nm^-3]
    grey_op_params.m1_pars.J[id_anue] = j_anue * unit_ene_dens;       // [MeV nm^-3]
    grey_op_params.m1_pars.J[id_nux] = 0.5 * j_nux * unit_ene_dens;   // [MeV nm^-3]
    grey_op_params.m1_pars.J[id_anux] = 0.5 * j_anux * unit_ene_dens; // [MeV nm^-3]

    grey_op_params.m1_pars.chi[id_nue] = chi_nue;
    grey_op_params.m1_pars.chi[id_anue] = chi_anue;
    grey_op_params.m1_pars.chi[id_nux] = chi_nux;
    grey_op_params.m1_pars.chi[id_anux] = chi_anux;

    grey_op_params.distr_pars =
        CalculateDistrParamsFromM1(&grey_op_params.m1_pars, &grey_op_params.eos_pars);
  } else {
    // compute neutrino distribution parameters assuming equilibrium
    grey_op_params.distr_pars = NuEquilibriumParams(&grey_op_params.eos_pars);

    // compute gray neutrino number and energy densities assuming equilibrium
    // N.B.: required for normalization factor of energy-averaged opacities
    ComputeM1DensitiesEq(&grey_op_params.eos_pars, &grey_op_params.distr_pars,
                         &grey_op_params.m1_pars);

    // populate eddington factor
    grey_op_params.m1_pars.chi[id_nue] = 0.333333333333333333333333333;
    grey_op_params.m1_pars.chi[id_anue] = 0.333333333333333333333333333;
    grey_op_params.m1_pars.chi[id_nux] = 0.333333333333333333333333333;
    grey_op_params.m1_pars.chi[id_anux] = 0.333333333333333333333333333;
  }
  // compute opacities
  M1Opacities opacities = ComputeM1Opacities(&nurates_params.quadrature,
                                             &nurates_params.quadrature, &grey_op_params);

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
  sigma_0_nux = opacities.kappa_0_a[id_nux];
  sigma_0_anux = opacities.kappa_0_a[id_anux];
  sigma_1_nue = opacities.kappa_a[id_nue];
  sigma_1_anue = opacities.kappa_a[id_anue];
  sigma_1_nux = opacities.kappa_a[id_nux];
  sigma_1_anux = opacities.kappa_a[id_anux];

  // extract scattering inverse mean-free path
  scat_0_nue = 0;
  scat_0_anue = 0;
  scat_0_nux = 0;
  scat_0_anux = 0;
  scat_1_nue = opacities.kappa_s[id_nue];
  scat_1_anue = opacities.kappa_s[id_anue];
  scat_1_nux = opacities.kappa_s[id_nux];
  scat_1_anux = opacities.kappa_s[id_anux];

  // Check for NaNs/Infs
  assert(Kokkos::isfinite(R_nue));
  assert(Kokkos::isfinite(R_anue));
  assert(Kokkos::isfinite(R_nux));
  assert(Kokkos::isfinite(R_anux));
  assert(Kokkos::isfinite(Q_nue));
  assert(Kokkos::isfinite(Q_anue));
  assert(Kokkos::isfinite(Q_nux));
  assert(Kokkos::isfinite(Q_anux));
  assert(Kokkos::isfinite(sigma_0_nue));
  assert(Kokkos::isfinite(sigma_0_anue));
  assert(Kokkos::isfinite(sigma_0_nux));
  assert(Kokkos::isfinite(sigma_0_anux));
  assert(Kokkos::isfinite(sigma_1_nue));
  assert(Kokkos::isfinite(sigma_1_anue));
  assert(Kokkos::isfinite(sigma_1_nux));
  assert(Kokkos::isfinite(sigma_1_anux));
  assert(Kokkos::isfinite(scat_0_nue));
  assert(Kokkos::isfinite(scat_0_anue));
  assert(Kokkos::isfinite(scat_0_nux));
  assert(Kokkos::isfinite(scat_0_anux));
  assert(Kokkos::isfinite(scat_1_nue));
  assert(Kokkos::isfinite(scat_1_anue));
  assert(Kokkos::isfinite(scat_1_nux));
  assert(Kokkos::isfinite(scat_1_anux));

  // convert to code units
  Real const unit_num_dens_dot = unit_num_dens / unit_time;
  Real const unit_ene_dens_dot = unit_ene_dens / unit_time;

  R_nue = R_nue / unit_num_dens_dot;
  R_anue = R_anue / unit_num_dens_dot;
  R_nux = R_nux / unit_num_dens_dot;
  R_anux = R_anux / unit_num_dens_dot;
  Q_nue = Q_nue / unit_ene_dens_dot;
  Q_anue = Q_anue / unit_ene_dens_dot;
  Q_nux = Q_nux / unit_ene_dens_dot;
  Q_anux = Q_anux / unit_ene_dens_dot;
  sigma_0_nue = sigma_0_nue * unit_length;
  sigma_0_anue = sigma_0_anue * unit_length;
  sigma_0_nux = sigma_0_nux * unit_length;
  sigma_0_anux = sigma_0_anux * unit_length;
  sigma_1_nue = sigma_1_nue * unit_length;
  sigma_1_anue = sigma_1_anue * unit_length;
  sigma_1_nux = sigma_1_nux * unit_length;
  sigma_1_anux = sigma_1_anux * unit_length;
  scat_0_nue = scat_0_nue * unit_length;
  scat_0_anue = scat_0_anue * unit_length;
  scat_0_nux = scat_0_nux * unit_length;
  scat_0_anux = scat_0_anux * unit_length;
  scat_1_nue = scat_1_nue * unit_length;
  scat_1_anue = scat_1_anue * unit_length;
  scat_1_nux = scat_1_nux * unit_length;
  scat_1_anux = scat_1_anux * unit_length;
}

//! \fn void NeutrinoDens(Real mu_n, Real mu_p, Real mu_e, Real nb, Real temp,
//!                       Real &n_nue, Real &n_anue, Real &n_nux, Real &en_nue,
//!                       Real &en_anue, Real &en_nux, NuratesParams nurates_params)
//
//   \brief Computes the neutrino number and energy density
//
//   \param[in]  mu_n            neutron chemical potential (MeV)
//   \param[in]  mu_p            proton chemical potential (MeV)
//   \param[in]  mu_e            electron chemical potential (Mev)
//   \param[in]  temp            temperature (MeV)
//   \param[out] n_nue           number density electron neutrinos (code units)
//   \param[out] n_anue          number density electron anti-neutrinos (code units)
//   \param[out] n_nux           number density mu/tau neutrinos (code units)
//   \param[out] en_nue          energy density electron neutrinos (code units)
//   \param[out] en_anue         energy density electron anti-neutrinos (code units)
//   \param[out] en_nux          energy density mu/tau neutrinos (code units)
//   \param[in]  nurates_params  struct for nurates parameters
//   \param[in]  code_units      code units
//   \param[in]  eos_units       eos units
KOKKOS_INLINE_FUNCTION
void NeutrinoDens(Real mu_n, Real mu_p, Real mu_e, Real temp, Real &n_nue, Real &n_anue,
                  Real &n_nux, Real &en_nue, Real &en_anue, Real &en_nux,
                  NuratesParams const &nurates_params,
                  Primitive::UnitSystem const &code_units,
                  Primitive::UnitSystem const &eos_units) {
  Real eta_nue = (mu_p + mu_e - mu_n) / temp;
  Real eta_anue = -eta_nue;
  Real eta_nux = 0.0;

  const Real hc_mevcm3 = HC_MEVCM * HC_MEVCM * HC_MEVCM;
  const Real temp3 = temp * temp * temp;
  const Real temp4 = temp3 * temp;

  n_nue = 4.0 * M_PI / hc_mevcm3 * temp3 * Fermi::fermi2(eta_nue);    // [cm^-3]
  n_anue = 4.0 * M_PI / hc_mevcm3 * temp3 * Fermi::fermi2(eta_anue);  // [cm^-3]
  n_nux = 16.0 * M_PI / hc_mevcm3 * temp3 * Fermi::fermi2(eta_nux);   // [cm^-3]

  en_nue = 4.0 * M_PI / hc_mevcm3 * temp4 * Fermi::fermi3(eta_nue);    // [MeV cm^-3]
  en_anue = 4.0 * M_PI / hc_mevcm3 * temp4 * Fermi::fermi3(eta_anue);  // [MeV cm^-3]
  en_nux = 16.0 * M_PI / hc_mevcm3 * temp4 * Fermi::fermi3(eta_nux);   // [MeV cm^-3]

  assert(Kokkos::isfinite(n_nue));
  assert(Kokkos::isfinite(n_anue));
  assert(Kokkos::isfinite(n_nux));
  assert(Kokkos::isfinite(en_nue));
  assert(Kokkos::isfinite(en_anue));
  assert(Kokkos::isfinite(en_nux));

  // convert to code units
  Primitive::UnitSystem cgs_units = Primitive::MakeCGS();
  // Note that the number densities are always in EOS units
  Real const unit_num_dens = cgs_units.NumberDensityConversion(eos_units);
  Real const unit_ene_dens =
      cgs_units.EnergyDensityConversion(code_units) * cgs_units.MeV; //@TODO: check this!

  n_nue = n_nue * unit_num_dens;
  n_anue = n_anue * unit_num_dens;
  n_nux = n_nux * unit_num_dens;

  en_nue = en_nue * unit_ene_dens;
  en_anue = en_anue * unit_ene_dens;
  en_nux = en_nux * unit_ene_dens;
}

}  // namespace radiationm1
#endif  // ENABLE_NURATES
#endif  // RADIATION_M1_NURATES_HPP
