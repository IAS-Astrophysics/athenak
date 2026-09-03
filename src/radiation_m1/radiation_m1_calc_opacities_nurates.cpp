//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_opacity.cpp
//! \brief calculate opacities for grey M1

#include <cstdio>
#include <iostream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_nurates.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::CalcOpacityNurates(Driver *pdrive, int stage) {
  // The opacities are constant throughout a timestep
  if (stage > 1) {
    return TaskStatus::complete;
  }

  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(
          pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    return CalcOpacityNurates_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
      pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    return CalcOpacityNurates_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

template <class EOSPolicy, class ErrorPolicy>
TaskStatus RadiationM1::CalcOpacityNurates_(Driver *pdrive, int stage) {
  assert(((nspecies == 3) || (nspecies == 4)));

  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &nspecies_ = nspecies;
  auto nvars_ = nvars;

  auto &adm = pmy_pack->padm->adm;
  auto &radiation_mask_ = radiation_mask;

  auto &m1_params_ = params;
  // Force the equilibrium distribution for the first eq_warmup_cycles cycles.
  // On a fresh (neutrinoless) start the M1 moments are floored, so
  // reconstructing the distribution from them (use_equilibrium_distribution =
  // false) yields garbage.
  NuratesParams nurates_params_ = nurates_params;
  if (pmy_pack->pmesh->ncycle < nurates_params_.eq_warmup_cycles) {
    nurates_params_.use_equilibrium_distribution = true;
  }

  auto &eta_0_ = eta_0;
  auto &abs_0_ = abs_0;
  auto &eta_1_ = eta_1;
  auto &abs_1_ = abs_1;
  auto &scat_1_ = scat_1;

  const bool peq_on_ = nurates_params_.use_partial_equilibrium;
  // Half-width of the tier-1 c_v secant, relative to T; a gate, not a Jacobian,
  // so a crude c_v is enough.
  const Real peq_cv_eps = 1.0e-2;
  // How far outside the tier-1 linear bound a root may land and still be believed.
  const Real peq_trust_c = 10.0;
  // Halvings of the weights allowed before the cell is declared unusable.
  const int peq_max_halvings = 4;

  auto &u0_ = u0;
  auto &chi_ = chi;

  DvceArray5D<Real> w0_ = w0;
  if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
  }

  Real dt_ = pmy_pack->pmesh->dt;

  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.GetBaryonMass();

  // conversion factors from cgs to code units
  auto code_units = eos.GetCodeUnitSystem();
  auto eos_units = eos.GetEOSUnitSystem();
  auto nurates_units = Primitive::MakeNGS();

  par_for(
      "radiation_m1_calc_opacity_nurates", DevExeSpace(), 0, nmb1, ks, ke, js,
      je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (radiation_mask_(m, k, j, i)) {
          for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
            abs_0_(m, nuidx, k, j, i) = 0;
            eta_0_(m, nuidx, k, j, i) = 0;

            abs_1_(m, nuidx, k, j, i) = 0;
            eta_1_(m, nuidx, k, j, i) = 0;
            scat_1_(m, nuidx, k, j, i) = 0;
          }
        } else {
          Real garr_dd[16];
          Real garr_uu[16];
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
          pack_n_d(adm.alpha(m, k, j, i), n_d);
          adm::SpacetimeMetric(
              adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
              adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
              adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
              adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
              adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_dd);
          adm::SpacetimeUpperMetric(
              adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
              adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
              adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
              adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
              adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_uu);
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              g_dd(a, b) = garr_dd[a + b * 4];
              g_uu(a, b) = garr_uu[a + b * 4];
            }
          }

          Real gam = adm::SpatialDet(
              adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
              adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
              adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
          Real volform = Kokkos::sqrt(gam);

          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};

          Real w_lorentz{};
          w_lorentz =
              get_w_lorentz(w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i),
                            w0_(m, IVZ, k, j, i), g_dd);
          pack_u_u(
              w_lorentz / adm.alpha(m, k, j, i),
              w0_(m, IVX, k, j, i) - w_lorentz * adm.beta_u(m, 0, k, j, i) /
                                         adm.alpha(m, k, j, i),
              w0_(m, IVY, k, j, i) - w_lorentz * adm.beta_u(m, 1, k, j, i) /
                                         adm.alpha(m, k, j, i),
              w0_(m, IVZ, k, j, i) - w_lorentz * adm.beta_u(m, 2, k, j, i) /
                                         adm.alpha(m, k, j, i),
              u_u);

          pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
                   adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i), v_u);
          tensor_contract(g_dd, u_u, u_d);
          tensor_contract(g_dd, v_u, v_d);
          calc_proj(u_d, u_u, proj_ud);

          // Compute lab frame energy density and number density
          Real J[4]{}, rnnu[4]{};
          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
            pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                     adm.beta_u(m, 2, k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i),
                     F_d);
            const Real E =
                u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
            apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                          chi_(m, nuidx, k, j, i), P_dd, m1_params_);

            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
            assemble_rT(n_d, E, F_d, P_dd, T_dd);

            J[nuidx] = calc_J_from_rT(T_dd, u_u);
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
            calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
            apply_floor(g_uu, J[nuidx], H_d, m1_params_);
            Real Gamma =
                compute_Gamma(w_lorentz, v_u, J[nuidx], E, F_d, m1_params_);
            rnnu[nuidx] =
                u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) / Gamma;
          }

          // local undensitized neutrino quantities
          Real nudens_0[4]{}, nudens_1[4]{}, chi_loc[4]{};
          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            nudens_0[nuidx] = rnnu[nuidx] / volform;
            nudens_1[nuidx] = J[nuidx] / volform;
            chi_loc[nuidx] = chi_(m, nuidx, k, j, i);
          }

          // fluid quantities
          Real nb = w0_(m, IDN, k, j, i) / mb;
          Real p = w0_(m, IPR, k, j, i);
          Real Y = w0_(m, IYF, k, j, i);
          Real T = eos.GetTemperatureFromP(nb, p, &Y);
          Real yp = eos.GetProtonFraction(nb, T, &Y);
          Real yn = eos.GetNeutronFraction(nb, T, &Y);
          Real mu_b = eos.GetBaryonChemicalPotential(nb, T, &Y);
          Real mu_q = eos.GetChargeChemicalPotential(nb, T, &Y);
          Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, T, &Y);

          Real mu_n = mu_b;
          Real mu_p = mu_b + mu_q;
          Real mu_e = mu_le - mu_q;

          // get emissivities and opacities
          Real eta_0_loc[4]{}, eta_1_loc[4]{};
          Real abs_0_loc[4]{}, abs_1_loc[4]{};
          Real scat_0_loc[4]{}, scat_1_loc[4]{};
          // non-thermal (inelastic scattering / NEPS) emissivity and absorption,
          // both ENERGY (..._1_...) and NUMBER (..._0_...) channels; non-zero only
          // when use_nonthermal_separated is set
          // NUMBER has no emissivity counterpart: NEPS is subtracted out of abs_0
          // and never re-enters, unlike ENERGY, where eta_1_non_th is added back.
          Real eta_1_non_th_loc[4]{}, abs_1_non_th_loc[4]{};
          Real abs_0_non_th_loc[4]{};

          // Note: everything sent and received are in code units
          ComputeNuratesOpacities(nb, T, yp, yn, mu_n, mu_p, mu_e, nudens_0,
                                  nudens_1, chi_loc, eta_0_loc, eta_1_loc,
                                  abs_0_loc, abs_1_loc, scat_0_loc, scat_1_loc,
                                  eta_1_non_th_loc, abs_1_non_th_loc,
                                  abs_0_non_th_loc, nurates_params_, code_units,
                                  eos_units, nurates_units);

          assert(Kokkos::isfinite(eta_0_loc[0]));
          assert(Kokkos::isfinite(eta_0_loc[1]));
          assert(Kokkos::isfinite(eta_0_loc[2]));
          assert(Kokkos::isfinite(eta_0_loc[3]));

          assert(Kokkos::isfinite(eta_1_loc[0]));
          assert(Kokkos::isfinite(eta_1_loc[1]));
          assert(Kokkos::isfinite(eta_1_loc[2]));
          assert(Kokkos::isfinite(eta_1_loc[3]));

          assert(Kokkos::isfinite(abs_0_loc[0]));
          assert(Kokkos::isfinite(abs_0_loc[1]));
          assert(Kokkos::isfinite(abs_0_loc[2]));
          assert(Kokkos::isfinite(abs_0_loc[3]));

          assert(Kokkos::isfinite(abs_1_loc[0]));
          assert(Kokkos::isfinite(abs_1_loc[1]));
          assert(Kokkos::isfinite(abs_1_loc[2]));
          assert(Kokkos::isfinite(abs_1_loc[3]));

          assert(Kokkos::isfinite(scat_0_loc[0]));
          assert(Kokkos::isfinite(scat_0_loc[1]));
          assert(Kokkos::isfinite(scat_0_loc[2]));
          assert(Kokkos::isfinite(scat_0_loc[3]));

          assert(Kokkos::isfinite(scat_1_loc[0]));
          assert(Kokkos::isfinite(scat_1_loc[1]));
          assert(Kokkos::isfinite(scat_1_loc[2]));
          assert(Kokkos::isfinite(scat_1_loc[3]));

          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            eta_0_loc[nuidx] = (eta_0_loc[nuidx] > 0) ? eta_0_loc[nuidx] : 0;
            eta_1_loc[nuidx] = (eta_1_loc[nuidx] > 0) ? eta_1_loc[nuidx] : 0;
            abs_0_loc[nuidx] = (abs_0_loc[nuidx] > 0) ? abs_0_loc[nuidx] : 0;
            abs_1_loc[nuidx] = (abs_1_loc[nuidx] > 0) ? abs_1_loc[nuidx] : 0;
            eta_1_non_th_loc[nuidx] =
                (eta_1_non_th_loc[nuidx] > 0) ? eta_1_non_th_loc[nuidx] : 0;
            abs_1_non_th_loc[nuidx] =
                (abs_1_non_th_loc[nuidx] > 0) ? abs_1_non_th_loc[nuidx] : 0;
            abs_0_non_th_loc[nuidx] =
                (abs_0_non_th_loc[nuidx] > 0) ? abs_0_non_th_loc[nuidx] : 0;
          }

          Real nudens_0_thin[4]{}, nudens_1_thin[4]{},
              nudens_0_peq[4]{}, nudens_1_peq[4]{};
          // Thermal absorption after the non-LTE correction (Kirchhoff applies to
          // it alone) and the correction factor itself, needed below by the
          // no-Kirchhoff path. Both default to a no-op if the block is skipped.
          Real abs_0_th[4]{}, abs_1_th[4]{};
          Real corr_ae[4] = {1.0, 1.0, 1.0, 1.0};

          if (nurates_params_.use_kirchhoff_law ||
              nurates_params_.use_equilibrium_distribution) {
            // compute neutrino black body function assuming fixed temperature
            // and Ye
            NeutrinoDens(mu_n, mu_p, mu_e, T, nudens_0_thin[0],
                         nudens_0_thin[1], nudens_0_thin[2], nudens_1_thin[0],
                         nudens_1_thin[1], nudens_1_thin[2], nurates_params_,
                         code_units, eos_units, nurates_units);

            nudens_0_thin[2] *= 0.5;
            nudens_1_thin[2] *= 0.5;
            nudens_0_thin[3] = nudens_0_thin[2];
            nudens_1_thin[3] = nudens_1_thin[2];

            // ----------------------------------------------------------------
            // Non-LTE correction (kappa ~ E_nu^2), applied to the OPACITIES
            // only: the emissivities are set at the end of the kernel, where
            // Kirchhoff's law has an equilibrium distribution to multiply by.
            //
            // corr_fac comes from the LOCAL blackbody at (T^n, Ye^n), not from
            // the equilibrium the predictor settles on -- the predictor's
            // weights are built from kappa_abs, so the other choice would make
            // kappa depend on the weights that depend on kappa.
            //
            // Absorption is charged-current: corr_ae is 1 for the heavy-lepton
            // neutrinos (matches THC; otherwise their luminosity inflates ~3x)
            // and scales the THERMAL part only. Scattering is corrected on ALL
            // flavours, with no thermal split. The result is folded into the
            // *_loc arrays, so those are the opacities the rest of the step
            // sees; abs_*_th keep the thermal parts Kirchhoff multiplies.
            // ----------------------------------------------------------------
            for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
              Real corr_fac = 1.0;
              if (nurates_params_.use_equilibrium_distribution) {
                corr_fac = (J[nuidx] / rnnu[nuidx]) *
                           (nudens_0_thin[nuidx] / nudens_1_thin[nuidx]);
                if (!Kokkos::isfinite(corr_fac)) {
                  corr_fac = 1.0;
                }
                corr_fac *= corr_fac;
                corr_fac = Kokkos::fmax(
                    1.0 / nurates_params_.opacity_corr_fac_max,
                    Kokkos::fmin(corr_fac,
                                 nurates_params_.opacity_corr_fac_max));
              }
              corr_ae[nuidx] = (nuidx == 0 || nuidx == 1) ? corr_fac : 1.0;

              scat_1_loc[nuidx] *= corr_fac;

              // Asymmetry below is inherited, not chosen here: abs_1 gets its
              // non-thermal part back, abs_0 does not. New is that the
              // no-Kirchhoff path now takes the same split, so it also drops
              // abs_0_non_th, on all species. Both are no-ops without NEPS.
              abs_0_th[nuidx] = Kokkos::fmax(
                  abs_0_loc[nuidx] - abs_0_non_th_loc[nuidx], 0.0)*corr_ae[nuidx];
              abs_1_th[nuidx] = Kokkos::fmax(
                  abs_1_loc[nuidx] - abs_1_non_th_loc[nuidx], 0.0)*corr_ae[nuidx];
              abs_0_loc[nuidx] = abs_0_th[nuidx];
              abs_1_loc[nuidx] = abs_1_th[nuidx] + abs_1_non_th_loc[nuidx];
            }

            // ----------------------------------------------------------------
            // Partially-equilibrated (T*, Ye*) predictor: a one-parameter family
            // in w = a/(1+a), a = dtau*kappa. At w = 1 the residuals are
            // GetBetaEquilibriumTrapped's exactly, at w = 0 they return
            // (T, Y_e), the local blackbody just computed. It spans, per cell
            // and continuously in dt, the two states the optical-depth test
            // this replaced jumped between.
            // ----------------------------------------------------------------
            if (peq_on_) {
              // A FULL step, not the stage's beta*dt: this function returns
              // early for stage > 1, so this emissivity serves the whole cycle.
              const Real dtau = dt_ * adm.alpha(m, k, j, i) / w_lorentz;

              // ABSORPTION opacities only. The weights measure thermalisation,
              // and elastic scattering neither thermalises the energy
              // (u_a H^a = 0) nor changes the number density, so scat_1 has no
              // place in them. abs_0_loc is thermal-only and abs_1_loc thermal
              // plus non-thermal (above); both are stored verbatim below, so
              // each weight tracks the kappa its own channel integrates.
              //
              // The electron pair takes one weight per species, because lumping
              // a pair under a common weight is exact only where the two terms
              // that weight multiplies are equal -- and they differ by
              // exp(eta), with the lepton residual their difference, so the two
              // errors reinforce.
              //
              // The heavy pairs share one weight because nothing in the
              // opacities distinguishes nu_x from its antiparticle: the two
              // differ only numerically. That weight comes from a J-weighted
              // kappa, so kappa_bar*J equals the sum of the per-species
              // kappa_x*J_x at t^n, falling back to the arithmetic mean when
              // the field is empty and there is nothing to weight with. Once
              // the physics does distinguish them, they need splitting the same
              // way: eta = 0 equalises their equilibrium densities but not the
              // actual ones, so no single weight is exact for both sides.
              Real J_x = nudens_1[2];
              Real kJ_x = abs_1_loc[2] * nudens_1[2];
              Real ks_x = abs_1_loc[2];
              int n_x = 1;
              if (nspecies_ > 3) {
                J_x += nudens_1[3];
                kJ_x += abs_1_loc[3] * nudens_1[3];
                ks_x += abs_1_loc[3];
                n_x = 2;
              }

              const Real kbar_1x = (J_x > 0.0) ? kJ_x/J_x : ks_x/n_x;

              const Real a_1p = dtau*abs_1_loc[0];
              const Real a_1m = dtau*abs_1_loc[1];
              const Real a_1x = dtau*kbar_1x;
              const Real a_0p = dtau*abs_0_loc[0];
              const Real a_0m = dtau*abs_0_loc[1];
              const Real w_1p = a_1p/(1.0 + a_1p);
              const Real w_1m = a_1m/(1.0 + a_1m);
              const Real w_1x = a_1x/(1.0 + a_1x);
              const Real w_0p = a_0p/(1.0 + a_0p);
              const Real w_0m = a_0m/(1.0 + a_0m);

              Real T_star = T;
              Real Ye_star = Y;

              // Tier-0 gate: no EOS calls at all. An optically thin cell has
              // nothing to equilibrate with and must cost nothing. Ternaries not
              // fmax, per eos_compose.hpp:188 (SYCL's fmax(x, NaN) = NaN), so a
              // NaN weight gates the cell out deliberately, not by luck.
              const bool w_finite = Kokkos::isfinite(w_1p) &&
                                    Kokkos::isfinite(w_1m) &&
                                    Kokkos::isfinite(w_1x) &&
                                    Kokkos::isfinite(w_0p) &&
                                    Kokkos::isfinite(w_0m);
              Real w_max = (w_1p > w_1m) ? w_1p : w_1m;
              w_max = (w_max > w_1x) ? w_max : w_1x;
              w_max = (w_max > w_0p) ? w_max : w_0p;
              w_max = (w_max > w_0m) ? w_max : w_0m;

              if (w_finite && w_max >= nurates_params_.peq_w_floor) {
                Real Y_part[3] = {Y, 0.0, 0.0};

                // Tier-1 gate: first-order bounds on the excursion the solve
                // would produce, from the blackbody already in hand and c_v. A
                // cell already at equilibrium must also cost nothing. T is
                // clamped to the table because weight_idx_lt clamps the index
                // but not the interpolation weight, so an out-of-range T
                // extrapolates; dividing by the actual T_hi - T_lo keeps a
                // one-sided secant at the edge correct.
                const Real T_tab_min = eos.GetMinimumTemperature()*
                                       eos_units.TemperatureConversion(code_units);
                const Real T_tab_max = eos.GetMaximumTemperature()*
                                       eos_units.TemperatureConversion(code_units);
                Real T_lo = T*(1.0 - peq_cv_eps);
                Real T_hi = T*(1.0 + peq_cv_eps);
                T_lo = (T_lo > T_tab_min) ? T_lo : T_tab_min;
                T_hi = (T_hi < T_tab_max) ? T_hi : T_tab_max;
                const Real cv = (T_hi > T_lo)
                                    ? (eos.GetEnergy(nb, T_hi, Y_part) -
                                       eos.GetEnergy(nb, T_lo, Y_part))/(T_hi - T_lo)
                                    : 0.0;
                const bool cv_ok = Kokkos::isfinite(cv) && cv > 0.0;

                Real J_x_eq = nudens_1_thin[2];
                if (nspecies_ > 3) {
                  J_x_eq += nudens_1_thin[3];
                }

                // Sum of absolute per-species terms, not the absolute value of
                // the weighted net: the gate's only job is to skip cells where
                // nothing happens, and only the sum is guaranteed not to
                // under-estimate the move, so only it cannot gate out a cell
                // that would have moved.
                const Real dlnT_hat =
                    cv_ok ? (w_1p*Kokkos::fabs(nudens_1_thin[0] - nudens_1[0]) +
                             w_1m*Kokkos::fabs(nudens_1_thin[1] - nudens_1[1]) +
                             w_1x*Kokkos::fabs(J_x_eq - J_x))/(T*cv)
                          : 0.0;
                const Real dYe_hat =
                    (w_0p*Kokkos::fabs(nudens_0_thin[0] - nudens_0[0]) +
                     w_0m*Kokkos::fabs(nudens_0_thin[1] - nudens_0[1]))/nb;

                // A bad c_v removes the gate and the trust region both --
                // everything below divides by T*cv. Predict nothing instead.
                if (cv_ok && !(dlnT_hat < nurates_params_.peq_dlnT_tol &&
                               dYe_hat < nurates_params_.peq_dYe_tol)) {
                  // Trust region. The gate estimates do double duty: a root far
                  // outside the linear bound is a converged-but-wrong root --
                  // the failure mode that matters, since a small energy residual
                  // admits a badly wrong T wherever the thermal energy is a small
                  // fraction of the total. The floor at the gate tolerances lets
                  // an already-equilibrated cell still move imperceptibly, and is
                  // where the ternaries' NaN branch lands: reject, not admit.
                  const Real dlnT_trust = peq_trust_c*dlnT_hat;
                  const Real dYe_trust = peq_trust_c*dYe_hat;
                  const Real dlnT_max =
                      (dlnT_trust > nurates_params_.peq_dlnT_tol)
                          ? dlnT_trust : nurates_params_.peq_dlnT_tol;
                  const Real dYe_max =
                      (dYe_trust > nurates_params_.peq_dYe_tol)
                          ? dYe_trust : nurates_params_.peq_dYe_tol;

                  const Real e_mat = eos.GetEnergy(nb, T, Y_part);

                  // On failure, halve all five weights and retry: that slides
                  // the problem along the same one-parameter family toward the
                  // trivial one, so every intermediate point is a valid scheme.
                  // Scaling all five is the same as scaling the pair mean and
                  // the half-difference, so this attacks the split terms too.
                  // A cell that never produces an accepted root keeps (T, Y_e).
                  Real f_soft = 1.0;
                  for (int n_soft = 0; n_soft <= peq_max_halvings;
                       ++n_soft, f_soft *= 0.5) {
                    const Real u[PEQ_NWEIGHTS] = {
                        f_soft*w_1p, f_soft*w_1m, f_soft*w_1x,
                        f_soft*w_0p, f_soft*w_0m};

                    const Real e_rhs = e_mat + u[PEQ_W1_NUE]*nudens_1[0] +
                                       u[PEQ_W1_ANUE]*nudens_1[1] + u[PEQ_W1_X]*J_x;
                    Real Yl_rhs[3] = {Y + (u[PEQ_W0_NUE]*nudens_0[0] -
                                           u[PEQ_W0_ANUE]*nudens_0[1])/nb, 0.0, 0.0};

                    Real T_try = T;
                    Real Ye_try[3] = {Y, 0.0, 0.0};
                    bool ok = eos.GetBetaEquilibriumPartial(
                        nb, e_rhs, Yl_rhs, u, T_try, &Ye_try[0], T, Y_part);

                    if (ok && Kokkos::fabs(Kokkos::log(T_try/T)) <= dlnT_max &&
                        Kokkos::fabs(Ye_try[0] - Y) <= dYe_max) {
                      T_star = T_try;
                      Ye_star = Ye_try[0];
                      break;
                    }
                  }
                }
              }

              // The equilibrium the cell is predicted to be radiating towards.
              // Evaluated unconditionally: a gated or unusable cell has
              // (T*, Ye*) = (T, Y_e), so this reproduces nudens_*_thin bit for
              // bit and the w -> 0 limit costs no special case. The solve and
              // this evaluation do not share a function, but they agree on the
              // mathematics to 1-2 ulp: func_eq_weak's closed forms are exactly
              // what FDI_p2/FDI_p3 reflect on.
              Real Ye_arr[3] = {Ye_star, 0.0, 0.0};
              Real mu_b_s = eos.GetBaryonChemicalPotential(nb, T_star, Ye_arr);
              Real mu_q_s = eos.GetChargeChemicalPotential(nb, T_star, Ye_arr);
              Real mu_le_s =
                  eos.GetElectronLeptonChemicalPotential(nb, T_star, Ye_arr);

              NeutrinoDens(mu_b_s, mu_b_s + mu_q_s, mu_le_s - mu_q_s, T_star,
                           nudens_0_peq[0], nudens_0_peq[1], nudens_0_peq[2],
                           nudens_1_peq[0], nudens_1_peq[1], nudens_1_peq[2],
                           nurates_params_, code_units, eos_units,
                           nurates_units);
              nudens_0_peq[2] *= 0.5;
              nudens_1_peq[2] *= 0.5;
              nudens_0_peq[3] = nudens_0_peq[2];
              nudens_1_peq[3] = nudens_1_peq[2];

              // Finiteness screen, the live counterpart of the asserts the
              // trapped branch carried (and -DNDEBUG removes). Nothing
              // downstream catches this: abs_*_th * my_nudens_* would carry a
              // NaN into eta_* and on into the source term. Fall back to the
              // local blackbody, the w -> 0 answer.
              bool peq_finite = true;
              for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
                peq_finite = peq_finite &&
                             Kokkos::isfinite(nudens_0_peq[nuidx]) &&
                             Kokkos::isfinite(nudens_1_peq[nuidx]);
              }
              if (!peq_finite) {
                for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
                  nudens_0_peq[nuidx] = nudens_0_thin[nuidx];
                  nudens_1_peq[nuidx] = nudens_1_thin[nuidx];
                }
                T_star = T;
                Ye_star = Y;
              }
            }
          }

          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            // store opacities and emissivities
            eta_0_(m, nuidx, k, j, i) = eta_0_loc[nuidx];
            eta_1_(m, nuidx, k, j, i) = eta_1_loc[nuidx];
            abs_0_(m, nuidx, k, j, i) = abs_0_loc[nuidx];
            abs_1_(m, nuidx, k, j, i) = abs_1_loc[nuidx];
            scat_1_(m, nuidx, k, j, i) = scat_1_loc[nuidx];

            Real my_nudens_0{}, my_nudens_1{};
            if (nurates_params_.use_kirchhoff_law ||
                nurates_params_.use_equilibrium_distribution) {
              if (peq_on_) {
                // One equilibrium per cell, at the state the matter is predicted
                // to reach over this step: the local blackbody as dt -> 0, the
                // trapped weak equilibrium as dt -> infinity, continuous between.
                my_nudens_0 = nudens_0_peq[nuidx];
                my_nudens_1 = nudens_1_peq[nuidx];
              } else {
                // The predictor's dt -> 0 limit, and all that is left with it off.
                my_nudens_0 = nudens_0_thin[nuidx];
                my_nudens_1 = nudens_1_thin[nuidx];
              }
            }

            // Correction for the NEPS
            // emissivity when using the equilibrium distribution. The NEPS
            // in-scattering emissivity scales with the neutrino occupation
            // (~ g_nu); the equilibrium distribution over-populates the field in
            // the decoupling region, over-estimating it. Scale the non-thermal
            // (NEPS) emissivity by the ratio of the ACTUAL M1 density to the
            // equilibrium density (a grey proxy for g_field/g_eq), using the number
            // ratio to preserve the spectrum and apply the ratio to both
            // the number (nudens_0) and energy (nudens_1) channels. Capped
            // at 1 so it only removes the spurious excess, never amplifies; deep
            // in the optically-thick core the field ~ equilibrium so the factor
            // -> 1. This mimics the reconstructed-distribution result without
            // reconstructing the spectrum. Only the non-thermal (NEPS) part is
            // touched; the spontaneous (beta) emissivity has no g_nu dependence.
            if (nurates_params_.use_equilibrium_distribution) {
              Real f_occ_0 =
                  (my_nudens_0 > 0.0) ? nudens_0[nuidx] / my_nudens_0 : 1.0;
              if (!Kokkos::isfinite(f_occ_0)) f_occ_0 = 1.0;
              f_occ_0 = Kokkos::fmin(f_occ_0, 1.0);
              eta_1_non_th_loc[nuidx] *= f_occ_0;
            }

            // The emissivities, and the only place they are set. Kirchhoff
            // derives them from the corrected THERMAL opacity and the
            // equilibrium distribution, inheriting the non-LTE correction
            // through abs_*_th; NEPS emission is added back afterwards, kept
            // out of thermalization. Without Kirchhoff, bns_nurates' own
            // emissivities stand, scaled like the opacities they pair with.
            if (nurates_params_.use_kirchhoff_law) {
              eta_0_(m, nuidx, k, j, i) =
                  (abs_0_th[nuidx] > 0)
                      ? abs_0_th[nuidx] * my_nudens_0
                      : eta_0_loc[nuidx];
              eta_1_(m, nuidx, k, j, i) =
                  (abs_1_th[nuidx] > 0)
                      ? abs_1_th[nuidx] * my_nudens_1 + eta_1_non_th_loc[nuidx]
                      : eta_1_loc[nuidx];
            } else {
              eta_0_(m, nuidx, k, j, i) = eta_0_loc[nuidx] * corr_ae[nuidx];
              eta_1_(m, nuidx, k, j, i) = eta_1_loc[nuidx] * corr_ae[nuidx];
            }
          }
        }
      });
  return TaskStatus::complete;
}
}  // namespace radiationm1
