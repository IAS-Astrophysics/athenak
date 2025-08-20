//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_opacity.cpp
//! \brief calculate opacities for grey M1

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_nurates.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::CalcOpacityNurates(Driver *pdrive, int stage) {
  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    return CalcOpacityNurates_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  auto *ptest_nlog =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                     Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    return CalcOpacityNurates_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl;
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
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nspecies_ = nspecies;
  auto nvars_ = nvars;
  
  auto &adm = pmy_pack->padm->adm;
  auto &radiation_mask_ = radiation_mask;

  auto &m1_params_ = params;
  auto &nurates_params_ = nurates_params;

  auto &eta_0_ = eta_0;
  auto &abs_0_ = abs_0;
  auto &eta_1_ = eta_1;
  auto &abs_1_ = abs_1;
  auto &scat_1_ = scat_1;

  auto &u0_ = u0;
  auto &chi_ = chi;

  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  DvceArray5D<Real> &w0_ = u_mu_data;
  if (ishydro) {
    //w0_ = pmy_pack->phydro->w0; // @TODO
  } else if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
  }
  
  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  if ((m1_params_.opacity_one_dt) && (stage != 1)) {
    // Keep opacities constant throught the timestep
    return TaskStatus::complete;
  }
  
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.GetBaryonMass();

  // conversion factors from cgs to code units
  auto code_units = eos.GetCodeUnitSystem();
  auto eos_units = eos.GetEOSUnitSystem();
  
  par_for(
      "radiation_m1_calc_opacity_nurates", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (radiation_mask(m, k, j, i)) {
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
          adm::SpacetimeMetric(adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
                               adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                               adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                               adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                               adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
                               garr_dd);
          adm::SpacetimeUpperMetric(
              adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
              adm.beta_u(m, 2, k, j, i), adm.g_dd(m, 0, 0, k, j, i),
              adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
              adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i),
              adm.g_dd(m, 2, 2, k, j, i), garr_uu);
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              g_dd(a, b) = garr_dd[a + b * 4];
              g_uu(a, b) = garr_uu[a + b * 4];
            }
          }

          Real gam =
              adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                              adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                              adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
          Real volform = Kokkos::sqrt(gam);

          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};

          Real w_lorentz{};
	  if (use_u_mu_data) {
            w_lorentz = adm.alpha(m,k,j,i) * u_mu_(m, 0, k, j, i);
            pack_u_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i), u_mu_(m, 2, k, j, i),
                     u_mu_(m, 3, k, j, i), u_u);
          } else {
            // w_lorentz = Kokkos::sqrt(1. + w0_(m, IVX, k, j, i) * w0_(m, IVX, k, j, i) +
            //                          w0_(m, IVY, k, j, i) * w0_(m, IVY, k, j, i) +
            //                          w0_(m, IVZ, k, j, i) * w0_(m, IVZ, k, j, i)); 
	    w_lorentz = get_w_lorentz(w0_(m, IVX, k, j, i),
				      w0_(m, IVY, k, j, i),
				      w0_(m, IVZ, k, j, i),
				      g_dd);
	    pack_u_u(w_lorentz / adm.alpha(m, k, j, i),
                     w0_(m, IVX, k, j, i)
		     - w_lorentz * adm.beta_u(m, 0, k, j, i) / adm.alpha(m, k, j, i),
                     w0_(m, IVY, k, j, i)
		     - w_lorentz * adm.beta_u(m, 1, k, j, i) / adm.alpha(m, k, j, i),
                     w0_(m, IVZ, k, j, i)
		     - w_lorentz * adm.beta_u(m, 2, k, j, i) / adm.alpha(m, k, j, i),
                     u_u);
	  }
	  
          pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
                   adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i),
		   v_u);
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
            const Real E = u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
            apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                          chi_(m, nuidx, k, j, i), P_dd, m1_params_);

            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
            assemble_rT(n_d, E, F_d, P_dd, T_dd);

            J[nuidx] = calc_J_from_rT(T_dd, u_u);
            Real Gamma = compute_Gamma(w_lorentz, v_u, J[nuidx], E, F_d, m1_params_);
            rnnu[nuidx] = u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) / Gamma;
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
	  
          // Note: everything sent and received are in code units
          bns_nurates(
              nb, T, Y, mu_n, mu_p, mu_e, nudens_0[0], nudens_1[0], chi_loc[0],
              nudens_0[1], nudens_1[1], chi_loc[1], nudens_0[2], nudens_1[2], chi_loc[2],
              nudens_0[3], nudens_1[3], chi_loc[3], eta_0_loc[0], eta_0_loc[1],
              eta_0_loc[2], eta_0_loc[3], eta_1_loc[0], eta_1_loc[1], eta_1_loc[2],
              eta_1_loc[3], abs_0_loc[0], abs_0_loc[1], abs_0_loc[2], abs_0_loc[3],
              abs_1_loc[0], abs_1_loc[1], abs_1_loc[2], abs_1_loc[3], scat_0_loc[0],
              scat_0_loc[1], scat_0_loc[2], scat_0_loc[3], scat_1_loc[0], scat_1_loc[1],
              scat_1_loc[2], scat_1_loc[3], nurates_params_, code_units, eos_units);

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

          Real tau{}, nudens_0_trap[4]{}, nudens_1_trap[4]{}, nudens_0_thin[4]{},
              nudens_1_thin[4]{};

          if (nurates_params_.use_kirchhoff_law ||
              nurates_params_.use_equilibrium_distribution) {
            // effective optical depth to decide whether to compute black body function
            // for neutrinos assuming neutrino tapping or at fixed temperature and Ye
            Real tau =
                Kokkos::min(Kokkos::sqrt(abs_1_loc[0] * (abs_1_loc[0] + scat_1_loc[0])),
                            Kokkos::sqrt(abs_1_loc[1] * (abs_1_loc[1] + scat_1_loc[1]))) *
                beta_dt;

            // compute neutrino black body function assuming trapped neutrinos
            if (nurates_params_.opacity_tau_trap >= 0 &&
                tau > nurates_params_.opacity_tau_trap) {
              // fluid frame number densities for nue, anue, numu, anumu, nutau, anutau
              // factor of 1/2 appears because we evolve mu and tau species together
              Real n_nu[6] = {rnnu[0],      rnnu[1],      rnnu[2] / 2.,
                              rnnu[3] / 2., rnnu[2] / 2., rnnu[3] / 2.};
              Real Y_part[3] = {Y, 0., 0.};  // only ye is used

              Real Y_lep[3]{};
              eos.GetLeptonFractions(nb, Y_part, n_nu, Y_lep);

              Real e = eos.GetEnergy(nb, T, Y_part) + J[0] + J[1] + J[2] + J[3];

              Real temperature_trap{}, Y_e_trap[3]{};
              bool res = eos.GetBetaEquilibriumTrapped(nb, e, Y_lep, temperature_trap,
                                                       &Y_e_trap[0], T, Y_part);

              if (!res) {
                // trying to recompute weak equilibrium neglecting current neutrino data
                Real n_nu_zero[6]{};
                Real Y_lep_zero[3]{};
                eos.GetLeptonFractions(nb, Y_part, n_nu_zero, Y_lep_zero);
                Real e_zero = eos.GetEnergy(nb, T, Y_part);
                bool res = eos.GetBetaEquilibriumTrapped(
                    nb, e_zero, Y_lep_zero, temperature_trap, &Y_e_trap[0], T, Y_part);
              }

              Real mu_b_eq =
                  eos.GetBaryonChemicalPotential(nb, temperature_trap, &Y_e_trap[0]);
              Real mu_q_eq =
                  eos.GetChargeChemicalPotential(nb, temperature_trap, &Y_e_trap[0]);
              Real mu_le_eq = eos.GetElectronLeptonChemicalPotential(nb, T, &Y_e_trap[0]);

              Real mu_n_eq = mu_b_eq;
              Real mu_p_eq = mu_b_eq + mu_q_eq;
              Real mu_e_eq = mu_le_eq - mu_q_eq;

              NeutrinoDens(mu_n_eq, mu_p_eq, mu_e_eq, temperature_trap, nudens_0_trap[0],
                           nudens_0_trap[1], nudens_0_trap[2], nudens_1_trap[0],
                           nudens_1_trap[1], nudens_1_trap[2], nurates_params_,
                           code_units, eos_units);

              assert(Kokkos::isfinite(nudens_0_trap[0]));
              assert(Kokkos::isfinite(nudens_0_trap[1]));
              assert(Kokkos::isfinite(nudens_0_trap[2]));
              assert(Kokkos::isfinite(nudens_1_trap[0]));
              assert(Kokkos::isfinite(nudens_1_trap[1]));
              assert(Kokkos::isfinite(nudens_1_trap[2]));

              nudens_0_trap[2] *= 0.5;
              nudens_1_trap[2] *= 0.5;
              nudens_0_trap[3] = nudens_0_trap[2];
              nudens_1_trap[3] = nudens_1_trap[2];
            }

            // compute neutrino black body function assuming fixed temperature and Ye
            NeutrinoDens(mu_n, mu_p, mu_e, T, nudens_0_thin[0], nudens_0_thin[1],
                         nudens_0_thin[2], nudens_1_thin[0], nudens_1_thin[1],
                         nudens_1_thin[2], nurates_params_, code_units, eos_units);

            nudens_0_thin[2] *= 0.5;
            nudens_1_thin[2] *= 0.5;
            nudens_0_thin[3] = nudens_0_thin[2];
            nudens_1_thin[3] = nudens_1_thin[2];
          }

          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            // store opacities and emissivities
            eta_0(m, nuidx, k, j, i) = eta_0_loc[nuidx];
            eta_1(m, nuidx, k, j, i) = eta_1_loc[nuidx];
            abs_0(m, nuidx, k, j, i) = abs_0_loc[nuidx];
            abs_1(m, nuidx, k, j, i) = abs_1_loc[nuidx];
            scat_1(m, nuidx, k, j, i) = scat_1_loc[nuidx];

            Real my_nudens_0{}, my_nudens_1{}, corr_fac{1};
            if (nurates_params_.use_kirchhoff_law ||
                nurates_params_.use_equilibrium_distribution) {
              // combine optically thin and optically thick limits
              if (nurates_params_.opacity_tau_trap < 0 ||
                  tau <= nurates_params_.opacity_tau_trap) {
                my_nudens_0 = nudens_0_thin[nuidx];
                my_nudens_1 = nudens_1_thin[nuidx];
              } else if (tau > nurates_params_.opacity_tau_trap +
                                   nurates_params_.opacity_tau_delta) {
                my_nudens_0 = nudens_0_trap[nuidx];
                my_nudens_1 = nudens_1_trap[nuidx];
              } else {
                Real const lam = (tau - nurates_params_.opacity_tau_trap) /
                                 nurates_params_.opacity_tau_delta;
                my_nudens_0 =
                    lam * nudens_0_trap[nuidx] + (1 - lam) * nudens_0_thin[nuidx];
                my_nudens_1 =
                    lam * nudens_1_trap[nuidx] + (1 - lam) * nudens_1_thin[nuidx];
              }

              // Correction factor for absorption opacities for non-LTE effects
              // (kappa ~ E_nu^2)
              corr_fac = 1.0;
              corr_fac = (J[nuidx] / rnnu[nuidx]) * (my_nudens_0 / my_nudens_1);
              if (!Kokkos::isfinite(corr_fac)) {
                corr_fac = 1.0;
              }
              corr_fac *= corr_fac;
              corr_fac = Kokkos::max(
                  1.0 / nurates_params_.opacity_corr_fac_max,
                  Kokkos::min(corr_fac, nurates_params_.opacity_corr_fac_max));
            }

            if (nurates_params_.use_equilibrium_distribution) {
              eta_0(m, nuidx, k, j, i) *= corr_fac;
              eta_1(m, nuidx, k, j, i) *= corr_fac;
              abs_0(m, nuidx, k, j, i) *= corr_fac;
              abs_1(m, nuidx, k, j, i) *= corr_fac;
              scat_1(m, nuidx, k, j, i) *= corr_fac;
            }

            if (nurates_params_.use_kirchhoff_law) {
	      // enforce Kirchhoff's laws.
	      // eta_0(m, nuidx, k, j, i) = abs_0(m, nuidx, k, j, i) * my_nudens_0;
	      // eta_1(m, nuidx, k, j, i) = abs_1(m, nuidx, k, j, i) * my_nudens_1;
	      if (nuidx == 0 || nuidx == 1) {
		eta_0(m, nuidx, k, j, i) = abs_0(m, nuidx, k, j, i) * my_nudens_0;
		eta_1(m, nuidx, k, j, i) = abs_1(m, nuidx, k, j, i) * my_nudens_1;
	      } else {
		abs_0(m, nuidx, k, j, i) = (my_nudens_0 > m1_params_.rad_N_floor ?
					    eta_0(m, nuidx, k, j, i)/my_nudens_0 : 0);
		abs_1(m, nuidx, k, j, i) = (my_nudens_0 > m1_params_.rad_E_floor ?
					    eta_1(m, nuidx, k, j, i)/my_nudens_1 : 0);
	      }
            }
          }
        }
      });

  return TaskStatus::complete;
}
}  // namespace radiationm1
