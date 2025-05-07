//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_opacity.cpp
//! \brief calculate opacities for grey M1

#include <coordinates/cell_locations.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_nurates.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::CalcOpacityToy(Driver *pdrive, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int &ng = indcs.ng;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto nvarstotm1 = pmy_pack->pradm1->nvarstot - 1;
  auto nvars_ = pmy_pack->pradm1->nvars;
  auto &params_ = pmy_pack->pradm1->params;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  auto &u0_ = pmy_pack->pradm1->u0;

  auto &eta_0_ = pmy_pack->pradm1->eta_0;
  auto &abs_0_ = pmy_pack->pradm1->abs_0;
  auto &eta_1_ = pmy_pack->pradm1->eta_1;
  auto &abs_1_ = pmy_pack->pradm1->abs_1;
  auto &scat_1_ = pmy_pack->pradm1->scat_1;
  auto &toy_opacity_fn_ = pmy_pack->pradm1->toy_opacity_fn;
  auto &chi_ = pmy_pack->pradm1->chi;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  par_for(
      "radiation_m1_calc_toy_opacity", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0,
      nspecies_ - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int nuidx) {
        Real &x1min = mbsize.d_view(m).x1min;
        Real &x1max = mbsize.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real dx = (x1max - x1min) / static_cast<Real>(nx1);
        Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

        Real &x2min = mbsize.d_view(m).x2min;
        Real &x2max = mbsize.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real dy = (x2max - x2min) / static_cast<Real>(nx2);
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        Real &x3min = mbsize.d_view(m).x3min;
        Real &x3max = mbsize.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real dz = (x3max - x3min) / static_cast<Real>(nx3);
        Real x3 = CellCenterX(k - ks, nx3, x3min, x3max);
        toy_opacity_fn_(x1, x2, x3, dx, dy, dz, nuidx, eta_0_(m, nuidx, k, j, i),
                        abs_0_(m, nuidx, k, j, i), eta_1_(m, nuidx, k, j, i),
                        abs_1_(m, nuidx, k, j, i), scat_1_(m, nuidx, k, j, i));
      });

  return TaskStatus::complete;
}

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
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int &ng = indcs.ng;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto nvarstotm1 = pmy_pack->pradm1->nvarstot - 1;
  auto nvars_ = pmy_pack->pradm1->nvars;
  auto &params_ = pmy_pack->pradm1->params;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &m1_ = pmy_pack->pradm1->u0;
  auto &w0_ = pmy_pack->pmhd->w0;

  auto &eta_0_ = pmy_pack->pradm1->eta_0;
  auto &abs_0_ = pmy_pack->pradm1->abs_0;
  auto &eta_1_ = pmy_pack->pradm1->eta_1;
  auto &abs_1_ = pmy_pack->pradm1->abs_1;
  auto &scat_1_ = pmy_pack->pradm1->scat_1;
  auto &toy_opacity_fn_ = pmy_pack->pradm1->toy_opacity_fn;
  auto &chi_ = pmy_pack->pradm1->chi;
  auto &nurates_params_ = pmy_pack->pradm1->nurates_params;
  auto &radiation_mask_ = pmy_pack->pradm1->radiation_mask;
  auto &adm = pmy_pack->padm->adm;
  auto &dyngr = pmy_pack->pdyngr;
  auto &u0_ = u0;
  auto &u_mu_ = u_mu;

  const Real nux_weight = (nspecies == 3 ? 1.0 : 0.5);

  // This is a ugly hack stolen from eos_compose_test.cpp
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.ps.GetEOS().GetBaryonMass();

  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  par_for(
      "radiation_m1_calc_nurates_opacity", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (radiation_mask(m, k, j, i)) {
          for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
            if (nspecies_ != 1) {
              abs_0_(m, nuidx, k, j, i) = 0;
              eta_0_(m, nuidx, k, j, i) = 0;
            }
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

          Real w_lorentz = u_mu_(m, 0, k, j, i);
          pack_u_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i), u_mu_(m, 2, k, j, i),
                   u_mu_(m, 3, k, j, i), u_u);
          tensor_contract(g_dd, u_u, u_d);
          pack_v_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i), u_mu_(m, 2, k, j, i),
                   u_mu_(m, 3, k, j, i), v_u);
          tensor_contract(g_dd, v_u, v_d);
          calc_proj(u_d, u_u, proj_ud);

          Real J[4]{};
          Real rnnu[4]{};
          for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
            pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                     adm.beta_u(m, 2, k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), F_d);
            const Real E = u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
            apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                          chi_(m, nuidx, k, j, i), P_dd, params_);

            // compute radiation quantities in fluid frame
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
            assemble_rT(n_d, E, F_d, P_dd, T_dd);
            J[nuidx] = calc_J_from_rT(T_dd, u_u);

            Real Gamma = compute_Gamma(w_lorentz, v_u, J[nuidx], E, F_d, params_);
            rnnu[nuidx] = u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) / Gamma;
          }

          // fluid quantities
          Real nb = w0_(m, IDN, k, j, i) / mb;
          Real p = w0_(m, IPR, k, j, i);
          Real Y = w0_(m, PYF, k, j, i);
          Real T = eos.GetTemperatureFromP(nb, p, &Y);
          Real mu_b = eos.GetBaryonChemicalPotential(nb, T, &Y);
          Real mu_q = eos.GetChargeChemicalPotential(nb, T, &Y);
          Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, T, &Y);

          Real mu_n = mu_b;
          Real mu_p = mu_b + mu_q;
          Real mu_e = mu_le - mu_q;

          // Local neutrino quantities (undesitized)
          Real nudens_0[4], nudens_1[4], chi_loc[4];
          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            nudens_0[nuidx] = rnnu[nuidx] / volform;
            nudens_1[nuidx] = J[nuidx] / volform;
            chi_loc[nuidx] = chi_(m, nuidx, k, j, i);
          }

          // Get emissivities and opacities
          Real eta_0_loc[4]{}, eta_1_loc[4]{};
          Real abs_0_loc[4]{}, abs_1_loc[4]{};
          Real scat_0_loc[4]{}, scat_1_loc[4]{};

          // Note: everything sent and received are in code units
          bns_nurates(nb, T, Y, mu_n, mu_p, mu_e, nudens_0[0], nudens_1[0], chi_loc[0],
                      nudens_0[1], nudens_1[1], chi_loc[1], nudens_0[2], nudens_1[2],
                      chi_loc[2], nudens_0[3], nudens_1[3], chi_loc[3], eta_0_loc[0],
                      eta_0_loc[1], eta_0_loc[2], eta_0_loc[3], eta_1_loc[0],
                      eta_1_loc[1], eta_1_loc[2], eta_1_loc[3], abs_0_loc[0],
                      abs_0_loc[1], abs_0_loc[2], abs_0_loc[3], abs_1_loc[0],
                      abs_1_loc[1], abs_1_loc[2], abs_1_loc[3], scat_0_loc[0],
                      scat_0_loc[1], scat_0_loc[2], scat_0_loc[3], scat_1_loc[0],
                      scat_1_loc[1], scat_1_loc[2], scat_1_loc[3], nurates_params_);

          assert(isfinite(scat_0_loc[0]));
          assert(isfinite(scat_0_loc[1]));
          assert(isfinite(scat_0_loc[2]));
          assert(isfinite(scat_1_loc[0]));
          assert(isfinite(scat_1_loc[1]));
          assert(isfinite(scat_1_loc[2]));

          assert(isfinite(abs_0_loc[0]));
          assert(isfinite(abs_0_loc[1]));
          assert(isfinite(abs_0_loc[2]));
          assert(isfinite(abs_1_loc[0]));
          assert(isfinite(abs_1_loc[1]));
          assert(isfinite(abs_1_loc[2]));
          abs_0_loc[3] = abs_0_loc[2];
          abs_1_loc[3] = abs_1_loc[2];

          assert(isfinite(eta_0_loc[0]));
          assert(isfinite(eta_0_loc[1]));
          assert(isfinite(eta_0_loc[2]));
          assert(isfinite(eta_1_loc[0]));
          assert(isfinite(eta_1_loc[1]));
          assert(isfinite(eta_1_loc[2]));
          eta_0_loc[2] = nux_weight * eta_0_loc[2];
          eta_1_loc[2] = nux_weight * eta_1_loc[2];
          eta_0_loc[3] = eta_0_loc[2];
          eta_1_loc[3] = eta_1_loc[2];

          // An effective optical depth used to decide whether to compute
          // the black body function for neutrinos assuming neutrino trapping
          // or at a fixed temperature and Ye
          Real nudens_0_trap[4], nudens_1_trap[4];
          Real nudens_0_thin[4], nudens_1_thin[4];
          Real tau = Kokkos::min(Kokkos::sqrt(abs_1_loc[0] * scat_1_loc[0]),
                                 Kokkos::sqrt(abs_1_loc[1] * scat_1_loc[1])) *
                     beta_dt;

          // Compute the neutrino black body function assuming trapped neutrinos
          if (nurates_params_.opacity_tau_trap >= 0 &&
              tau > nurates_params_.opacity_tau_trap) {
            Real temperature_trap{}, Ye_trap{};

            // Compute local neutrino densities (undensitized)
            Real nudens_0[3] = {rnnu[0] / volform, rnnu[1] / volform, rnnu[2] / volform};
            Real nudens_1[3] = {J[0] / volform, J[1] / volform, J[2] / volform};

            if (nspecies_ == 4) {
              nudens_0[2] += rnnu[3] / volform;
              nudens_1[2] += J[3] / volform;
            }

            // @TODO: call weak equilibrium
            bool ierr{};

            if (ierr) {

            }
            assert(Kokkos::isfinite(nudens_0_trap[0]));
            assert(Kokkos::isfinite(nudens_0_trap[1]));
            assert(Kokkos::isfinite(nudens_0_trap[2]));
            assert(Kokkos::isfinite(nudens_1_trap[0]));
            assert(Kokkos::isfinite(nudens_1_trap[1]));
            assert(Kokkos::isfinite(nudens_1_trap[2]));
            nudens_0_trap[2] = nux_weight * nudens_0_trap[2];
            nudens_1_trap[2] = nux_weight * nudens_1_trap[2];
            nudens_0_trap[3] = nudens_0_trap[2];
            nudens_1_trap[3] = nudens_1_trap[2];
          }

          // Compute the neutrino black body function assuming fixed temperature and Y_e
          // @TODO: call neutrino density
          bool ierr{};

          nudens_0_thin[2] = nux_weight * nudens_0_thin[2];
          nudens_1_thin[2] = nux_weight * nudens_1_thin[2];
          nudens_0_thin[3] = nudens_0_thin[2];
          nudens_1_thin[3] = nudens_1_thin[2];

          // Correct cross-sections for incoming neutrino energy
          for (int ig = 0; ig < nspecies_; ++ig) {
            // Set the neutrino black body function
            Real nudens_0, nudens_1;
            if (nurates_params_.opacity_tau_trap < 0 ||
                tau <= nurates_params_.opacity_tau_trap) {
              nudens_0 = nudens_0_thin[ig];
              nudens_1 = nudens_1_thin[ig];
            } else if (tau > nurates_params_.opacity_tau_trap +
                                 nurates_params_.opacity_tau_delta) {
              nudens_0 = nudens_0_trap[ig];
              nudens_1 = nudens_1_trap[ig];
            } else {
              const Real lam = (tau - nurates_params_.opacity_tau_trap) /
                               nurates_params_.opacity_tau_delta;
              nudens_0 = lam * nudens_0_trap[ig] + (1 - lam) * nudens_0_thin[ig];
              nudens_1 = lam * nudens_1_trap[ig] + (1 - lam) * nudens_1_thin[ig];
            }

            // Correct absorption opacities for non-LTE effects
            // (kappa ~ E_nu^2)
            Real corr_fac = 1.0;
            corr_fac = (J[ig] / rnnu[ig]) * (nudens_0 / nudens_1);
            if (!isfinite(corr_fac)) {
              corr_fac = 1.0;
            }
            corr_fac *= corr_fac;
            corr_fac =
                Kokkos::max(1.0 / nurates_params_.opacity_corr_fac_max,
                            Kokkos::min(corr_fac, nurates_params_.opacity_corr_fac_max));

            // Extract scattering opacity
            scat_1_(m, k, j, i) = corr_fac * (scat_1_loc[ig] - abs_1_loc[ig]);

            // Enforce Kirchhoff's laws.
            // . For the heavy lepton neutrinos this is implemented by
            //   changing the opacities.
            // . For the electron type neutrinos this is implemented by
            //   changing the emissivities.
            // It would be better to have emissivities and absorptivities
            // that satisfy Kirchhoff's law.
            if (ig == 2 || ig == 3) {
              eta_0_(m, k, j, i) = eta_0_loc[ig];
              eta_1_(m, k, j, i) = eta_1_loc[ig];
              abs_0_(m, k, j, i) =
                  (nudens_0 > params_.rad_N_floor ? eta_0_(m, k, j, i) / nudens_0 : 0);
              abs_1_(m, k, j, i) =
                  (nudens_1 > params_.rad_E_floor ? eta_1_(m, k, j, i) / nudens_1 : 0);
            } else {
              abs_0_(m, k, j, i) = corr_fac * abs_0_loc[ig];
              abs_1_(m, k, j, i) = corr_fac * abs_1_loc[ig];
              eta_0_(m, k, j, i) = abs_0_(m, k, j, i) * nudens_0;
              eta_1_(m, k, j, i) = abs_1_(m, k, j, i) * nudens_1;
            }
          }
        }
      });

  return TaskStatus::complete;
}
}  // namespace radiationm1