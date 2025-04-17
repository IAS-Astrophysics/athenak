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
  auto &nurates_params_ = pmy_pack->pradm1->nurates_params;
  auto &radiation_mask_ = pmy_pack->pradm1->radiation_mask;
  auto &adm = pmy_pack->padm->adm;

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
          const Real nux_weight = (nspecies_ == 3) ? 1.0 : 0.5;

          Real rho{};
          Real temperature{};
          Real Y_e{};

          Real kappa_0_loc[4]{}, kappa_1_loc[4]{}, abs_0_loc[4]{}, abs_1_loc[4]{},
              eta_0_loc[4]{}, eta_1_loc[4]{};

          // Calculate opacities
          // get the transport absorption and scattering opacities here

          const Real tau = Kokkos::min(Kokkos::sqrt(abs_1_loc[0] * kappa_1_loc[0]),
                                       Kokkos::sqrt(abs_1_loc[1] * kappa_1_loc[1])) *
                           beta_dt;

          // Compute neutrino black body function assuming trapped neutrinos
          Real nudens_0_trap[4]{}, nudens_1_trap[4]{};

          if (params_.opacity_tau_trap >= 0 && tau > params_.opacity_tau_trap) {
            // compute or get these things
            Real rJ[4]{};
            Real nb{};
            Real temp{};
            Real ye{};
            Real mu_n{};
            Real mu_p{};
            Real mu_e{};

            Real temperature_trap{}, Y_e_trap{};

            Real gam =
                adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                                adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                                adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
            Real volform = Kokkos::sqrt(gam);

            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
            Real w_lorentz = u_mu_(m, 0, k, j, i);
            pack_u_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i), u_mu_(m, 2, k, j, i),
                     u_mu_(m, 3, k, j, i), u_u);
            pack_v_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i), u_mu_(m, 2, k, j, i),
                     u_mu_(m, 3, k, j, i), v_u);

            Real rnnu[4]{};
            for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
              pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                       adm.beta_u(m, 2, k, j, i),
                       u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                       u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                       u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), F_d);
              const Real E = u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);

              //const Real Gamma = compute_Gamma(w_lorentz, v_u, J, E, F_d, params_);
              //rnnu[nuidx] = u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) / Gamma;
            }

            // Compute undensitized local neutrino densities
            Real nudens_0[3] = {rnnu[0] / volform, rnnu[1] / volform, rnnu[2] / volform};
            Real nudens_1[3] = {rJ[0] / volform, rJ[1] / volform, rJ[2] / volform};

            if (nspecies_ == 4) {
              nudens_0[2] += rnnu[3] / volform;
              nudens_1[2] += rJ[3] / volform;
            }

            /*
            auto ierr = WeakEquilibrium(
                rho, temp, ye, nudens_0[0], nudens_0[1], nudens_0[2], nudens_1[0],
                nudens_1[1], nudens_1[2], &temperature_trap, &Y_e_trap, &nudens_0_trap[0],
                &nudens_0_trap[1], &nudens_0_trap[2], &nudens_1_trap[0],
                &nudens_1_trap[1], &nudens_1_trap[2]);

            if (ierr) {
              // Try to recompute the weak equilibrium using neglecting
              // current neutrino data
              ierr = WeakEquilibrium(
                  rho, temp, ye, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &temperature_trap,
                  &Y_e_trap, &nudens_0_trap[0], &nudens_0_trap[1], &nudens_0_trap[2],
                  &nudens_1_trap[0], &nudens_1_trap[1], &nudens_1_trap[2]);
            }*/
          }
          nudens_0_trap[2] = nux_weight * nudens_0_trap[2];
          nudens_1_trap[2] = nux_weight * nudens_1_trap[2];
          nudens_0_trap[3] = nudens_0_trap[2];
          nudens_1_trap[3] = nudens_1_trap[2];

          // These are fluid frame quantities
          Real n_nue{};
          Real j_nue{};
          Real chi_nue{};
          Real n_nua{};
          Real j_nua{};
          Real chi_nua{};
          Real n_nux{};
          Real j_nux{};
          Real chi_nux{};
          /*
          bns_nurates(&nb, &temp, &ye, &mu_n, &mu_p, &mu_e, &n_nue, &j_nue, &chi_nue,
                      &n_nua, &j_nua, &chi_nua, &n_nux, &j_nux, &chi_nux,
                      eta_0_(m, id_nue, k, j, i), eta_0_(m, id_anue, k, j, i),
                      eta_0_(m, id_nux, k, j, i), eta_1_(m, id_nue, k, j, i),
                      eta_1_(m, id_anue, k, j, i), eta_1_(m, id_nux, k, j, i),
                      &sigma_0_nue, Real & sigma_0_nua, Real & sigma_0_nux,
                      Real & sigma_1_nue, Real & sigma_1_nua, Real & sigma_1_nux,
                      Real & scat_0_nue, Real & scat_0_nua, Real & scat_0_nux,
                      scat_1_(m, id_nue, k, j, i), scat_1_(m, id_anue, k, j, i),
                      scat_1_(m, id_nux, k, j, i), nurates_params_);*/
        }
      });

  return TaskStatus::complete;
}
}  // namespace radiationm1