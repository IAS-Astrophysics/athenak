//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_update.cpp
//! \brief beam time update for grey M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "globals.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_compute_opacities.hpp"
#include "radiation_m1_helpers.hpp"
#include "z4c/z4c.hpp"

namespace radiationm1 {

//! \file radiation_m1_update.cpp
//! \brief perform update for M1 Steps
//!  1. F^m   = F^k + dt/2 [ A[F^k] + S[F^m]   ]
//!  2. F^k+1 = F^k + dt   [ A[F^m] + S[F^k+1] ]
//!  At each step we solve an implicit problem in the form
//!     F = F^* + cdt S[F]
//!  Where F^* = F^k + cdt A
TaskStatus RadiationM1::TimeUpdate(Driver *d, int stage) {
  const int NGHOST = 2;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0_ = u0;
  auto &P_dd_ = pmy_pack->pradm1->P_dd;
  auto &u1_ = u1;
  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto nvars_ = pmy_pack->pradm1->nvars;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto &source_limiter_ = source_limiter;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &params_ = pmy_pack->pradm1->params;

  Real &gam0 = d->gam0[stage - 1];
  Real &gam1 = d->gam1[stage - 1];
  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);
  Real dt = pmy_pack->pmesh->dt;

  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  int scr_level = 0;
  size_t scr_size = 1;

  par_for_outer(
      "radiation_m1_update", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks,
      ke, js, je, is, ie,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                    const int i) {
        Real volform = 1; //@TODO: fix this!

        // 4-metric, 3-metric inverse, shift, normal, extrinsic curvature
        Real garr_dd[16];
        Real garr_uu[16];
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> K_dd{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> gamma_ud{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_u{};
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
        pack_n_d(adm.alpha(m, k, j, i), n_d);
        tensor_contract(g_uu, n_d, n_u);
        pack_beta_u(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                    adm.beta_u(m, 2, k, j, i), beta_u);
        tensor_contract(g_dd, beta_u, beta_d);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            g_dd(a, b) = garr_dd[a + b * 4];
            g_uu(a, b) = garr_uu[a + b * 4];
          }
        }
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            gamma_ud(a, b) = (a == b) + n_u(a) * n_d(b);
          }
        }
        K_dd(0, 0) = adm.vK_dd(m, 0, 0, k, j, i);
        K_dd(0, 1) = adm.vK_dd(m, 0, 1, k, j, i);
        K_dd(0, 2) = adm.vK_dd(m, 0, 2, k, j, i);
        K_dd(1, 1) = adm.vK_dd(m, 1, 1, k, j, i);
        K_dd(1, 2) = adm.vK_dd(m, 1, 2, k, j, i);
        K_dd(3, 3) = adm.vK_dd(m, 2, 2, k, j, i);

        // Lorentz factor, four velocity, three velocity, projection
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> proj_ud{};
        Real w_lorentz = u_mu_(m, 0, k, j, i);
        pack_u_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i),
                 u_mu_(m, 2, k, j, i), u_mu_(m, 3, k, j, i), u_u);
        tensor_contract(g_dd, u_u, u_d);
        pack_v_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i),
                 u_mu_(m, 2, k, j, i), u_mu_(m, 3, k, j, i), v_u);
        tensor_contract(g_dd, v_u, v_d);
        calc_proj(u_d, u_u, proj_ud);

        // derivatives of lapse
        Real ideltax[3] = {1 / mbsize.d_view(m).dx1, 1 / mbsize.d_view(m).dx2,
                           1 / mbsize.d_view(m).dx3};
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
        dalpha_d(0) = Dx<NGHOST>(0, ideltax, adm.alpha, m, k, j, i);
        dalpha_d(1) =
            (multi_d) ? Dx<NGHOST>(1, ideltax, adm.alpha, m, k, j, i) : 0.;
        dalpha_d(2) =
            (three_d) ? Dx<NGHOST>(2, ideltax, adm.alpha, m, k, j, i) : 0.;

        // derivatives of shift (\p_i beta_u(j))
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du;
        for (int a = 0; a < 3; ++a) {
          dbeta_du(0, a) = Dx<NGHOST>(0, ideltax, adm.beta_u, m, a, k, j, i);
          dbeta_du(1, a) =
              (multi_d) ? Dx<NGHOST>(1, ideltax, adm.beta_u, m, a, k, j, i)
                        : 0.;
          dbeta_du(2, a) =
              (three_d) ? Dx<NGHOST>(2, ideltax, adm.beta_u, m, a, k, j, i)
                        : 0.;
        }

        // derivatives of spatial metric (\p_k gamma_ij)
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            dg_ddd(0, a, b) =
                Dx<NGHOST>(0, ideltax, adm.g_dd, m, a, b, k, j, i);
            dg_ddd(1, a, b) =
                (multi_d) ? Dx<NGHOST>(1, ideltax, adm.g_dd, m, a, b, k, j, i)
                          : 0.;
            dg_ddd(2, a, b) =
                (three_d) ? Dx<NGHOST>(2, ideltax, adm.g_dd, m, a, b, k, j, i)
                          : 0.;
          }
        }

        // @TODO: call opacities here
        M1Opacities opacities = ComputeM1Opacities(params_);

        // [1] Compute contribution from flux and geometric sources
        Real rEFN[M1_TOTAL_NUM_SPECIES][5];
        par_for_inner(member, 0, nspecies_ - 1, [&](const int nuidx) {
          // [1.A: fluxes]
          for (int var = 0; var < nvars_; ++var) {
            rEFN[nuidx][var] =
                (flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i + 1) -
                 flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                ideltax[0];
            if (multi_d) {
              rEFN[nuidx][var] +=
                  (flx2(m, CombinedIdx(nuidx, var, nvars_), k, j + 1, i) -
                   flx2(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                  ideltax[1];
            }
            if (three_d) {
              rEFN[nuidx][var] +=
                  (flx3(m, CombinedIdx(nuidx, var, nvars_), k + 1, j, i) -
                   flx3(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                  ideltax[2];
            }
          }
          // [1.B geometric sources]
          // Load lab radiation quantities
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u{};
          pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                   u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), F_d);
          tensor_contract(g_uu, F_d, F_u);

          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_uu{};
          pack_P_dd(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                    adm.beta_u(m, 2, k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 0, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 1, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 2, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 3, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 4, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 5, 6), k, j, i), P_dd);
          const Real E = u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i);

          // geometric sources
          // rEFN[M1_E_IDX] +=
          //    adm.alpha(m, k, j, i) * tensor_dot(g_uu, P_dd, K_dd) -
          //    tensor_dot(g_uu, F_d, dalpha_d); @TODO: fix this!
          for (int a = 0; a < 3; ++a) {
            rEFN[nuidx][a + 1] -= E * dalpha_d(a);
            for (int b = 0; b < 3; ++b) {
              rEFN[nuidx][a + 1] += F_d(b) * dbeta_du(a, b);
            }
            for (int b = 0; b < 3; ++b)
              for (int c = 0; c < 3; ++c) {
                rEFN[nuidx][a + 1] +=
                    adm.alpha(m, k, j, i) / 2. * P_uu(b, c) * dg_ddd(a, b, c);
              }
          }
        });
        member.team_barrier();

        // [2] Compute sources
        Real DrEFN[M1_TOTAL_NUM_SPECIES][5]{};
        par_for_inner(member, 0, nspecies_ - 1, [&](const int nuidx) {
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> S_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> tS_d{};
          pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                   u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), F_d);
          const Real E = u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i);
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
          pack_P_dd(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                    adm.beta_u(m, 2, k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 0, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 1, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 2, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 3, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 4, 6), k, j, i),
                    P_dd_(m, CombinedIdx(nuidx, 5, 6), k, j, i), P_dd);

          if (params_.src_update == Explicit) {
            // Compute fluid radiation quantities
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
            assemble_rT(n_d, E, F_d, P_dd, T_dd);
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_u{};

            Real J = calc_J_from_rT(T_dd, u_u);
            calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
            tensor_contract(g_uu, H_d, H_u);
            const Real Gamma =
                compute_Gamma(w_lorentz, v_u, J, E, F_d, params_);

            // Compute radiation sources
            calc_rad_sources(opacities.eta_1[nuidx] * volform,
                             opacities.abs_1[nuidx], opacities.scat_1[nuidx],
                             u_d, J, H_d, S_d);
            DrEFN[nuidx][M1_E_IDX] =
                dt * calc_rE_source(adm.alpha(m, k, j, i), n_u, S_d);

            calc_rF_source(adm.alpha(m, k, j, i), gamma_ud, S_d, tS_d);
            DrEFN[nuidx][M1_FX_IDX] = dt * tS_d(1);
            DrEFN[nuidx][M1_FY_IDX] = dt * tS_d(2);
            DrEFN[nuidx][M1_FZ_IDX] = dt * tS_d(3);

            if (nspecies_ > 1) {
              const Real N =
                  u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i);
              DrEFN[nuidx][M1_N_IDX] = dt * adm.alpha(m, k, j, i) *
                                       (volform * opacities.eta_0[nuidx] -
                                        opacities.abs_0[nuidx] * N / Gamma);
            }
          }
          // semi-implicit solver here
        });
        member.team_barrier();

        // [3] Limit sources
        Real tau;
        Real DDxp[M1_TOTAL_NUM_SPECIES];
        Real dens{};
        Real source_Ye_max{};
        Real source_Ye_min{};
        Real Y_e{};
        Real theta = 1.0;
        if (source_limiter_ >= 0) {
          theta = 1.0;
          Real DTau_sum = 0.0;
          for (int ig = 0; ig < nspecies_; ++ig) {
            Real Estar = u0_(m, CombinedIdx(ig, M1_E_IDX, nvars_), k, j, i) +
                         dt * rEFN[ig][M1_E_IDX];
            if (DrEFN[ig][M1_E_IDX] < 0) {
              theta = Kokkos::min(-source_limiter * Kokkos::max(Estar, 0.0) /
                                      DrEFN[ig][M1_E_IDX],
                                  theta);
            }
            DTau_sum -= DrEFN[ig][M1_E_IDX];
          }
          if (DTau_sum < 0) {
            theta = Kokkos::min(
                -source_limiter * Kokkos::max(tau, 0.0) / DTau_sum, theta);
          }

          if (nspecies_ > 1) {
            Real DDxp_sum = 0.0;
            for (int ig = 0; ig < nspecies_; ++ig) {
              Real Nstar = u0_(m, CombinedIdx(ig, M1_N_IDX, nvars_), k, j, i) +
                           dt * rEFN[ig][M1_N_IDX];
              if (DrEFN[ig][M1_N_IDX] < 0) {
                theta = Kokkos::min(-source_limiter * Kokkos::max(Nstar, 0.0) /
                                        DrEFN[ig][M1_N_IDX],
                                    theta);
              }
              DDxp_sum += DDxp[ig];
            }
            const Real DYe = DDxp_sum / dens;
            if (DYe > 0) {
              theta = Kokkos::min<Real>(
                  source_limiter * Kokkos::max(source_Ye_max - Y_e, 0.0) / DYe,
                  theta);
            } else if (DYe < 0) {
              theta = Kokkos::min<Real>(
                  source_limiter * Kokkos::min(source_Ye_min - Y_e, 0.0) / DYe,
                  theta);
            }
          }
          theta = Kokkos::max<Real>(0.0, theta);
        }

        // [4] update fields
        par_for_inner(member, 0, nspecies_ - 1, [&](const int nuidx) {
          Real Ef = u1_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) -
                    beta_dt * rEFN[nuidx][M1_E_IDX] +
                    theta * DrEFN[nuidx][M1_E_IDX];
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Ff_d{};
          Real Fxf = u1_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) -
                     beta_dt * rEFN[nuidx][M1_FX_IDX] +
                     theta * DrEFN[nuidx][M1_FX_IDX];
          Real Fyf = u1_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) -
                     beta_dt * rEFN[nuidx][M1_FY_IDX] +
                     theta * DrEFN[nuidx][M1_FY_IDX];
          Real Fzf = u1_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) -
                     beta_dt * rEFN[nuidx][M1_FZ_IDX] +
                     theta * DrEFN[nuidx][M1_FZ_IDX];
          pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i), Fxf, Fyf, Fzf, Ff_d);
          apply_floor(g_uu, Ef, Ff_d, params_);
          u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) = Ef;
          u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) = Fxf;
          u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) = Fyf;
          u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) = Fzf;

          if (nspecies_ > 1) {
            Real Nf = u1_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) -
                      beta_dt * rEFN[nuidx][M1_N_IDX] +
                      theta * DrEFN[nuidx][M1_N_IDX];
            Nf = Kokkos::max<Real>(Nf, params_.rad_N_floor);
            u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) = Nf;
          }
        });
      });
  return TaskStatus::complete;
}
} // namespace radiationm1