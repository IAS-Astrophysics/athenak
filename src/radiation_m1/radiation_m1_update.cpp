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
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_calc_closure.hpp"
#include "radiation_m1_helpers.hpp"
#include "radiation_m1_sources.hpp"
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
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0_ = u0;
  auto &chi_ = pmy_pack->pradm1->chi;
  auto &u1_ = u1;
  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;

  auto &eta_0_ = pmy_pack->pradm1->eta_0;
  auto &abs_0_ = pmy_pack->pradm1->abs_0;
  auto &eta_1_ = pmy_pack->pradm1->eta_1;
  auto &abs_1_ = pmy_pack->pradm1->abs_1;
  auto &scat_1_ = pmy_pack->pradm1->scat_1;

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto nvars_ = pmy_pack->pradm1->nvars;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &params_ = pmy_pack->pradm1->params;

  auto &BrentFunc_ = pmy_pack->pradm1->BrentFunc;
  auto &HybridsjFunc_ = pmy_pack->pradm1->HybridsjFunc;

  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  int scr_level = 0;
  size_t scr_size = 1;

  par_for_outer(
      "radiation_m1_update", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
      is, ie,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                    const int i) {
        // [A] Compute gr quantities: metric, shift, extrinsic curvature, etc.
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
        adm::SpacetimeMetric(adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
                             adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                             adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                             adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                             adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
                             garr_dd);
        adm::SpacetimeUpperMetric(adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
                                  adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                  adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                                  adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                                  adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
                                  garr_uu);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            g_dd(a, b) = garr_dd[a + b * 4];
            g_uu(a, b) = garr_uu[a + b * 4];
          }
        }
        pack_n_d(adm.alpha(m, k, j, i), n_d);
        tensor_contract(g_uu, n_d, n_u);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            gamma_ud(a, b) = (a == b) + n_u(a) * n_d(b);
          }
        }

        pack_beta_u(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                    adm.beta_u(m, 2, k, j, i), beta_u);
        tensor_contract(g_dd, beta_u, beta_d);

        K_dd(0, 0) = adm.vK_dd(m, 0, 0, k, j, i);
        K_dd(0, 1) = adm.vK_dd(m, 0, 1, k, j, i);
        K_dd(0, 2) = adm.vK_dd(m, 0, 2, k, j, i);
        K_dd(1, 1) = adm.vK_dd(m, 1, 1, k, j, i);
        K_dd(1, 2) = adm.vK_dd(m, 1, 2, k, j, i);
        K_dd(3, 3) = adm.vK_dd(m, 2, 2, k, j, i);

        Real gam =
            adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
        Real volform = Kokkos::sqrt(gam);

        // [B] Compute derivatives of gr quantities (note: 3-arrays, no \p_t!)
        // [B.1] Derivatives of lapse (\p_i alpha)
        Real ideltax[3] = {1 / mbsize.d_view(m).dx1, 1 / mbsize.d_view(m).dx2,
                           1 / mbsize.d_view(m).dx3};
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d{};
        dalpha_d(0) = Dx<M1_NGHOST>(0, ideltax, adm.alpha, m, k, j, i);
        dalpha_d(1) = (multi_d) ? Dx<M1_NGHOST>(1, ideltax, adm.alpha, m, k, j, i) : 0.;
        dalpha_d(2) = (three_d) ? Dx<M1_NGHOST>(2, ideltax, adm.alpha, m, k, j, i) : 0.;

        // [B.2] Derivatives of shift (\p_i beta_u(j))
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du{};
        for (int a = 0; a < 3; ++a) {
          dbeta_du(0, a) = Dx<M1_NGHOST>(0, ideltax, adm.beta_u, m, a, k, j, i);
          dbeta_du(1, a) =
              (multi_d) ? Dx<M1_NGHOST>(1, ideltax, adm.beta_u, m, a, k, j, i) : 0.;
          dbeta_du(2, a) =
              (three_d) ? Dx<M1_NGHOST>(2, ideltax, adm.beta_u, m, a, k, j, i) : 0.;
        }

        // [B.3] Derivatives of spatial metric (\p_k gamma_ij)
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd{};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            dg_ddd(0, a, b) = Dx<M1_NGHOST>(0, ideltax, adm.g_dd, m, a, b, k, j, i);
            dg_ddd(1, a, b) =
                (multi_d) ? Dx<M1_NGHOST>(1, ideltax, adm.g_dd, m, a, b, k, j, i) : 0.;
            dg_ddd(2, a, b) =
                (three_d) ? Dx<M1_NGHOST>(2, ideltax, adm.g_dd, m, a, b, k, j, i) : 0.;
          }
        }

        // [C] Compute fluid quantities
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

        // [D] Capture quantities from EOS
        Real mb{};
        Real dens{};
        Real Y_e{};
        Real tau{};

        // [E] Compute contribution from flux and geometric sources
        Real rEFN[M1_TOTAL_NUM_SPECIES][5];
        Real DDxp[M1_TOTAL_NUM_SPECIES];
        for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
          // [E.1: Contribution from fluxes]
          for (int var = 0; var < nvars_; ++var) {
            rEFN[nuidx][var] = -(flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i + 1) -
                                 flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                               ideltax[0];
            if (multi_d) {
              rEFN[nuidx][var] +=
                  -(flx2(m, CombinedIdx(nuidx, var, nvars_), k, j + 1, i) -
                    flx2(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                  ideltax[1];
            }
            if (three_d) {
              rEFN[nuidx][var] +=
                  -(flx3(m, CombinedIdx(nuidx, var, nvars_), k + 1, j, i) -
                    flx3(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                  ideltax[2];
            }
          }
          // [E.2 Contribution from geometric sources]
          // Load lab radiation quantities
          if (params_.gr_sources) {
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u{};
            const Real E = u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i);
            pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                     u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), F_d);
            tensor_contract(g_uu, F_d, F_u);

            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_uu{};
            apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                          chi_(m, nuidx, k, j, i), P_dd, params_);

            // geometric sources
            rEFN[nuidx][M1_E_IDX] +=
                adm.alpha(m, k, j, i) * tensor_dot(g_uu, P_dd, K_dd) -
                tensor_dot(g_uu, F_d, dalpha_d);
            for (int a = 0; a < 3; ++a) {
              rEFN[nuidx][a + 1] -= E * dalpha_d(a);
              for (int b = 0; b < 3; ++b) {
                rEFN[nuidx][a + 1] += F_d(b) * dbeta_du(a, b);
              }
              for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                  rEFN[nuidx][a + 1] +=
                      adm.alpha(m, k, j, i) / 2. * P_uu(b, c) * dg_ddd(a, b, c);
                }
              }
            }
          }
        }

        // [F] Compute contribution from matter sources
        Real DrEFN[M1_TOTAL_NUM_SPECIES][5]{};
        Real theta{};
        if (params_.matter_sources) {
          for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
            // radiation fields
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> S_d{};
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> tS_d{};
            pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                     u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), F_d);
            const Real E = u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
            apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                          chi_(m, nuidx, k, j, i), P_dd, params_);

            if (params_.src_update == Explicit) {
              // compute radiation quantities in fluid frame
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
              assemble_rT(n_d, E, F_d, P_dd, T_dd);
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_u{};

              Real J = calc_J_from_rT(T_dd, u_u);
              calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
              tensor_contract(g_uu, H_d, H_u);
              const Real Gamma = compute_Gamma(w_lorentz, v_u, J, E, F_d, params_);

              // Compute radiation sources
              calc_rad_sources(eta_1_(m, nuidx, k, j, i) * volform,
                               abs_1_(m, nuidx, k, j, i), scat_1_(m, nuidx, k, j, i), u_d,
                               J, H_d, S_d);
              DrEFN[nuidx][M1_E_IDX] =
                  beta_dt * calc_rE_source(adm.alpha(m, k, j, i), n_u, S_d);

              calc_rF_source(adm.alpha(m, k, j, i), gamma_ud, S_d, tS_d);
              DrEFN[nuidx][M1_FX_IDX] = beta_dt * tS_d(1);
              DrEFN[nuidx][M1_FY_IDX] = beta_dt * tS_d(2);
              DrEFN[nuidx][M1_FZ_IDX] = beta_dt * tS_d(3);

              if (nspecies_ > 1) {
                const Real N = u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i);
                DrEFN[nuidx][M1_N_IDX] = beta_dt * adm.alpha(m, k, j, i) *
                                         (volform * eta_0_(m, nuidx, k, j, i) -
                                          abs_0_(m, nuidx, k, j, i) * N / Gamma);
              }
            }

            if (params_.src_update == Implicit) {
              // Boost to the fluid frame, compute fluid matter interaction and
              // boost back. Use these values for implicit solve

              // advect radiation
              Real Estar = u1_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) +
                           beta_dt * rEFN[nuidx][M1_E_IDX];
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fstar_d{};
              pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                       u1_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) +
                           beta_dt * rEFN[nuidx][M1_FX_IDX],
                       u1_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) +
                           beta_dt * rEFN[nuidx][M1_FY_IDX],
                       u1_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) +
                           beta_dt * rEFN[nuidx][M1_FZ_IDX],
                       Fstar_d);
              apply_floor(g_uu, Estar, Fstar_d, params_);
              Real Nstar{};
              if (nspecies_ > 1) {
                Nstar = Kokkos::max<Real>(
                    u1_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) +
                        beta_dt * rEFN[nuidx][M1_N_IDX],
                    params_.rad_N_floor);
              }

              // Compute quantities in the fluid frame
              Real Enew{};
              Real chival{};
              calc_closure(BrentFunc_, g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
                           Estar, Fstar_d, chival, P_dd, params_, params_.closure_type);
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd{};
              assemble_rT(n_d, Estar, Fstar_d, P_dd, rT_dd);
              const Real Jstar = calc_J_from_rT(rT_dd, u_u);
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hstar_d{};
              calc_H_from_rT(rT_dd, u_u, proj_ud, Hstar_d);

              // Estimate interaction with matter
              const Real dtau = beta_dt * (adm.alpha(m, k, j, i) / w_lorentz);
              Real Jnew = (Jstar + dtau * eta_1_(m, nuidx, k, j, i) * volform) /
                          (1 + dtau * abs_1_(m, nuidx, k, j, i));

              // Only three components of H^a are independent H^0 is found by
              // requiring H^a u_a = 0
              const Real khat = (abs_1_(m, nuidx, k, j, i) + scat_1_(m, nuidx, k, j, i));
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hnew_d{};
              for (int a = 1; a < 4; ++a) {
                Hnew_d(a) = Hstar_d(a) / (1 + dtau * khat);
              }
              Hnew_d(0) = 0.0;
              for (int a = 1; a < 4; ++a) {
                Hnew_d(0) -= Hnew_d(a) * (u_u(a) / u_u(0));
              }

              // Update Tmunu
              const Real H2 = tensor_dot(g_uu, Hnew_d, Hnew_d);
              const Real xi =
                  Kokkos::sqrt(H2) * (Jnew > params_.rad_E_floor ? 1 / Jnew : 0);
              chival = closure_fun(xi, params_.closure_type);
              // chival = 1. / 3.;

              const Real dthick = 3. * (1. - chival) / 2.;
              const Real dthin = 1. - dthick;

              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> K_thin_dd{};
              calc_Kthin(g_uu, n_d, w_lorentz, u_d, proj_ud, Jnew, Hnew_d, K_thin_dd,
                         params_.rad_E_floor);

              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> K_thick_dd{};
              calc_Kthick(g_dd, u_d, Jnew, Hnew_d, K_thick_dd);

              for (int a = 0; a < 4; ++a) {
                for (int b = a; b < 4; ++b) {
                  rT_dd(a, b) =
                      Jnew * u_d(a) * u_d(b) + Hnew_d(a) * u_d(b) + Hnew_d(b) * u_d(a) +
                      dthin * Jnew * (Hnew_d(a) * Hnew_d(b) * (H2 > 0 ? 1 / H2 : 0)) +
                      dthick * Jnew * (g_dd(a, b) + u_d(a) * u_d(b)) / 3;
                }
              }

              // Boost back to the lab frame
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fnew_d{};
              Enew = calc_J_from_rT(rT_dd, n_u);
              calc_H_from_rT(rT_dd, n_u, gamma_ud, Fnew_d);
              apply_floor(g_uu, Enew, Fnew_d, params_);

              auto src_signal = source_update(
                  BrentFunc_, HybridsjFunc_, beta_dt, adm.alpha(m, k, j, i), g_dd, g_uu,
                  n_d, n_u, gamma_ud, u_d, u_u, v_d, v_u, proj_ud, w_lorentz, Estar,
                  Fstar_d, Estar, Fstar_d, volform * eta_1_(m, nuidx, k, j, i),
                  abs_1_(m, nuidx, k, j, i), scat_1_(m, nuidx, k, j, i), chival, Enew,
                  Fnew_d, params_, params_.closure_type);
              apply_floor(g_uu, Enew, Fnew_d, params_);

              // Update closure
              apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, Enew, Fnew_d,
                            chival, P_dd, params_);

              // Compute new radiation energy density in the fluid frame
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
              assemble_rT(n_d, Enew, Fnew_d, P_dd, T_dd);
              Jnew = calc_J_from_rT(T_dd, u_u);

              // Compute changes in radiation energy and momentum
              DrEFN[nuidx][M1_E_IDX] = Enew - Estar;
              DrEFN[nuidx][M1_FX_IDX] = Fnew_d(1) - Fstar_d(1);
              DrEFN[nuidx][M1_FY_IDX] = Fnew_d(2) - Fstar_d(2);
              DrEFN[nuidx][M1_FZ_IDX] = Fnew_d(3) - Fstar_d(3);

              if (nspecies_ > 1) {
                // Compute updated Gamma
                const Real Gamma =
                    compute_Gamma(w_lorentz, v_u, Jnew, Enew, Fnew_d, params_);

                // N^k+1 = N^* + dt ( eta - abs N^k+1 )
                DrEFN[nuidx][M1_N_IDX] =
                    (Nstar + beta_dt * adm.alpha(m, k, j, i) * volform *
                                 eta_0_(m, nuidx, k, j, i)) /
                        (1 + beta_dt * adm.alpha(m, k, j, i) * abs_0_(m, nuidx, k, j, i) /
                                 Gamma) -
                    Nstar;
              }
            }
            // fluid lepton sources
            if (nspecies_ > 1) {
              DDxp[nuidx] = -mb * (DrEFN[nuidx][M1_N_IDX] * (nuidx == 0) -
                                   DrEFN[nuidx][M1_N_IDX] * (nuidx == 1));
            }
          }

          // [G] Limit sources
          theta = 1.0;
          if (params_.theta_limiter && params_.source_limiter >= 0) {
            theta = 1.0;
            Real DTau_sum = 0.0;
            for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
              Real Estar = u1_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) +
                           beta_dt * rEFN[nuidx][M1_E_IDX];
              if (DrEFN[nuidx][M1_E_IDX] < 0) {
                theta = Kokkos::min<Real>(-params_.source_limiter *
                                              Kokkos::max<Real>(Estar, 0.0) /
                                              DrEFN[nuidx][M1_E_IDX],
                                          theta);
              }
              DTau_sum -= DrEFN[nuidx][M1_E_IDX];
            }
            if (DTau_sum < 0) {
              theta = Kokkos::min(
                  -params_.source_limiter * Kokkos::max<Real>(tau, 0.0) / DTau_sum,
                  theta);
            }

            if (nspecies_ > 1) {
              Real DDxp_sum = 0.0;
              for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
                Real Nstar = u1_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) +
                             beta_dt * rEFN[nuidx][M1_N_IDX];
                if (DrEFN[nuidx][M1_N_IDX] < 0) {
                  theta = Kokkos::min(-params_.source_limiter *
                                          Kokkos::max<Real>(Nstar, 0.0) /
                                          DrEFN[nuidx][M1_N_IDX],
                                      theta);
                }
                DDxp_sum += DDxp[nuidx];
              }
              const Real DYe = DDxp_sum / dens;
              if (DYe > 0) {
                theta = Kokkos::min<Real>(
                    params_.source_limiter *
                        Kokkos::max<Real>(params_.source_Ye_max - Y_e, 0.0) / DYe,
                    theta);
              } else if (DYe < 0) {
                theta = Kokkos::min<Real>(
                    params_.source_limiter *
                        Kokkos::min(params_.source_Ye_min - Y_e, 0.0) / DYe,
                    theta);
              }
            }
            theta = Kokkos::max<Real>(0.0, theta);
          }
        } else {
          theta = 0;
        }

        // [H] Update fields
        for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
          Real Ef = u1_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) +
                    beta_dt * rEFN[nuidx][M1_E_IDX] + theta * DrEFN[nuidx][M1_E_IDX];
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Ff_d{};
          Real Fxf = u1_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) +
                     beta_dt * rEFN[nuidx][M1_FX_IDX] + theta * DrEFN[nuidx][M1_FX_IDX];
          Real Fyf = u1_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) +
                     beta_dt * rEFN[nuidx][M1_FY_IDX] + theta * DrEFN[nuidx][M1_FY_IDX];
          Real Fzf = u1_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) +
                     beta_dt * rEFN[nuidx][M1_FZ_IDX] + theta * DrEFN[nuidx][M1_FZ_IDX];
          pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i), Fxf, Fyf, Fzf, Ff_d);
          apply_floor(g_uu, Ef, Ff_d, params_);
          u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) = Ef;
          u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) =
              Ff_d(1);  //@TODO: fix this with floored value
          u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) = Ff_d(2);
          u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) = Ff_d(3);

          if (nspecies_ > 1) {
            Real Nf = u1_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) -
                      beta_dt * rEFN[nuidx][M1_N_IDX] + theta * DrEFN[nuidx][M1_N_IDX];
            Nf = Kokkos::max<Real>(Nf, params_.rad_N_floor);
            u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) = Nf;
          }
        }
      });
  return TaskStatus::complete;
}
}  // namespace radiationm1