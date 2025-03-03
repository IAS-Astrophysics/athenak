//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_closure.cpp
//! \brief calculate lab frame pressure

#include "radiation_m1_calc_closure.hpp"
#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_helpers.hpp"

namespace radiationm1 {
TaskStatus RadiationM1::CalcClosure(Driver *pdrive, int stage) {
  auto &size = pmy_pack->pmb->mb_size;
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto &u0_ = pmy_pack->pradm1->u0;
  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  auto &chi_ = pmy_pack->pradm1->chi;
  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &nvars_ = pmy_pack->pradm1->nvars;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto &radiation_mask_ = pmy_pack->pradm1->radiation_mask;
  auto &closure_ = pmy_pack->pradm1->closure;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;
  RadiationM1Params &params_ = pmy_pack->pradm1->params;

  // index limits with ghost
  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;

  size_t scr_size = 1;
  int scr_level = 0;
  par_for_outer(
      "radiation_m1_calc_closure", DevExeSpace(), scr_size, scr_level, 0, nmb1,
      ksg, keg, jsg, jeg, isg, ieg,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                    const int i) {
        if (radiation_mask_(m, k, j, i)) {
          par_for_inner(member, 0, nspecies_ - 1, [&](const int nuidx) {
            u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) = 0;
            if (nspecies_ > 1) {
              u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) = 0;
            }
            chi_(m, nuidx, k, j, i) = 0;
          });
        } else {
          // calculate metric and inverse metric
          Real garr_dd[16];
          Real garr_uu[16];
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
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
          // store normal, shift
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_d{};
          pack_n_d(adm.alpha(m, k, j, i), n_d);
          pack_beta_u(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                      adm.beta_u(m, 2, k, j, i), beta_u);
          tensor_contract(g_dd, beta_u, beta_d);
          // store Lorentz factor, four velocity, three velocity, projection
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};
          Real w_lorentz = u_mu_(m, 0, k, j, i);
          pack_u_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i),
                   u_mu_(m, 2, k, j, i), u_mu_(m, 3, k, j, i), u_u);
          tensor_contract(g_dd, u_u, u_d);
          pack_v_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i),
                   u_mu_(m, 2, k, j, i), u_mu_(m, 3, k, j, i), v_u);
          tensor_contract(g_dd, v_u, v_d);
          calc_proj(u_d, u_u, proj_ud);

          par_for_inner(member, 0, nspecies_ - 1, [&](const int nuidx) {
            const Real E = u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i);
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
            pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                     u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i),
                     F_d);
            Real chi{};
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Ptemp_dd{};
            calc_closure(closure_, g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                         chi, Ptemp_dd, params_);
            chi_(m, nuidx, k, j, i) = chi;
          });
        }
      });
  return TaskStatus::complete;
}
} // namespace radiationm1
