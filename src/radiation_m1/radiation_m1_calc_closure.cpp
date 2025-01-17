//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_closure.cpp
//! \brief calculate lab frame pressure

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_closure.hpp"

namespace radiationm1 {
TaskStatus RadiationM1::CalcClosure(Driver *pdrive, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto &u0_ = pmy_pack->pradm1->u0;
  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  auto &P_dd_ = pmy_pack->pradm1->P_dd;
  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &nvars_ = pmy_pack->pradm1->nvars;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto &radiation_mask_ = pmy_pack->pradm1->radiation_mask;

  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;
  RadiationM1Params &params_ = pmy_pack->pradm1->params;

  // index limits with ghost
  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;

  size_t scr_size = ScrArray2D<Real>::shmem_size(4, 4) * 4 +
                    ScrArray1D<Real>::shmem_size(4) * 4;
  int scr_level = 0;
  par_for_outer(
      "radiation_m1_calc_closure", DevExeSpace(), scr_size, scr_level, 0, nmb1,
      ksg, keg, jsg, jeg, isg, ieg,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                    const int i) {
        if (radiation_mask_(m, k, j, i)) {
          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, 1, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, 2, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, 3, nvars_), k, j, i) = 0;
            if (nspecies_ > 1) {
              u0_(m, CombinedIdx(nuidx, 4, nvars_), k, j, i) = 0;
            }
            P_dd_(m, CombinedIdx(nuidx, 0, 6), k, j, i) = 0;
            P_dd_(m, CombinedIdx(nuidx, 1, 6), k, j, i) = 0;
            P_dd_(m, CombinedIdx(nuidx, 2, 6), k, j, i) = 0;
            P_dd_(m, CombinedIdx(nuidx, 3, 6), k, j, i) = 0;
            P_dd_(m, CombinedIdx(nuidx, 4, 6), k, j, i) = 0;
            P_dd_(m, CombinedIdx(nuidx, 5, 6), k, j, i) = 0;
          }
        } else {
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
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
          n_d(0) = -adm.alpha(m, k, j, i);
          n_d(1) = 0;
          n_d(2) = 0;
          n_d(3) = 0;

          Real w_lorentz = u_mu_(m, 0, k, j, i);
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> proj_ud{};
          u_u(0) = u_mu_(m, 0, k, j, i);
          u_u(1) = u_mu_(m, 1, k, j, i);
          u_u(2) = u_mu_(m, 2, k, j, i);
          u_u(3) = u_mu_(m, 3, k, j, i);
          tensor_contract(g_dd, u_u, u_d);
          calc_proj(u_d, u_u, proj_ud);

          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
          v_u(0) = u_u(0) / w_lorentz;
          v_u(1) = u_u(1) / w_lorentz;
          v_u(2) = u_u(2) / w_lorentz;
          v_u(3) = u_u(3) / w_lorentz;
          tensor_contract(g_dd, v_u, v_d);

          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};

          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            Real E = u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i);
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
            F_d(0) = 0; // @TODO: checkme!
            F_d(1) = u0_(m, CombinedIdx(nuidx, 1, nvars_), k, j, i);
            F_d(2) = u0_(m, CombinedIdx(nuidx, 2, nvars_), k, j, i);
            F_d(3) = u0_(m, CombinedIdx(nuidx, 3, nvars_), k, j, i);
            Real chi{};
            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Ptemp_dd{};

            calc_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                         chi, Ptemp_dd, params_);

            P_dd_(m, CombinedIdx(nuidx, 0, 6), k, j, i) = Ptemp_dd(0, 0); // Pxx
            P_dd_(m, CombinedIdx(nuidx, 1, 6), k, j, i) = Ptemp_dd(0, 1); // Pxy
            P_dd_(m, CombinedIdx(nuidx, 2, 6), k, j, i) = Ptemp_dd(0, 2); // Pxz
            P_dd_(m, CombinedIdx(nuidx, 3, 6), k, j, i) = Ptemp_dd(1, 1); // Pyy
            P_dd_(m, CombinedIdx(nuidx, 4, 6), k, j, i) = Ptemp_dd(1, 2); // Pyz
            P_dd_(m, CombinedIdx(nuidx, 5, 6), k, j, i) = Ptemp_dd(2, 2); // Pzz
          }
        }
      });
  return TaskStatus::complete;
}
} // namespace radiationm1
