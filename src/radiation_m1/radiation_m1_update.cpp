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
#include "radiation_m1_closure.hpp"

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
  auto &u1_ = u1;
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
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  int scr_level = 0;
  size_t scr_size = 1;

  par_for_outer(
      "radiation_m1_update", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks,
      ke, js, je, is, ie,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                    const int i) {
        // metric and other quantities
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

        // Step 1 -- compute the sources
        par_for_inner(member, 0, nspecies_ - 1, [&](const int nuidx) {
          // Source RHS are stored here
          Real DrEFN[5]{};
          // #if (THC_M1_SRC_METHOD == THC_M1_SRC_EXPL)
          //  add sources later
          for (int var = 0; var < nspecies_; ++var) {
            DrEFN[var] = 0;
          }
          // #endif // (THC_M1_SRC_METHOD == THC_M1_SRC_EXPL)
          //  Step 2 -- limit the sources
          Real theta = 1.0;

          // Step 3 -- update fields
          // compute dF1/dx1
          Real rEFN[5]{};
          for (int var = 0; var < nspecies_; ++var) {
            rEFN[var] =
                (flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i) -
                 flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i - 1)) /
                mbsize.d_view(m).dx1;

            if (multi_d) {
              rEFN[var] +=
                  (flx2(m, CombinedIdx(nuidx, var, nvars_), k, j, i) -
                   flx2(m, CombinedIdx(nuidx, var, nvars_), k, j - 1, i)) /
                  mbsize.d_view(m).dx2;
            }

            if (three_d) {
              rEFN[var] +=
                  (flx3(m, CombinedIdx(nuidx, var, nvars_), k, j, i) -
                   flx3(m, CombinedIdx(nuidx, var, nvars_), k - 1, j, i)) /
                  mbsize.d_view(m).dx3;
            }
          }

          // Update radiation quantities
          Real E = u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) -
                   beta_dt * rEFN[M1_E_IDX] + theta * DrEFN[M1_E_IDX];
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          Real Fx = u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) -
                    beta_dt * rEFN[M1_FX_IDX] + theta * DrEFN[M1_FX_IDX];
          Real Fy = u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) -
                    beta_dt * rEFN[M1_FY_IDX] + theta * DrEFN[M1_FY_IDX];
          Real Fz = u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) -
                    beta_dt * rEFN[M1_FZ_IDX] + theta * DrEFN[M1_FZ_IDX];
          pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i), Fx, Fy, Fz, F_d);
          apply_floor(g_uu, E, F_d, params_);
          u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) = E;
          u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) = Fx;
          u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) = Fy;
          u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) = Fz;

          if (nspecies_ > 1) {
            Real N = u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) +
                     beta_dt * rEFN[M1_N_IDX] + theta * DrEFN[M1_N_IDX];
            N = Kokkos::max(N, params_.rad_N_floor);
            u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) = N;
          }
        });
      });
  return TaskStatus::complete;
}
} // namespace radiationm1