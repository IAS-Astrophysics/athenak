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

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto nvars_ = pmy_pack->pradm1->nvars;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto &source_limiter_ = source_limiter;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = d->gam0[stage - 1];
  Real &gam1 = d->gam1[stage - 1];
  Real beta_dt = (d->beta[stage - 1]) * (pmy_pack->pmesh->dt);
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  int scr_level = 0;
  size_t scr_size = 1;

  par_for_outer(
      "radiation_m1_update", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks,
      ke, js, je, is, ie,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                    const int i) {
        // call bns_nurates in outer loop!

        // Step 1 -- compute the sources
        par_for_inner(member, 0, nspecies_ - 1, [&](const int nuidx) {
          // Source RHS are stored here
          Real DrE{};
          Real DrFx{};
          Real DrFy{};
          Real DrFz{};
          Real DrN{};
#if (THC_M1_SRC_METHOD == THC_M1_SRC_EXPL)
          // calc_rad_sources(eta_1[i4D]*volform[ijk],
          //         abs_1[i4D], scat_1[i4D], u_d, J, H_d, &S_d);
          // DrE[nuidx] = beta_dt*calc_rE_source(alp[ijk], n_u, S_d);

          // calc_rF_source(alp[ijk], gamma_ud, S_d, &tS_d);
          // DrFx[nuidx] = beta_dt*tS_d(1);
          // DrFy[nuidx] = beta_dt*tS_d(2);
          // DrFz[nuidx] = beta_dt*tS_d(3);
          // DrN[nuidx] = beta_dt*alp[ijk]*(volform[ijk]*eta_0[i4D] -
          // abs_0[i4D]*rN[i4D]/Gamma);
          DrE = 0.0;
          DrFx = 0.0;
          DrFy = 0.0;
          DrFz = 0.0;
          DrN = 0.0;
#endif // (THC_M1_SRC_METHOD == THC_M1_SRC_EXPL)
       // Step 2 -- limit the sources
          Real theta = 1.0;
          //   if (source_limiter_ >= 0) {
          //       theta = 1.0;
          //       Real DTau_sum = 0.0;
          //       par_for_inner(member, 0, nspecies_, [&](const int nuidx) {
          //           Real Estar = rE_p[i4D] + beta_dt*rE_rhs[i4D];
          //           if (DrE[nuidx] < 0) {
          //               theta =
          //               Kokkos::min(-source_limiter*Kokkos::max(Estar,
          //               0.0)/DrE[nuidx], theta);
          //           }
          //           DTau_sum -= DrE[nuidx];
          //       });
          //       member.team_barrier();
          //       if (DTau_sum < 0) {
          //           theta = Kokkos::min(-source_limiter*Kokkos::max(tau[ijk],
          //           0.0)/DTau_sum, theta);
          //       }

          //   if (nspecies > 1) {
          //       Real DDxp_sum = 0.0;
          //       par_for_inner(member, 0, nspecies_, [&](const int nuidx) {
          //           Real Nstar = u0_(m, CombinedIdx(nuidx, 4, nvars_), k, j,
          //           i)
          //           + beta_dt*rN_rhs[i4D]; if (DrN[nuidx] < 0) {
          //               theta =
          //               Kokkos::min(-source_limiter*Kokkos::max(Nstar,
          //               0.0)/DrN[nuidx], theta);
          //           }
          //           DDxp_sum += DDxp[nuidx];
          //       DTau_sum -= DrE[nuidx];
          //       });
          //       member.team_barrier();
          //       const Real DYe = DDxp_sum/dens[ijk];
          //       if (DYe > 0) {
          //           theta =
          //           Kokkos::min(source_limiter*Kokkos::max(source_Ye_max -
          //           Y_e[ijk], 0.0)/DYe, theta);
          //       }
          //       else if (DYe < 0) {
          //           theta =
          //           Kokkos::min(source_limiter*Kokkos::min(source_Ye_min -
          //           Y_e[ijk], 0.0)/DYe, theta);
          //       }
          //   }

          //       theta = Kokkos::max(0.0, theta);
          //   }
          // Step 3 -- update fields
          // compute dF1/dx1
          Real rE_rhs = (flx1(m, CombinedIdx(nuidx, 0, nvars_), k, j, i + 1) -
                         flx1(m, CombinedIdx(nuidx, 0, nvars_), k, j, i)) /
                        mbsize.d_view(m).dx1;
          Real rFx_rhs = (flx1(m, CombinedIdx(nuidx, 1, nvars_), k, j, i + 1) -
                          flx1(m, CombinedIdx(nuidx, 1, nvars_), k, j, i)) /
                         mbsize.d_view(m).dx1;
          Real rFy_rhs = (flx1(m, CombinedIdx(nuidx, 2, nvars_), k, j, i + 1) -
                          flx1(m, CombinedIdx(nuidx, 2, nvars_), k, j, i)) /
                         mbsize.d_view(m).dx1;
          Real rFz_rhs = (flx1(m, CombinedIdx(nuidx, 3, nvars_), k, j, i + 1) -
                          flx1(m, CombinedIdx(nuidx, 3, nvars_), k, j, i)) /
                         mbsize.d_view(m).dx1;
          Real rN_rhs = (flx1(m, CombinedIdx(nuidx, 4, nvars_), k, j, i + 1) -
                         flx1(m, CombinedIdx(nuidx, 4, nvars_), k, j, i)) /
                        mbsize.d_view(m).dx1;

          // Add dF2/dx2
          if (multi_d) {
            rE_rhs += (flx1(m, CombinedIdx(nuidx, 0, nvars_), k, j + 1, i) -
                       flx1(m, CombinedIdx(nuidx, 0, nvars_), k, j, i)) /
                      mbsize.d_view(m).dx1;
            rFx_rhs += (flx1(m, CombinedIdx(nuidx, 1, nvars_), k, j + 1, i) -
                        flx1(m, CombinedIdx(nuidx, 1, nvars_), k, j, i)) /
                       mbsize.d_view(m).dx1;
            rFy_rhs += (flx1(m, CombinedIdx(nuidx, 2, nvars_), k, j + 1, i) -
                        flx1(m, CombinedIdx(nuidx, 2, nvars_), k, j, i)) /
                       mbsize.d_view(m).dx1;
            rFz_rhs += (flx1(m, CombinedIdx(nuidx, 3, nvars_), k, j + 1, i) -
                        flx1(m, CombinedIdx(nuidx, 3, nvars_), k, j, i)) /
                       mbsize.d_view(m).dx1;
            rN_rhs += (flx1(m, CombinedIdx(nuidx, 4, nvars_), k, j + 1, i) -
                       flx1(m, CombinedIdx(nuidx, 4, nvars_), k, j, i)) /
                      mbsize.d_view(m).dx1;
          }

          // Add dF3/dx3
          if (three_d) {
            rE_rhs += (flx1(m, CombinedIdx(nuidx, 0, nvars_), k + 1, j, i) -
                       flx1(m, CombinedIdx(nuidx, 0, nvars_), k, j, i)) /
                      mbsize.d_view(m).dx1;
            rFx_rhs += (flx1(m, CombinedIdx(nuidx, 1, nvars_), k + 1, j, i) -
                        flx1(m, CombinedIdx(nuidx, 1, nvars_), k, j, i)) /
                       mbsize.d_view(m).dx1;
            rFy_rhs += (flx1(m, CombinedIdx(nuidx, 2, nvars_), k + 1, j, i) -
                        flx1(m, CombinedIdx(nuidx, 2, nvars_), k, j, i)) /
                       mbsize.d_view(m).dx1;
            rFz_rhs += (flx1(m, CombinedIdx(nuidx, 3, nvars_), k + 1, j, i) -
                        flx1(m, CombinedIdx(nuidx, 3, nvars_), k, j, i)) /
                       mbsize.d_view(m).dx1;
            rN_rhs += (flx1(m, CombinedIdx(nuidx, 4, nvars_), k + 1, j, i) -
                       flx1(m, CombinedIdx(nuidx, 4, nvars_), k, j, i)) /
                      mbsize.d_view(m).dx1;
          }

          // Update radiation quantities
          u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i) =
              u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i) +
              beta_dt * rE_rhs + theta * DrE;
          u0_(m, CombinedIdx(nuidx, 1, nvars_), k, j, i) =
              u0_(m, CombinedIdx(nuidx, 1, nvars_), k, j, i) +
              beta_dt * rFx_rhs + theta * DrFx;
          u0_(m, CombinedIdx(nuidx, 2, nvars_), k, j, i) =
              u0_(m, CombinedIdx(nuidx, 2, nvars_), k, j, i) +
              beta_dt * rFy_rhs + theta * DrFy;
          u0_(m, CombinedIdx(nuidx, 3, nvars_), k, j, i) =
              u0_(m, CombinedIdx(nuidx, 3, nvars_), k, j, i) +
              beta_dt * rFz_rhs + theta * DrFz;
          //   apply_floor(g_uu, &u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j,
          //   i), &F_d);

          if (nspecies_ > 1) {
            u0_(m, CombinedIdx(nuidx, 4, nvars_), k, j, i) =
                u0_(m, CombinedIdx(nuidx, 4, nvars_), k, j, i) +
                beta_dt * rN_rhs + theta * DrN;
            //   u0_(m, CombinedIdx(nuidx, 4, nvars_), k, j, i) =
            //    Kokkos::max(u0_(m, CombinedIdx(nuidx, 4, nvars_), k, j, i),
            //    rad_N_floor);
          }
        });
        member.team_barrier();
      });
  return TaskStatus::complete;
}
} // namespace radiationm1