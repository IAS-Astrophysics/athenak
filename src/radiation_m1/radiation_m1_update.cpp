//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================


#include "athena.hpp"
#include "globals.hpp"
#include "athena_tensor.hpp"
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
TaskStatus RadiationM1::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &mbsize  = pmy_pack->pmb->mb_size;
  auto &nspecies_ = nspecies;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("radiation_m1_update",DevExeSpace(),scr_size,scr_level,0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j, const int i) {
      // Step 1 -- compute the sources
      par_for_inner(member, 0, nspecies_, [&](const int n) {
#if (THC_M1_SRC_METHOD == THC_M1_SRC_EXPL)
        calc_rad_sources(eta_1[i4D]*volform[ijk],
                abs_1[i4D], scat_1[i4D], u_d, J, H_d, &S_d);
        DrE[ig] = dt*calc_rE_source(alp[ijk], n_u, S_d);

        calc_rF_source(alp[ijk], gamma_ud, S_d, &tS_d);
        DrFx[ig] = dt*tS_d(1);
        DrFy[ig] = dt*tS_d(2);
        DrFz[ig] = dt*tS_d(3);
        DrN[ig] = dt*alp[ijk]*(volform[ijk]*eta_0[i4D] - abs_0[i4D]*rN[i4D]/Gamma);
      });
      member.team_barrier();
      // Step 2 -- limit the sources
      Real theta = 1.0;
      if (source_limiter >= 0) {
          theta = 1.0;
          Real DTau_sum = 0.0;
          par_for_inner(member, 0, nspecies_, [&](const int n) {
              REAL Estar = rE_p[i4D] + dt*rE_rhs[i4D];
              if (DrE[ig] < 0) {
                  theta = min(-source_limiter*max(Estar, 0.0)/DrE[ig], theta);
              }
              DTau_sum -= DrE[ig];
          });
          member.team_barrier();
          if (DTau_sum < 0) {
              theta = min(-source_limiter*max(tau[ijk], 0.0)/DTau_sum, theta);
          }

          if (nspecies > 1) {
              Real DDxp_sum = 0.0;
              par_for_inner(member, 0, nspecies_, [&](const int n) {
                  Real Nstar = rN_p[i4D] + dt*rN_rhs[i4D];
                  if (DrN[ig] < 0) {
                      theta = min(-source_limiter*max(Nstar, 0.0)/DrN[ig], theta);
                  }
                  DDxp_sum += DDxp[ig];
              DTau_sum -= DrE[ig];
              });
              member.team_barrier();
              const Real DYe = DDxp_sum/dens[ijk];
              if (DYe > 0) {
                  theta = min(source_limiter*max(source_Ye_max - Y_e[ijk], 0.0)/DYe, theta);
              }
              else if (DYe < 0) {
                  theta = min(source_limiter*min(source_Ye_min - Y_e[ijk], 0.0)/DYe, theta);
              }
          }

          theta = max(0.0, theta);
      }
      // Step 3 -- update fields
      par_for_inner(member, 0, nspecies_, [&](const int n) {

          // Update radiation quantities
          Real E =  rE_p[i4D] + dt*rE_rhs[i4D]  + theta*DrE[ig];
          F_d(1)      = rFx_p[i4D] + dt*rFx_rhs[i4D] + theta*DrFx[ig];
          F_d(2)      = rFy_p[i4D] + dt*rFy_rhs[i4D] + theta*DrFy[ig];
          F_d(3)      = rFz_p[i4D] + dt*rFz_rhs[i4D] + theta*DrFz[ig];
          apply_floor(g_uu, &E, &F_d);

          Real N = 0; (void)N;
          if (nspecies > 1) {
              N = rN_p[i4D] + dt*rN_rhs[i4D]  + theta*DrN[ig];
              N = max(N, rad_N_floor);
          }
          rE[i4D]  = E;
          unpack_F_d(F_d, &rFx[i4D], &rFy[i4D], &rFz[i4D]);
          if (nspecies > 1) rN[i4D] = N;

      });
      member.team_barrier();
  });
  return TaskStatus::complete;
}
}