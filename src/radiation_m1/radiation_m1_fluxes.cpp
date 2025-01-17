//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_fluxes.cpp
//! \brief calculate 3D fluxes for M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_closure.hpp"

namespace radiationm1 {
TaskStatus RadiationM1::Fluxes(Driver *pdrive, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto nvarstotm1 = pmy_pack->pradm1->nvarstot - 1;
  auto nvars_ = pmy_pack->pradm1->nvars;
  auto &params_ = pmy_pack->pradm1->params;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  auto &u0_ = pmy_pack->pradm1->u0;
  auto &P_dd_ = pmy_pack->pradm1->P_dd;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = 2;
  int scr_level = 0;
  auto &flx1_ = uflx.x1f;

  int il = is, iu = ie + 1, jl = js, ju = je, kl = ks, ku = ke;

  par_for_outer(
      "radiation_m1_flux_x1", DevExeSpace(), scr_size, scr_level, 0, nmb1, kl,
      ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                    const int i) {
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_uu;
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu;
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_u;
        beta_u(0) = 0;
        beta_u(1) = adm.beta_u(m, 0, k, j, i);
        beta_u(2) = adm.beta_u(m, 1, k, j, i);
        beta_u(3) = adm.beta_u(m, 2, k, j, i);
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u;
        u_u(0) = u_mu_(m, 0, k, j, i);
        u_u(1) = u_mu_(m, 1, k, j, i);
        u_u(2) = u_mu_(m, 2, k, j, i);
        u_u(3) = u_mu_(m, 3, k, j, i);
        Real lorentz_w = u_u(0);
        v_u(0) = 0;
        v_u(1) = u_u(1) / lorentz_w;
        v_u(2) = u_u(2) / lorentz_w;
        v_u(3) = u_u(3) / lorentz_w;

        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u;
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd;
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> P_ud;
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> fnu_u;

        for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
          P_dd(0, 0) = 0;
          P_dd(0, 1) = adm.beta_u(m, 0, k, j, i);
          P_dd(0, 2) = adm.beta_u(m, 1, k, j, i);
          P_dd(0, 3) = adm.beta_u(m, 2, k, j, i);
          P_dd(1, 1) = P_dd_(m, CombinedIdx(nuidx, 0, 6), k, j, i);
          P_dd(1, 2) = P_dd_(m, CombinedIdx(nuidx, 1, 6), k, j, i);
          P_dd(1, 3) = P_dd_(m, CombinedIdx(nuidx, 2, 6), k, j, i);
          P_dd(2, 2) = P_dd_(m, CombinedIdx(nuidx, 3, 6), k, j, i);
          P_dd(2, 3) = P_dd_(m, CombinedIdx(nuidx, 4, 6), k, j, i);
          P_dd(3, 3) = P_dd_(m, CombinedIdx(nuidx, 5, 6), k, j, i);

          Real J = 0; //@TODO
          Real E = u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i);
          Real N = u0_(m, CombinedIdx(nuidx, 4, nvars_), k, j, i);
          assemble_fnu(u_u, J, H_u, fnu_u, params_);
          const Real Gamma = compute_Gamma(lorentz_w, v_u, J, E, F_d, params_);

          // Note that nnu is densitized here
          Real nnu{};
          if (nspecies_ > 1) {
            nnu = N / Gamma;
          }

          Real flux[4];
          int dir = 0;
          flux[0] =
              calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir + 1, 1);
          flux[1] =
              calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir + 1, 2);
          flux[2] =
              calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir + 1, 3);
          if (nspecies_ > 1) {
            flux[3] = adm.alpha(m, k, j, i) * nnu * fnu_u(dir + 1);
          }

          // Speed of light -- note that gamma_uu has NDIM=3
          const Real clam[2] = {
              adm.alpha(m, k, j, i) * std::sqrt(gamma_uu(dir, dir)) -
                  beta_u(dir + 1),
              -adm.alpha(m, k, j, i) * std::sqrt(gamma_uu(dir, dir)) -
                  beta_u(dir + 1)};
          const Real clight =
              Kokkos::max(Kokkos::abs(clam[0]), Kokkos::abs(clam[1]));

          // In some cases the eigenvalues in the thin limit
          // overestimate the actual characteristic speed and can
          // become larger than c
          Real cmax = clight;

          // Remove dissipation at high Peclet numbers
          /*
          const Real kapa = 0.5 * (abs_1[i4D] + abs_1[i4Dp] + scat_1[i4D] + scat_1[i4Dp]);
          Real A = 1.0;
          if (kapa * delta[dir] > 1.0) {
            A = Kokkos::min(1.0, 1.0 / (delta[dir] * kapa));
            A = max(A, mindiss);
          }

          CCTK_REAL const ujm = cons[GFINDEX1D(__k - 1, ig, iv)];
          CCTK_REAL const uj = cons[GFINDEX1D(__k, ig, iv)];
          CCTK_REAL const ujp = cons[GFINDEX1D(__k + 1, ig, iv)];
          CCTK_REAL const ujpp = cons[GFINDEX1D(__k + 2, ig, iv)];

          CCTK_REAL const fj = flux[GFINDEX1D(__k, ig, iv)];
          CCTK_REAL const fjp = flux[GFINDEX1D(__k + 1, ig, iv)];

          CCTK_REAL const cc = cmax[GFINDEX1D(__k, ig, 0)];
          CCTK_REAL const ccp = cmax[GFINDEX1D(__k + 1, ig, 0)];
          CCTK_REAL const cmx = std::max(cc, ccp);

          CCTK_REAL const dup = ujpp - ujp;
          CCTK_REAL const duc = ujp - uj;
          CCTK_REAL const dum = uj - ujm;

          bool sawtooth = false;
          CCTK_REAL phi = 0;
          if (dup * duc > 0 && dum * duc > 0) {
            phi = minmod2(dum / duc, dup / duc, minmod_theta);
          } else if (dup * duc < 0 && dum * duc < 0) {
            sawtooth = true;
          }
          assert(isfinite(phi));

          CCTK_REAL const flux_low = 0.5 * (fj + fjp - cmx * (ujp - uj));
          CCTK_REAL const flux_high = 0.5 * (fj + fjp);

          CCTK_REAL flux_num = flux_high - (sawtooth ? 1 : A) * (1 - phi) *
                                               (flux_high - flux_low);
          flux_jp[PINDEX1D(ig, iv)] = flux_num; */
        }
      });
  return TaskStatus::complete;
}
} // namespace radiationm1
