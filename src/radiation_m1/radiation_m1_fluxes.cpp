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

KOKKOS_INLINE_FUNCTION
void CalcFlux(const int m, const int k, const int j, const int i,
              const int nuidx, const int dir, const DvceArray5D<Real> &u0_,
              const DvceArray5D<Real> &P_dd_,
              const AthenaTensor<Real, TensorSymm::NONE, 4, 1> &u_mu_,
              const adm::ADM::ADM_vars &adm, const RadiationM1Params params_,
              const int nvars_, const int nspecies_, Real flux[5], Real &cmax) {
  // load 4-metric, 3-metric inverse, shift, normal
  Real garr_dd[16];
  Real garr_uu[16];
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_uu{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
  pack_n_d(adm.alpha(m, k, j, i), n_d);
  adm::SpacetimeMetric(adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
                       adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                       adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                       adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                       adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
                       garr_dd);
  adm::SpacetimeUpperMetric(
      adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
      adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
      adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
      adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
      adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_uu);
  pack_beta_u(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
              adm.beta_u(m, 2, k, j, i), beta_u);
  tensor_contract(g_dd, beta_u, beta_d);
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      g_dd(a, b) = garr_dd[a + b * 4];
      g_uu(a, b) = garr_uu[a + b * 4];
    }
  }
  for (int a = 1; a < 4; ++a) {
    for (int b = 1; b < 4; ++b) {
      gamma_uu(a - 1, b - 1) = garr_uu[a + b * 4] - beta_u(a) * beta_u(b) /
                                                        (adm.alpha(m, k, j, i) *
                                                         adm.alpha(m, k, j, i));
    }
  }

  // load Lorentz factor, four velocity, three velocity, projection
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> proj_ud{};
  Real w_lorentz = u_mu_(m, 0, k, j, i);
  pack_u_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i), u_mu_(m, 2, k, j, i),
           u_mu_(m, 3, k, j, i), u_u);
  tensor_contract(g_dd, u_u, u_d);
  pack_v_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i), u_mu_(m, 2, k, j, i),
           u_mu_(m, 3, k, j, i), v_u);
  tensor_contract(g_dd, v_u, v_d);
  calc_proj(u_d, u_u, proj_ud);

  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u{};
  pack_F_d(beta_d(1), beta_d(2), beta_d(3),
           u0_(m, CombinedIdx(nuidx, 1, nvars_), k, j, i),
           u0_(m, CombinedIdx(nuidx, 2, nvars_), k, j, i),
           u0_(m, CombinedIdx(nuidx, 1, nvars_), k, j, i), F_d);
  tensor_contract(g_uu, F_d, F_u);

  // lab frame pressure
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> P_ud{};
  pack_P_dd(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
            adm.beta_u(m, 2, k, j, i),
            P_dd_(m, CombinedIdx(nuidx, 0, 6), k, j, i),
            P_dd_(m, CombinedIdx(nuidx, 1, 6), k, j, i),
            P_dd_(m, CombinedIdx(nuidx, 2, 6), k, j, i),
            P_dd_(m, CombinedIdx(nuidx, 3, 6), k, j, i),
            P_dd_(m, CombinedIdx(nuidx, 4, 6), k, j, i),
            P_dd_(m, CombinedIdx(nuidx, 5, 6), k, j, i), P_dd);
  tensor_contract(g_uu, P_dd, P_ud);
  const Real E = u0_(m, CombinedIdx(nuidx, 0, nvars_), k, j, i);

  // compute fluid frame quantities
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
  assemble_rT(n_d, E, F_d, P_dd, T_dd);
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> fnu_u{};

  Real J = calc_J_from_rT(T_dd, u_u);
  calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
  tensor_contract(g_uu, H_d, H_u);

  Real N = u0_(m, CombinedIdx(nuidx, 4, nvars_), k, j, i);
  assemble_fnu(u_u, J, H_u, fnu_u, params_);
  const Real Gamma = compute_Gamma(w_lorentz, v_u, J, E, F_d, params_);

  // Note that nnu is densitized here
  Real nnu{};
  if (nspecies_ > 1) {
    nnu = N / Gamma;
  }

  flux[0] = calc_E_flux(adm.alpha(m, k, j, i), beta_u, E, F_u, dir + 1);
  flux[1] = calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir + 1, 1);
  flux[2] = calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir + 1, 2);
  flux[3] = calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir + 1, 3);
  if (nspecies_ > 1) {
    flux[4] = adm.alpha(m, k, j, i) * nnu * fnu_u(dir + 1);
  } else {
    flux[4] = 0;
  }

  // Speed of light -- note that gamma_uu has NDIM=3
  const Real clam[2] = {
      adm.alpha(m, k, j, i) * std::sqrt(gamma_uu(dir, dir)) - beta_u(dir + 1),
      -adm.alpha(m, k, j, i) * std::sqrt(gamma_uu(dir, dir)) - beta_u(dir + 1)};
  const Real clight = Kokkos::max(Kokkos::abs(clam[0]), Kokkos::abs(clam[1]));
  cmax = clight;
}

TaskStatus RadiationM1::Fluxes(Driver *pdrive, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

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
  auto &P_dd_ = pmy_pack->pradm1->P_dd;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = 1;
  int scr_level = 0;
  auto &flx1_ = uflx.x1f;

  int il = is - 1, iu = ie, jl = js, ju = je, kl = ks, ku = ke;

  par_for(
      "radiation_m1_flux_x1", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu, 0,
      nspecies_ - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    const int nuidx) {
        int dir = 1;
        Real flux_i[5]{};
        Real flux_ip1[5]{};
        Real cmax_i, cmax_ip1;
        CalcFlux(m, k, j, i, nuidx, dir, u0_, P_dd_, u_mu_, adm, params_,
                 nvars_, nspecies_, flux_i, cmax_i);
        CalcFlux(m, k, j, i + 1, nuidx, dir, u0_, P_dd_, u_mu_, adm, params_,
                 nvars_, nspecies_, flux_ip1, cmax_ip1);

        Real flux_ip12_lo[5]{};
        Real flux_ip12_ho[5]{};

        Real cmax_ip12 = Kokkos::max(cmax_i, cmax_ip1);
        Real kappa_ave = 0; //@TODO: fix this!
        Real A_ip12 = Kokkos::min(1., 1. / mbsize.d_view(m).dx1);

        for (int var = 0; var < nvars_; ++var) {
          const Real ujm = (i - 1) < 0 ? 0
                                       : u0_(m, CombinedIdx(nuidx, var, nvars_),
                                             k, j, i - 1);
          const Real uj = u0_(m, CombinedIdx(nuidx, var, nvars_), k, j, i);
          const Real ujp = u0_(m, CombinedIdx(nuidx, var, nvars_), k, j, i + 1);
          const Real ujpp =
              (i + 2) > ie
                  ? 0
                  : u0_(m, CombinedIdx(nuidx, var, nvars_), k, j, i + 2);

          const Real dup = ujpp - ujp;
          const Real duc = ujp - uj;
          const Real dum = uj - ujm;

          bool sawtooth = false;
          Real phi_ip12 = 0;
          if (dup * duc > 0 && dum * duc > 0) {
            phi_ip12 = minmod2(dum / duc, dup / duc, params_.minmod_theta);
          } else if (dup * duc < 0 && dum * duc < 0) {
            sawtooth = true;
          }

          flux_ip12_lo[var] = (flux_i[var] + flux_ip1[var]) / 2.;
          flux_ip12_ho[var] =
              (flux_i[var] + flux_ip1[var]) / 2. -
              cmax_ip12 *
                  (u0_(m, CombinedIdx(nuidx, var, nvars_), k, j, i + 1) -
                   u0_(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) /
                  2.;

          flx1_(m, var, k, j, i) =
              flux_ip12_ho[var] - (sawtooth ? 1 : A_ip12) * (1 - phi_ip12) *
                                      (flux_ip12_ho[var] - flux_ip12_lo[var]);
        }
      });
  return TaskStatus::complete;
}
} // namespace radiationm1
