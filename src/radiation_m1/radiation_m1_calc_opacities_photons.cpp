//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_opacities_photons.cpp
//! \brief calculate photon opacities for grey M1

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "hydro/hydro.hpp"
#include "radiation/radiation_opacities.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "units/units.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::CalcOpacityPhotons(Driver *pdrive, int stage) {
  // The opacities are constant throughout a timestep
  if (stage > 1) {
    return TaskStatus::complete;
  }

  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(
          pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    return CalcOpacityPhotons_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
      pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    return CalcOpacityPhotons_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

template <class EOSPolicy, class ErrorPolicy>
TaskStatus RadiationM1::CalcOpacityPhotons_(Driver *pdrive, int stage) {
  assert(nspecies == 1);

  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nspecies_ = nspecies;
  auto nvars_ = nvars;

  auto &adm = pmy_pack->padm->adm;
  auto &radiation_mask_ = radiation_mask;

  auto &m1_params_ = params;

  auto &eta_1_ = eta_1;
  auto &abs_1_ = abs_1;
  auto &scat_1_ = scat_1;

  auto &u0_ = u0;
  auto &w0_ = pmy_pack->pmhd->w0;
  auto &chi_ = chi;

  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  if ((m1_params_.opacity_one_dt) && (stage != 2)) {
    // Keep opacities constant throught the timestep
    return TaskStatus::complete;
  }

  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.GetBaryonMass();

  // conversion factors from cgs to code units
  auto code_units = eos.GetCodeUnitSystem();
  auto eos_units = eos.GetEOSUnitSystem();

  // Extract radiation constant and units
  Real &arad_ = photon_op_params.arad;
  Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
  Real mean_mol_weight_ = 1.0;
  Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
  if (isunits) {
    density_scale_ = pmy_pack->punit->density_cgs();
    temperature_scale_ = pmy_pack->punit->temperature_cgs();
    length_scale_ = pmy_pack->punit->length_cgs();
    mean_mol_weight_ = pmy_pack->punit->mu();
    rosseland_coef_ = pmy_pack->punit->rosseland_coef_cgs;
    planck_minus_rosseland_coef_ =
        pmy_pack->punit->planck_minus_rosseland_coef_cgs;
  }

  bool power_opacity_ = photon_op_params.is_power_opacity;
  Real kappa_a_ = photon_op_params.kappa_a;
  Real kappa_s_ = photon_op_params.kappa_s;
  Real kappa_p_ = photon_op_params.kappa_p;

  Real gm1{};
  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  if (ishydro) {
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  } else if (ismhd) {
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
  }

  par_for(
      "radiation_m1_calc_opacity_photons", DevExeSpace(), 0, nmb1, ks, ke, js,
      je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (radiation_mask(m, k, j, i)) {
          abs_1_(m, 0, k, j, i) = 0;
          eta_1_(m, 0, k, j, i) = 0;
          scat_1_(m, 0, k, j, i) = 0;
        } else {
          Real garr_dd[16];
          Real garr_uu[16];
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
          pack_n_d(adm.alpha(m, k, j, i), n_d);
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

          Real gam = adm::SpatialDet(
              adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
              adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
              adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
          Real volform = Kokkos::sqrt(gam);

          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};

          Real w_lorentz{};
          if (use_u_mu_data) {
            w_lorentz = adm.alpha(m, k, j, i) * u_mu_(m, 0, k, j, i);
            pack_u_u(u_mu_(m, 0, k, j, i), u_mu_(m, 1, k, j, i),
                     u_mu_(m, 2, k, j, i), u_mu_(m, 3, k, j, i), u_u);
          } else {
            w_lorentz =
                get_w_lorentz(w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i),
                              w0_(m, IVZ, k, j, i), g_dd);
            pack_u_u(
                w_lorentz / adm.alpha(m, k, j, i),
                w0_(m, IVX, k, j, i) - w_lorentz * adm.beta_u(m, 0, k, j, i) /
                                           adm.alpha(m, k, j, i),
                w0_(m, IVY, k, j, i) - w_lorentz * adm.beta_u(m, 1, k, j, i) /
                                           adm.alpha(m, k, j, i),
                w0_(m, IVZ, k, j, i) - w_lorentz * adm.beta_u(m, 2, k, j, i) /
                                           adm.alpha(m, k, j, i),
                u_u);
          }
          pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
                   adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i), v_u);
          tensor_contract(g_dd, u_u, u_d);
          tensor_contract(g_dd, v_u, v_d);
          calc_proj(u_d, u_u, proj_ud);

          // Compute lab frame energy density and number density
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i),
                   u0_(m, CombinedIdx(0, M1_FX_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(0, M1_FY_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(0, M1_FZ_IDX, nvars_), k, j, i), F_d);
          const Real E = u0_(m, CombinedIdx(0, M1_E_IDX, nvars_), k, j, i);
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
          apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                        chi_(m, 0, k, j, i), P_dd, m1_params_);

          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
          assemble_rT(n_d, E, F_d, P_dd, T_dd);

          Real J = calc_J_from_rT(T_dd, u_u);

          // fluid quantities
          Real &wdn = w0_(m, IDN, k, j, i);
          Real &wen = w0_(m, IEN, k, j, i);

          // derived quantities
          Real pgas = gm1 * wen;
          Real tgas = pgas / wdn;

          // local undensitized photon quantities
          Real nudens_1 = J / volform;
          Real chi_loc = chi_(m, 0, k, j, i);

          // get emissivities and opacities
          Real eta_1_loc{}, abs_1_loc{}, scat_1_loc{};

          // set photon opacities
          Real sigma_a, sigma_s, sigma_p;
          OpacityFunction(wdn, density_scale_, tgas, temperature_scale_,
                          length_scale_, gm1, mean_mol_weight_, power_opacity_,
                          rosseland_coef_, planck_minus_rosseland_coef_,
                          kappa_a_, kappa_s_, kappa_p_, sigma_a, sigma_s,
                          sigma_p);

          // compute opacities from sigma_a, sigma_s, sigma_p
          eta_1_loc = sigma_p;
          abs_1_loc = sigma_p;
          scat_1_loc = sigma_s + sigma_a;

          assert(Kokkos::isfinite(eta_1_loc));
          assert(Kokkos::isfinite(abs_1_loc));
          assert(Kokkos::isfinite(scat_1_loc));

          eta_1(m, 0, k, j, i) = eta_1_loc;
          abs_1(m, 0, k, j, i) = abs_1_loc;
          scat_1(m, 0, k, j, i) = scat_1_loc;
        }
      });

  return TaskStatus::complete;
}
} // namespace radiationm1
