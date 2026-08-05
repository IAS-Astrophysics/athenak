//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_compton.cpp
//! \brief Compton terms for M1 photon transport

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation_opacities.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "units/units.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \fn RadiationM1::CalcComptonPhotons
TaskStatus RadiationM1::CalcComptonPhotons(Driver *pdrive, int stage) {

  auto *ptest_ideal = dynamic_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_ideal != nullptr) {
    return CalcComptonPhotons_<Primitive::IdealGas, Primitive::ResetFloor>(pdrive, stage);
  }

  // only ideal-gas supported
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Compton requires an ideal gas EOS!\n";
  abort();
}

//----------------------------------------------------------------------------------------
//! \fn RadiationM1::CalcComptonPhotons_
template <class EOSPolicy, class ErrorPolicy>
TaskStatus RadiationM1::CalcComptonPhotons_(Driver *pdrive, int stage) {
  assert(nspecies == 1);

  if (stage != 2) {
    return TaskStatus::complete;
  }

  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto nvars_ = nvars;
  auto &adm = pmy_pack->padm->adm;
  auto &m1_params_ = params;
  auto &radiation_mask_ = radiation_mask;

  auto &u0_ = u0;
  auto &chi_ = chi;

  DvceArray5D<Real> w0_ = w0;
  bool ismhd_ = ismhd;
  if (ismhd_) {
    w0_ = pmy_pack->pmhd->w0;
  }

  DvceArray5D<Real> umhd0_;
  if (ismhd_) {
    umhd0_ = pmy_pack->pmhd->u0;
  }
  bool backreact_ = params.backreact;

  Real dt_ = pmy_pack->pmesh->dt;

  // opacity unit scales
  Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
  Real mean_mol_weight_ = 1.0;
  Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
  if (isunits) {
    density_scale_ = pmy_pack->punit->density_cgs();
    temperature_scale_ = pmy_pack->punit->temperature_cgs();
    length_scale_ = pmy_pack->punit->length_cgs();
    mean_mol_weight_ = pmy_pack->punit->mu();
    rosseland_coef_ = pmy_pack->punit->rosseland_coef_cgs;
    planck_minus_rosseland_coef_ = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
  }
  bool power_opacity_ = photon_op_params.is_power_opacity;
  Real kappa_a_ = photon_op_params.kappa_a;
  Real kappa_s_ = photon_op_params.kappa_s;
  Real kappa_p_ = photon_op_params.kappa_p;
  Real arad_ = photon_op_params.arad;
  Real inv_t_electron_ = photon_op_params.inv_t_electron;

  Real gm1 = 0.0;
  if (ishydro) {
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  } else if (ismhd_) {
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
  }

  par_for(
      "radiation_m1_compton", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (radiation_mask_(m, k, j, i)) {
          return;
        }

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

        Real w_lorentz = get_w_lorentz(w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i),
                                       w0_(m, IVZ, k, j, i), g_dd);
        pack_u_u(w_lorentz / adm.alpha(m, k, j, i),
                 w0_(m, IVX, k, j, i) -
                     w_lorentz * adm.beta_u(m, 0, k, j, i) / adm.alpha(m, k, j, i),
                 w0_(m, IVY, k, j, i) -
                     w_lorentz * adm.beta_u(m, 1, k, j, i) / adm.alpha(m, k, j, i),
                 w0_(m, IVZ, k, j, i) -
                     w_lorentz * adm.beta_u(m, 2, k, j, i) / adm.alpha(m, k, j, i),
                 u_u);
        pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
                 adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                 adm.beta_u(m, 2, k, j, i), v_u);
        tensor_contract(g_dd, u_u, u_d);
        tensor_contract(g_dd, v_u, v_d);
        calc_proj(u_d, u_u, proj_ud);

        // reconstruct densitized comoving energy density J
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

        // undesitized J
        Real J_phys = J / volform;
        if (!(J_phys > 0.0) || arad_ <= 0.0 || inv_t_electron_ <= 0.0) {
          return;
        }

        // fluid quantities
        Real wdn = w0_(m, IDN, k, j, i);
        Real pgas = w0_(m, IPR, k, j, i);
        Real tgas = pgas / wdn;

        // scattering opacity
        Real sigma_a, sigma_s, sigma_p;
        OpacityFunction(wdn, density_scale_, tgas, temperature_scale_, length_scale_,
                        gm1, mean_mol_weight_, power_opacity_, rosseland_coef_,
                        planck_minus_rosseland_coef_, kappa_a_, kappa_s_, kappa_p_,
                        sigma_a, sigma_s, sigma_p);
        if (!(sigma_s > 0.0)) {
          return;
        }

        Real trad = Kokkos::pow(J_phys / arad_, 0.25);
        if (Kokkos::fabs(trad - tgas) < 1.0e-12) {
          return;
        }


        Real u0comp = u_u(0);
        Real dtcsigs = dt_ * sigma_s;
        Real dtaucsigs = dtcsigs / u0comp;
        Real suma1 = 4.0 * dtcsigs * inv_t_electron_;
        Real suma2 = 4.0 * dtaucsigs * inv_t_electron_ * gm1 / wdn;
        Real jr_cm = J_phys;

        Real coef1 = (1.0 + suma2 * jr_cm) / (suma1 * jr_cm) * arad_;
        Real coef0 = -(1.0 + suma2 * jr_cm) / suma1 - tgas;
        Real tradnew = trad;
        if (!FourthPolyRoot(coef1, coef0, tradnew) ||
            !Kokkos::isfinite(tradnew)) {
          return;
        }

        Real Jnew_phys = arad_ * tradnew * tradnew * tradnew * tradnew;
        Real ratio = Jnew_phys / J_phys;
        if (!Kokkos::isfinite(ratio) || ratio <= 0.0) {
          return;
        }

        Real dE = E * (ratio - 1.0);
        Real fx = u0_(m, CombinedIdx(0, M1_FX_IDX, nvars_), k, j, i);
        Real fy = u0_(m, CombinedIdx(0, M1_FY_IDX, nvars_), k, j, i);
        Real fz = u0_(m, CombinedIdx(0, M1_FZ_IDX, nvars_), k, j, i);
        Real dFx = fx * (ratio - 1.0);
        Real dFy = fy * (ratio - 1.0);
        Real dFz = fz * (ratio - 1.0);

        u0_(m, CombinedIdx(0, M1_E_IDX, nvars_), k, j, i) = E + dE;
        u0_(m, CombinedIdx(0, M1_FX_IDX, nvars_), k, j, i) = fx + dFx;
        u0_(m, CombinedIdx(0, M1_FY_IDX, nvars_), k, j, i) = fy + dFy;
        u0_(m, CombinedIdx(0, M1_FZ_IDX, nvars_), k, j, i) = fz + dFz;

        if (backreact_ && ismhd_) {
          umhd0_(m, IEN, k, j, i) -= dE;
          umhd0_(m, IM1, k, j, i) -= dFx;
          umhd0_(m, IM2, k, j, i) -= dFy;
          umhd0_(m, IM3, k, j, i) -= dFz;
        }
      });

  return TaskStatus::complete;
}

}  // namespace radiationm1
