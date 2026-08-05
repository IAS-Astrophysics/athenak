//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_opacities_photons.cpp
//! \brief calculate photon opacities for grey M1

#include <type_traits>

#include "athena.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
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

  auto *ptest_ideal = dynamic_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_ideal != nullptr) {
    return CalcOpacityPhotons_<Primitive::IdealGas, Primitive::ResetFloor>(pdrive, stage);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

template <class EOSPolicy, class ErrorPolicy>
TaskStatus RadiationM1::CalcOpacityPhotons_(Driver *pdrive, int stage) {
  assert(nspecies == 1);

  constexpr bool is_ideal_eos = std::is_same_v<EOSPolicy, Primitive::IdealGas>;

  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto nmb1 = pmy_pack->nmb_thispack - 1;

  auto &eta_1_ = eta_1;
  auto &abs_1_ = abs_1;
  auto &scat_1_ = scat_1;

  DvceArray5D<Real> w0_ = w0;
  if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
  }

  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.GetBaryonMass();

  // conversion from code temperature to cgs for tabulated EOS
  auto code_units = eos.GetCodeUnitSystem();
  const Real temp_code_to_cgs = code_units.TemperatureConversion(Primitive::MakeCGS());

  // Extract radiation constant and units
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

  if constexpr (!is_ideal_eos) {
    // Check for units with tabulated EOS
    if (!isunits || photon_op_params.is_power_opacity || photon_op_params.is_compton) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl;
      std::cout << "Photon opacities with tabulated EOS require units and do not support power_opacity or compton\n";
      abort();
    }
    Real length_code_cgs = code_units.LengthConversion(Primitive::MakeCGS());
    Real density_code_cgs = code_units.MassDensityConversion(Primitive::MakeCGS());
    if (fabs(length_scale_ / length_code_cgs - 1.0) > 1.0e-3 || fabs(density_scale_ / density_code_cgs - 1.0) > 1.0e-3) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl;
      std::cout << "<units> block is inconsistent with the EOS code units\n";
      abort();
    }
  }

  bool power_opacity_ = photon_op_params.is_power_opacity;
  Real kappa_a_ = photon_op_params.kappa_a;
  Real kappa_s_ = photon_op_params.kappa_s;
  Real kappa_p_ = photon_op_params.kappa_p;
  Real arad_ = photon_op_params.arad;

  if (power_opacity_ && !isunits) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "power_opacity requires a <units> block\n";
    abort();
  }

  Real gm1{};
  if (ishydro) {
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  } else if (ismhd) {
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
  }

  auto & radiation_mask_ = radiation_mask;

  par_for(
      "radiation_m1_calc_opacity_photons", DevExeSpace(), 0, nmb1, ks, ke, js,
      je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (radiation_mask_(m, k, j, i)) {
          abs_1_(m, 0, k, j, i) = 0;
          eta_1_(m, 0, k, j, i) = 0;
          scat_1_(m, 0, k, j, i) = 0;
        } else {
          Real wdn = w0_(m, IDN, k, j, i);
          Real pgas = w0_(m, IPR, k, j, i);

          Real tgas;
          if constexpr (is_ideal_eos) {
            tgas = pgas / wdn;
          } else {
            Real nb = wdn / mb;
            Real Y = w0_(m, IYF, k, j, i);
            tgas = eos.GetTemperatureFromP(nb, pgas, &Y) * temp_code_to_cgs / temperature_scale_;
          }

          // set photon opacities
          Real sigma_a, sigma_s, sigma_p;
          OpacityFunction(wdn, density_scale_, tgas, temperature_scale_,
                          length_scale_, gm1, mean_mol_weight_, power_opacity_,
                          rosseland_coef_, planck_minus_rosseland_coef_,
                          kappa_a_, kappa_s_, kappa_p_, sigma_a, sigma_s,
                          sigma_p);

          // compute opacities from sigma_a, sigma_s, sigma_p
          Real eta_1_loc = (sigma_a + sigma_p) * arad_ * (tgas * tgas * tgas * tgas);
          Real abs_1_loc = sigma_a + sigma_p;
          Real scat_1_loc = sigma_s - sigma_p;

          assert(Kokkos::isfinite(eta_1_loc));
          assert(Kokkos::isfinite(abs_1_loc));
          assert(Kokkos::isfinite(scat_1_loc));

          eta_1_(m, 0, k, j, i) = eta_1_loc;
          abs_1_(m, 0, k, j, i) = abs_1_loc;
          scat_1_(m, 0, k, j, i) = scat_1_loc;
        }
      });

  return TaskStatus::complete;
}
} // namespace radiationm1
