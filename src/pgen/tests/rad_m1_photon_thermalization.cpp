//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_photon_thermalization.cpp
//  \brief Uniform single-zone thermalization test for M1 photons

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "driver/driver.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::RadiationM1PhotonThermalization
void ProblemGenerator::RadiationM1PhotonThermalization(ParameterInput *pin,
                                                       const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  auto *ptest_ideal =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_ideal != nullptr) {
    return RadiationM1PhotonThermalization_<Primitive::IdealGas,
                                            Primitive::ResetFloor>(pin, restart);
  }

  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_nqt != nullptr) {
    return RadiationM1PhotonThermalization_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                            Primitive::ResetFloor>(pin, restart);
  }

  auto *ptest_nlog =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                     Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_nlog != nullptr) {
    return RadiationM1PhotonThermalization_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                            Primitive::ResetFloor>(pin, restart);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::RadiationM1PhotonThermalization_
template <class EOSPolicy, class ErrorPolicy>
void ProblemGenerator::RadiationM1PhotonThermalization_(ParameterInput *pin,
                                                        const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "DynGRMHD needed for photon thermalization test" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The photon thermalization test requires a <radiation_m1> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->params.opacity_type != radiationm1::Photons) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The photon thermalization test requires opacity_type = photons"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int nmb1 = (pmbp->nmb_thispack - 1);
  auto &w0_ = pmbp->pmhd->w0;
  auto &uradm1_ = pmbp->pradm1->u0;
  auto &m1_nvars_ = pmbp->pradm1->nvars;

  // problem parameters
  Real rho = pin->GetReal("problem", "rho");
  Real temp = pin->GetReal("problem", "temp");
  Real erad0 = pin->GetReal("problem", "erad0");
  Real vx = pin->GetOrAddReal("problem", "vx", 0.0);
  Real vy = pin->GetOrAddReal("problem", "vy", 0.0);
  Real vz = pin->GetOrAddReal("problem", "vz", 0.0);
  Real w_lorentz = 1.0 / Kokkos::sqrt(1.0 - vx * vx - vy * vy - vz * vz);

  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr)
          ->eos.ps.GetEOSMutable();
  Real mb = eos.GetBaryonMass();
  Real nb = rho / mb;

  // initialize ADM variables to Minkowski
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  par_for("pgen_photon_therm_metric", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1), 0,
      (n1 - 1), KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
          }
        adm.psi4(m, k, j, i) = 1.;
        adm.alpha(m, k, j, i) = 1.;
      });

  // set primitive variables and radiation
  par_for(
      "pgen_photon_therm", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1), 0, (n1 - 1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real ye_ = 0.5;

        w0_(m, IDN, k, j, i) = rho;
        w0_(m, IVX, k, j, i) = vx * w_lorentz;
        w0_(m, IVY, k, j, i) = vy * w_lorentz;
        w0_(m, IVZ, k, j, i) = vz * w_lorentz;
        w0_(m, IPR, k, j, i) = eos.GetPressure(nb, temp, &ye_);

        uradm1_(m, radiationm1::CombinedIdx(0, M1_E_IDX, m1_nvars_), k, j, i) = erad0;
        uradm1_(m, radiationm1::CombinedIdx(0, M1_FX_IDX, m1_nvars_), k, j, i) = 0.;
        uradm1_(m, radiationm1::CombinedIdx(0, M1_FY_IDX, m1_nvars_), k, j, i) = 0.;
        uradm1_(m, radiationm1::CombinedIdx(0, M1_FZ_IDX, m1_nvars_), k, j, i) = 0.;
      });

  pmbp->pdyngr->PrimToConInit(0, (n1 - 1), 0, (n2 - 1), 0, (n3 - 1));
}
