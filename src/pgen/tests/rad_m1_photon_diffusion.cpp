//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_photon_diffusion.cpp
//  \brief photon diffusion test for the grey-M1 photon opacities

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::RadiationM1PhotonDiffusion
void ProblemGenerator::RadiationM1PhotonDiffusion(ParameterInput *pin,
                                                  const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  auto *ptest_ideal =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *>(
          pmbp->pdyngr);
  if (ptest_ideal != nullptr) {
    return RadiationM1PhotonDiffusion_<Primitive::IdealGas, Primitive::ResetFloor>(
        pin, restart);
  }

  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_nqt != nullptr) {
    return RadiationM1PhotonDiffusion_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                       Primitive::ResetFloor>(pin, restart);
  }

  auto *ptest_nlog =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                     Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_nlog != nullptr) {
    return RadiationM1PhotonDiffusion_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                       Primitive::ResetFloor>(pin, restart);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::RadiationM1PhotonDiffusion_
template <class EOSPolicy, class ErrorPolicy>
void ProblemGenerator::RadiationM1PhotonDiffusion_(ParameterInput *pin,
                                                   const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr || pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The photon diffusion test requires dyngr and <radiation_m1>" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->params.opacity_type != radiationm1::Photons) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The photon diffusion test requires opacity_type = photons" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int nmb1 = (pmbp->nmb_thispack - 1);
  auto &w0_ = pmbp->pmhd->w0;
  auto &uradm1_ = pmbp->pradm1->u0;
  auto &m1_nvars_ = pmbp->pradm1->nvars;
  auto &size = pmbp->pmb->mb_size;

  // problem parameters
  Real rho = pin->GetReal("problem", "rho");
  Real temp = pin->GetReal("problem", "temp");
  Real e_amp = pin->GetReal("problem", "e_amp");
  Real e_floor = pin->GetReal("problem", "e_floor");
  Real sigma0 = pin->GetReal("problem", "sigma0");
  Real x0 = pin->GetOrAddReal("problem", "x0", 0.0);

  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr)
          ->eos.ps.GetEOSMutable();
  Real mb = eos.GetBaryonMass();
  Real nb = rho / mb;

  Real dd = 1.0 / (3.0 * pmbp->pradm1->photon_op_params.kappa_s * rho);

  // Minkowski metric
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  par_for(
      "pgen_photon_diff_metric", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1), 0,
      (n1 - 1), KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
          }
        adm.psi4(m, k, j, i) = 1.;
        adm.alpha(m, k, j, i) = 1.;
      });

  // static gas, Gaussian radiation profile
  par_for(
      "pgen_photon_diff", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1), 0, (n1 - 1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real ye_ = 0.5;
        w0_(m, IDN, k, j, i) = rho;
        w0_(m, IVX, k, j, i) = 0.;
        w0_(m, IVY, k, j, i) = 0.;
        w0_(m, IVZ, k, j, i) = 0.;
        w0_(m, IPR, k, j, i) = eos.GetPressure(nb, temp, &ye_);

        Real x1min = size.d_view(m).x1min;
        Real x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);
        Real arg = (x1v - x0) / sigma0;
        Real E0 = e_floor + e_amp * Kokkos::exp(-0.5 * arg * arg);

        Real Fx0 = dd * (x1v - x0) / (sigma0 * sigma0) * (E0 - e_floor);

        uradm1_(m, radiationm1::CombinedIdx(0, M1_E_IDX, m1_nvars_), k, j, i) = E0;
        uradm1_(m, radiationm1::CombinedIdx(0, M1_FX_IDX, m1_nvars_), k, j, i) = Fx0;
        uradm1_(m, radiationm1::CombinedIdx(0, M1_FY_IDX, m1_nvars_), k, j, i) = 0.;
        uradm1_(m, radiationm1::CombinedIdx(0, M1_FZ_IDX, m1_nvars_), k, j, i) = 0.;
      });

  pmbp->pdyngr->PrimToConInit(0, (n1 - 1), 0, (n2 - 1), 0, (n3 - 1));
}
