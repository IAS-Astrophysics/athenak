//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_rhea_singlezone.cpp
//! \brief Radiation M1 single-zone Rhea flavor-mixing fixed-point convergence test
//!
//! This is a new, dedicated pgen -- not an extension of rad_m1_singlezone.cpp (which is
//! tied to bns-nurates equilibration and is not the right harness for an isolated
//! Rhea/flavor-mix smoke test). It sets up a homogeneous, periodic zone with a
//! hand-specified initial M1 state (four species, deliberately asymmetric within each
//! nu/nubar sector so the mixing-matrix reconstruction never hits its degenerate branch)
//! and a *fixed* matter background (matter at rest, v=0, flat conformally-flat metric,
//! matter/GR sources for M1 turned off) so that homogeneity + periodicity make the MHD
//! and M1 flux-divergence terms vanish identically, leaving `flavor_mix = rhea`'s BGK
//! relaxation as the only thing that can change u0 from one cycle to the next.
//!
//! `integrator = rk1` (forward Euler, one explicit stage per cycle) is used deliberately,
//! so that "successive RK stages" and "successive cycles" (the unit AthenaK's tab-file
//! output records) coincide exactly -- this sidesteps needing any new sub-cycle output
//! machinery to check stage-to-stage convergence at the right granularity.
//!
//! With `flavor_mix = rhea` already wired into RadiationM1::FlavorMix's task-graph
//! dispatch, this pgen does not need to do anything special to exercise the pipeline --
//! PackRheaInputs -> RheaModel::Predict -> ApplyRheaMixing already runs once per stage
//! automatically. This file's only job is to set up initial data for which the resulting
//! fixed-point iteration is analytically tractable, and to fail loudly at startup if the
//! run is not actually configured to exercise that path.

// C++ headers
#include <iostream>

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::RadiationM1RheaSingleZoneTest
//! \brief Dispatch to the EOS-templated implementation, exactly mirroring
//! RadiationM1SingleZoneTest's own dynamic_cast dispatch (rad_m1_singlezone.cpp).
void ProblemGenerator::RadiationM1RheaSingleZoneTest(ParameterInput *pin,
                                                      const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_nqt != nullptr) {
    return RadiationM1RheaSingleZoneTest_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                          Primitive::ResetFloor>(pin, restart);
  }

  auto *ptest_nlog =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                     Primitive::ResetFloor> *>(pmbp->pdyngr);
  if (ptest_nlog != nullptr) {
    return RadiationM1RheaSingleZoneTest_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                          Primitive::ResetFloor>(pin, restart);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

template <class EOSPolicy, class ErrorPolicy>
void ProblemGenerator::RadiationM1RheaSingleZoneTest_(ParameterInput *pin,
                                                       const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // -------------------------------------------------------------------------------------
  // Preconditions. Rhea mixing requires dynamical-GR MHD (PackRheaInputs needs
  // eos.GetEOSUnitSystem()) and flavor_mix = rhea specifically -- this pgen is only a
  // meaningful test of the Rhea path, not a generic single-zone harness, so it fails
  // loudly (matching rad_m1_singlezone.cpp's own style) rather than silently doing
  // something else.
  // -------------------------------------------------------------------------------------
  if (pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "DynGRMHD is required for the Rhea single zone convergence test"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The Rhea single zone convergence test problem generator requires "
                 "radiation-m1, but no <radiation_m1> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->params.flavor_mix_type != radiationm1::FlavMixRhea) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The Rhea single zone convergence test problem generator requires "
                 "flavor_mix = rhea" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->nspecies != 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Rhea flavor mixing requires nspecies == 4 (compile with "
                 "-DAthena_ENABLE_NURATES=ON, which sets M1_TOTAL_NUM_SPECIES=4)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int nmb1 = (pmbp->nmb_thispack - 1);
  auto &w0_ = pmbp->pmhd->w0;
  auto &uradm1_ = pmbp->pradm1->u0;
  auto &nspecies_ = pmbp->pradm1->nspecies;
  auto &m1_nvars_ = pmbp->pradm1->nvars;

  // -------------------------------------------------------------------------------------
  // Matter background (fixed for the whole run, by construction): homogeneous + periodic
  // means MHD's flux divergence is exactly zero every cycle (identical states on both
  // sides of every interface -> hlle/dc give exactly zero net flux, as
  // rad_m1_singlezone.cpp already relies on), so rho/temp/v never actually evolve;
  // matter_sources = gr_sources = backreact = false in the input file additionally
  // ensures no M1<->matter source coupling could perturb this. v = 0 (matter at rest) is
  // a *hard* requirement of this test's analytic fixed-point derivation: it is what makes
  // the fluid-frame and lab (Eulerian) frames coincide exactly, giving Gamma_s = 1 and
  // J_s = E_s, H_d = F_d exactly for every species, with no relativistic-beaming
  // correction to track by hand.
  // -------------------------------------------------------------------------------------
  Real rho = pin->GetReal("problem", "rho");
  Real temp = pin->GetReal("problem", "temp");
  Real ye = pin->GetReal("problem", "Y_e");

  Real mb{};
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr)
          ->eos.ps.GetEOSMutable();
  mb = eos.GetBaryonMass();

  Real rho_code = rho;
  Real nb = rho / mb;

  // -------------------------------------------------------------------------------------
  // Per-species initial M1 state. Species index convention (matches
  // ReconstructMixingMatrix/ApplyRheaMixing, radiation_m1_rhea_kernels.hpp): 0=nu_e,
  // 1=nu_ebar, 2=nu_x, 3=nu_xbar. N/E are deliberately distinct *within* each sector
  // {0,2} and {1,3} so the mixing-matrix reconstruction's degenerate (|N_e-N_x| ~ 0 ->
  // p=1 identity) branch is never hit at t=0; F = 0 for all species (isotropic) --
  // combined with the homogeneous/periodic setup, this stays exactly 0 for all time (Y
  // applied to an all-zero flux vector is exactly zero, regardless of Y), which is both
  // physically the simplest case and what keeps RestrictToPhysical's boost correction
  // exactly evaluable by hand.
  // -------------------------------------------------------------------------------------
  Real N_init[4], E_init[4];
  N_init[0] = pin->GetReal("problem", "rad_N0");
  N_init[1] = pin->GetReal("problem", "rad_N1");
  N_init[2] = pin->GetReal("problem", "rad_N2");
  N_init[3] = pin->GetReal("problem", "rad_N3");
  E_init[0] = pin->GetReal("problem", "rad_E0");
  E_init[1] = pin->GetReal("problem", "rad_E1");
  E_init[2] = pin->GetReal("problem", "rad_E2");
  E_init[3] = pin->GetReal("problem", "rad_E3");

  Real N0 = N_init[0], N1 = N_init[1], N2 = N_init[2], N3 = N_init[3];
  Real E0 = E_init[0], E1 = E_init[1], E2 = E_init[2], E3 = E_init[3];

  // initialize ADM variables: flat conformally-flat metric, matching
  // rad_m1_singlezone.cpp.
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  par_for(
      "pgen_rhea_singlezone_metric", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1), 0,
      (n1 - 1), KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int a = 0; a < 3; ++a) {
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
          }
          adm.beta_u(m, a, k, j, i) = 0.;
        }
        adm.psi4(m, k, j, i) = 1.;
        adm.alpha(m, k, j, i) = 1.;
      });

  // set primitive (matter, at rest) and M1 conserved variables
  par_for(
      "pgen_rhea_singlezone_data", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1), 0,
      (n1 - 1), KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real ye_ = ye;

        w0_(m, IDN, k, j, i) = rho_code;
        w0_(m, IVX, k, j, i) = 0.0;
        w0_(m, IVY, k, j, i) = 0.0;
        w0_(m, IVZ, k, j, i) = 0.0;
        w0_(m, IPR, k, j, i) = eos.GetPressure(nb, temp, &ye_);
        w0_(m, IYF, k, j, i) = ye;

        Real N_local[4] = {N0, N1, N2, N3};
        Real E_local[4] = {E0, E1, E2, E3};
        for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
          uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, m1_nvars_), k, j, i) =
              E_local[nuidx];
          uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, m1_nvars_), k, j, i) =
              N_local[nuidx];
          uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, m1_nvars_), k, j, i) =
              0.0;
          uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, m1_nvars_), k, j, i) =
              0.0;
          uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, m1_nvars_), k, j, i) =
              0.0;
        }
      });

  // Convert primitives to conserved vars
  pmbp->pdyngr->PrimToConInit(0, (n1 - 1), 0, (n2 - 1), 0, (n3 - 1));
}
