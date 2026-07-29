//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_transition.cpp
//  \brief Unit test for EOSTransition (CompOSE + Helmholtz blend).
//
//  Sweeps (n, T, Ye) across the joint table range -- Helmholtz low-density edge,
//  the freeze-out strip, and the CompOSE interior -- and checks that P, e, cs are
//  finite and that the temperature is recovered from the energy. Requires both a
//  CompOSE .athtab (with composition channels) and a Helmholtz .athtab.

#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

template<class LogPolicy>
void PerformTransitionTests(Mesh* pmesh, ParameterInput *pin);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::EOSTransition()
//! \brief Runs EOS transition unit tests

void ProblemGenerator::EOSTransition(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition unit test only works for DynGRMHD!" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string eos_string = pin->GetString("mhd", "dyn_eos");
  if (eos_string.compare("transition") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition unit test needs mhd/dyn_eos = transition!" << std::endl;
    exit(EXIT_FAILURE);
  }

  bool use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT", false);
  if (use_NQT) {
    PerformTransitionTests<Primitive::NQTLogs>(pmy_mesh_, pin);
  } else {
    PerformTransitionTests<Primitive::NormalLogs>(pmy_mesh_, pin);
  }

  std::cout << "Test Passed!\n";

  // Initialize the ADM variables to Minkowski to avoid trailing C2P failures.
  pmbp->padm->SetADMVariables(pmbp);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PerformTransitionTests()

template<class LogPolicy>
void PerformTransitionTests(Mesh *pmesh, ParameterInput *pin) {
  MeshBlockPack *pmbp = pmesh->pmb_pack;

  Primitive::EOS<Primitive::EOSTransition<LogPolicy>, Primitive::ResetFloor>& eos =
    static_cast<
      dyngr::DynGRMHDPS<
        Primitive::EOSTransition<LogPolicy>,
        Primitive::ResetFloor
      >*
    >(pmbp->pdyngr)->eos.ps.GetEOSMutable();

  LogPolicy logs;
  Real nmin = eos.GetMinimumDensity();
  Real nmax = eos.GetMaximumDensity();
  Real lnmin = logs.log2_(nmin);
  Real lnmax = logs.log2_(nmax);

  Real Ymin = eos.GetMinimumSpeciesFraction(0);
  Real Ymax = eos.GetMaximumSpeciesFraction(0);

  Real Tmin = eos.GetMinimumTemperature();
  Real Tmax = eos.GetMaximumTemperature();
  Real lTmin = logs.log2_(Tmin);
  Real lTmax = logs.log2_(Tmax);

  int nn = pin->GetOrAddInteger("problem", "nn", 100);
  int nY = pin->GetOrAddInteger("problem", "nY", 50);
  int nT = pin->GetOrAddInteger("problem", "nT", 100);

  Real dln = (lnmax - lnmin) / (nn - 1);
  Real dY  = (Ymax - Ymin) / (nY - 1);
  Real dlT = (lTmax - lTmin) / (nT - 1);

  // Hard-fail tolerance on the temperature round-trip. Kept loose because the
  // blended e(T) can be non-monotone across the freeze-out strip, where an
  // exact inverse is not guaranteed; gross failures still indicate a bug.
  Real tol = pin->GetOrAddReal("problem", "roundtrip_tol", 1e-6);

  const int ni = nT;
  const int nji = nY*ni;
  const int nkji = nn*nji;

  bool global_success = true;

  Kokkos::parallel_reduce("eos_transition_test",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nkji),
  KOKKOS_LAMBDA(const int &idx, bool &success) {
    int in = idx/nji;
    int iY = (idx - in*nji)/ni;
    const int iT = (idx - in*nji - iY*ni);

    Real n  = logs.exp2_(lnmin + in*dln);
    Real T  = logs.exp2_(lTmin + iT*dlT);
    Real Ye = Ymin + iY*dY;

    // Free-nucleon composition consistent with Ye (sanitized internally).
    Real Y[MAX_SPECIES] = {0.0};
    Y[SCYE] = Ye;
    Y[SCXN] = 1.0 - Ye;
    Y[SCXP] = Ye;
    Y[SCXA] = 0.0;
    Y[SCXH] = 0.0;
    Y[SCAH] = 1.0;
    Y[SCEB] = 0.0;

    Real P  = eos.GetPressure(n, T, Y);
    Real e  = eos.GetEnergy(n, T, Y);
    Real cs = eos.GetSoundSpeed(n, T, Y);

    if (!Kokkos::isfinite(P) || !Kokkos::isfinite(e) || !Kokkos::isfinite(cs) ||
        P <= 0.0 || e <= 0.0 || cs < 0.0 || cs > 1.0) {
      Kokkos::printf("Non-physical thermodynamics at n=%.6e T=%.6e Ye=%.6e: "
                     "P=%.6e e=%.6e cs=%.6e\n", n, T, Ye, P, e, cs);
      success = false;
      return;
    }

    Real T_test = eos.GetTemperatureFromE(n, e, Y);
    Real error = T_test/T - 1.0;
    if (Kokkos::fabs(error) > tol) {
      Kokkos::printf("Temperature poorly recovered at n=%.6e T=%.6e Ye=%.6e: "
                     "T_test=%.6e error=%.6e\n", n, T, Ye, T_test, error);
      success = false;
    }
  }, Kokkos::LAnd<bool>(global_success));

  if (!global_success) {
    std::cout << "The transition EOS test was not successful..." << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}
