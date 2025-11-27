//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_compose_test.cpp
//  \brief Unit test for EOSCompOSE to make sure it works properly.

#include <sstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

template<class LogPolicy>
void PerformTests(Mesh* pmesh, ParameterInput *pin);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSCompOSE unit test only works for DynGRMHD!\n";
    exit(EXIT_FAILURE);
  }

  std::string eos_string = pin->GetString("mhd", "dyn_eos");

  if (eos_string.compare("compose") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSCompOSE unit test needs mhd/dyn_eos = compose!\n";
    exit(EXIT_FAILURE);
  }

  bool use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT", false);

  if (use_NQT) {
    PerformTests<Primitive::NQTLogs>(pmy_mesh_, pin);
  } else {
    PerformTests<Primitive::NormalLogs>(pmy_mesh_, pin);
  }

  std::cout << "Test Passed!\n";

  // This is needed to initialize the ADM variables to Minkowski. Otherwise the pgen
  // will have a bunch of C2P failures at the end.
  pmbp->padm->SetADMVariables(pmbp);

  return;
}

template<class LogPolicy>
void PerformTests(Mesh *pmesh, ParameterInput *pin) {
  MeshBlockPack *pmbp = pmesh->pmb_pack;

  // Commit a crime against humanity to get access to the EOS
  Primitive::EOS<Primitive::EOSCompOSE<LogPolicy>, Primitive::ResetFloor>& eos =
    static_cast<
      dyngr::DynGRMHDPS<
        Primitive::EOSCompOSE<LogPolicy>,
        Primitive::ResetFloor
      >*
    >(pmbp->pdyngr)->eos.ps.GetEOSMutable();

  // Get the range of the table
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
  int nY = pin->GetOrAddInteger("problem", "nY", 100);
  int nT = pin->GetOrAddInteger("problem", "nT", 100);

  Real dln = (lnmax - lnmin) / (nn - 1);
  Real dY  = (Ymax - Ymin) / (nY - 1);
  Real dlT = (lTmax - lTmin) / (nT - 1);

  // To make sure things are working as intended, we want to test what happens when things
  // are below and above the ranges of the table.
  int inlo = -1;
  int inhi = nn;
  int iYlo = -1;
  int iYhi = nY;
  int iTlo = -1;
  int iThi = nT;

  bool global_success = true;

  Real tol = static_cast<Real>(std::numeric_limits<float>::epsilon());

  const int ni = (iThi - iTlo + 1);
  const int nji = (iYhi - iYlo + 1)*ni;
  const int nkji = (inhi - inlo + 1)*nji;

  // Check the table's ability to handle an exact conversion.
  Kokkos::parallel_reduce("pgen_test", Kokkos::RangePolicy<>(DevExeSpace(), 0, nkji),
  KOKKOS_LAMBDA(const int &idx, bool &success) {
    int in = idx/nji;
    int iY = (idx - in*nji)/ni;
    const int iT = (idx - in*nji - iY*ni) + iTlo;
    iY += iYlo;
    in += inlo;

    Real Y[MAX_SPECIES] = {0.0};

    // Calculate the table input.
    // Note that we do *NOT* clamp the input values to the table ranges. The table
    // frequently gets slightly invalid units, and it needs to be able to deal with them
    // in a sensible way.
    Real ln = lnmin + in*dln;
    Real lT = lTmin + iT*dlT;
    Real n = logs.exp2_(ln);
    Y[0] = Ymin + iY*dY;
    Real T = logs.exp2_(lT);

    // Try to calculate the pressure and energy. We don't do anything with the pressure
    // (since it's not guaranteed to be monotonic), but this checks that it will get
    // calculated without failing.
    Real P = eos.GetPressure(n, T, Y);
    Real e = eos.GetEnergy(n, T, Y);

    // Try to invert the energy to get temperature
    Real T_test = eos.GetTemperatureFromE(n, e, Y);

    // Check the error on T
    Real error = T_test/T - 1.;
    if (Kokkos::fabs(error) > tol) {
      // Check if the failure was because we were outside the table.
      if (!(n < nmin || n > nmax ||
          Y[0] < Ymin || Y[0] > Ymax ||
          T < Tmin || T > Tmax)) {
        Kokkos::printf("The following point was recovered poorly:\n"
                       "  n = %20.17g\n"
                       "  Y = %20.17g\n"
                       "  T = %20.17g\n"
                       "Calculated temperature:\n"
                       "  T_test = %20.17g\n"
                       "  error = %20.17g\n",
                       n, Y[0], T, T_test, error);
        success = false;
      } else if ( (logs.log2_(T_test) < lTmin) || (logs.log2_(T_test) > lTmax)) {
        Kokkos::printf("The following point recovers an invalid temperature:\n"
                       "  n = %20.17g\n"
                       "  Y = %20.17g\n"
                       "  T = %20.17g\n"
                       "Calculated temperature:\n"
                       "  T_test = %20.17g\n"
                       "  Tmin = %20.17g\n"
                       "  Tmax = %20.17g\n",
                       n, Y[0], T, T_test, logs.exp2_(lTmin), logs.exp2_(lTmax));
        success = false;
      }
    }
  }, Kokkos::LAnd<bool>(global_success));

  // Check the table's ability to recover the temperature correctly when the energy or
  // pressure falls below the zero-temperature limit. We adjust the bounds of density and
  // Y to be physical; they should already be physical by this point.
  bool pert_success = true;
  inlo = 0;
  inhi = nn - 1;
  iYlo = 0;
  iYhi = nY - 1;
  const int nj = (iYhi - iYlo + 1);
  const int nkj = (inhi - inlo + 1)*nj;
  Kokkos::parallel_reduce("pgen_test", Kokkos::RangePolicy<>(DevExeSpace(), 0, nkj),
  KOKKOS_LAMBDA(const int &idx, bool &success) {
    int in = idx/nj;
    const int iY = (idx - in*nj) + iYlo;
    in += inlo;

    Real Y[MAX_SPECIES] = {0.0};

    // Calculate the table input assuming zero temperature.
    Real ln = lnmin + in*dln;
    Real lT = lTmin;
    Real n = logs.exp2_(ln);
    Y[0] = Ymin + iY*dY;
    Real T = logs.exp2_(lT);

    // Try to calculate the pressure and energy.
    Real P = eos.GetPressure(n, T, Y);
    Real e = eos.GetEnergy(n, T, Y);

    // Perturb both the pressure and the energy downward a significant amount.
    Real P_pert = 0.5*P;
    Real e_pert = 0.5*e;

    // Check that we recover the minimum temperature.
    Real T_p = eos.GetTemperatureFromP(n, P_pert, Y);
    Real T_e = eos.GetTemperatureFromE(n, e_pert, Y);

    Real error_p = T_p/T - 1.;
    Real error_e = T_e/T - 1.;
    if (Kokkos::fabs(error_p) > tol) {
      Kokkos::printf("The temperature was not recovered correctly from pressure:\n" // NOLINT
                     "  n = %20.17g\n"
                     "  Y = %20.17g\n"
                     "  T = %20.17g\n"
                     "Calculated temperature:\n"
                     "  T_test = %20.17g\n"
                     "  error = %20.17g\n",
                     n, Y[0], T, T_p, error_p);
      success = false;
    }
    if (Kokkos::fabs(error_e) > tol) {
      Kokkos::printf("The temperature was not recovered correctly from energy:\n" // NOLINT
                     "  n = %20.17g\n"
                     "  Y = %20.17g\n"
                     "  T = %20.17g\n"
                     "Calculated temperature:\n"
                     "  T_test = %20.17g\n"
                     "  error = %20.17g\n",
                     n, Y[0], T, T_e, error_e);
      success = false;
    }
  }, Kokkos::LAnd<bool>(pert_success));

  global_success = global_success && pert_success;

  if (!global_success) {
    std::cout << "The test was not successful...\n";
    exit(EXIT_FAILURE);
  }

  return;
}
