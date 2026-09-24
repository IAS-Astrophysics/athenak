//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_compose.cpp
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

template<class LogPolicy>
void PerformNuEqTests(Mesh* pmesh, ParameterInput *pin);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::EOSCompose()
//! \brief Runs EOS compose unit tests

void ProblemGenerator::EOSCompose(ParameterInput *pin, const bool restart) {
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
    PerformNuEqTests<Primitive::NQTLogs>(pmy_mesh_, pin);
  } else {
    PerformTests<Primitive::NormalLogs>(pmy_mesh_, pin);
    PerformNuEqTests<Primitive::NormalLogs>(pmy_mesh_, pin);
  }

  std::cout << "Test Passed!\n";

  // This is needed to initialize the ADM variables to Minkowski. Otherwise the pgen
  // will have a bunch of C2P failures at the end.
  pmbp->padm->SetADMVariables(pmbp);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PerformTests()
//! \brief

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

//----------------------------------------------------------------------------------------
//! \fn void PerformNuEqTests()
//! \brief Tests the trapped weak-equilibrium solver through the public EOS interface.
//!
//!   A. Round trip. Given (n, T, Y_e), GetTrappedNeutrinos supplies the equilibrium
//!      neutrino content, from which (Y_le, e) follow, so (T, Y_e) is by construction an
//!      exact solution. GetBetaEquilibriumTrapped must recover it from a deliberately
//!      poor initial guess -- which is where a line search that amplifies rather than
//!      damps the Newton step shows up.
//!   B. Inconsistent states. With Y_le pushed far out of reach of the given energy the
//!      2D system has no solution, and the solver must still return a usable answer
//!      rather than the unequilibrated guess: either a constrained solution on a table
//!      edge or the energy-only fallback, which must leave Y_e untouched and satisfy the
//!      energy equation.
//!   C. Gradient step width. The Jacobian's temperature derivatives are secants of the
//!      trilinear interpolant, so their smoothness is set by the step width: one table
//!      cell is continuous in T, anything narrower returns the local cell slope and
//!      jumps at every cell boundary. eta_e_gradient itself is private, so what is
//!      measured here is the property of the two candidate widths that motivates it.
//!   D. Weight endpoints. GetBetaEquilibriumPartial is the same solve with one weight
//!      per neutrino channel, and its two endpoints are fixed in advance: unit weights
//!      must reproduce GetBetaEquilibriumTrapped bit for bit, and zero weights must
//!      return the matter state the right-hand sides were built from, from any guess.
//!      The interior of the family has no closed-form answer to test against.
//!
//! Energies here are always in code units, since they come from GetEnergy and
//! GetTrappedNeutrinos and go straight back into GetBetaEquilibriumTrapped. The
//! temperature bounds, however, are documented as EOS units, so the test assumes the
//! code temperature unit is also MeV -- true of both unit systems used with a table
//! (nuclear and geometric_solar). Under any other choice the sampled states land outside
//! the table and the test fails loudly rather than passing quietly.

template<class LogPolicy>
void PerformNuEqTests(Mesh *pmesh, ParameterInput *pin) {
  MeshBlockPack *pmbp = pmesh->pmb_pack;

  // Commit a crime against humanity to get access to the EOS
  Primitive::EOS<Primitive::EOSCompOSE<LogPolicy>, Primitive::ResetFloor>& eos =
    static_cast<
      dyngr::DynGRMHDPS<
        Primitive::EOSCompOSE<LogPolicy>,
        Primitive::ResetFloor
      >*
    >(pmbp->pdyngr)->eos.ps.GetEOSMutable();

  const Real nmin = eos.GetMinimumDensity();
  const Real nmax = eos.GetMaximumDensity();
  const Real Ymin = eos.GetMinimumSpeciesFraction(0);
  const Real Ymax = eos.GetMaximumSpeciesFraction(0);
  const Real Tmin = eos.GetMinimumTemperature();
  const Real Tmax = eos.GetMaximumTemperature();

  // Sample conditions where trapped neutrinos are a sensible description at all: a few
  // times nuclear density downwards, several MeV upwards, and neutron rich. The high-Yq
  // corner of a table is deliberately excluded -- matter that proton rich does not hold
  // trapped neutrinos -- as are the table edges, where a constrained solution on the
  // boundary is the correct answer rather than an interior one.
  LogPolicy logs;
  const Real ln_lo = logs.log2_((0.02 > nmin) ? 0.02 : 2.0*nmin);
  const Real ln_hi = logs.log2_((0.4 < nmax) ? 0.4 : 0.5*nmax);
  const Real lT_lo = logs.log2_((2.0 > Tmin) ? 2.0 : 2.0*Tmin);
  const Real lT_hi = logs.log2_((40.0 < Tmax) ? 40.0 : 0.5*Tmax);
  const Real Y_lo = (0.05 > Ymin) ? 0.05 : Ymin + 0.05*(Ymax - Ymin);
  const Real Y_hi = (0.3 < Ymax) ? 0.3 : Ymax - 0.05*(Ymax - Ymin);

  const int nn = 8;
  const int nY = 8;
  const int nT = 16;
  const Real dln = (ln_hi - ln_lo)/(nn - 1);
  const Real dY = (Y_hi - Y_lo)/(nY - 1);
  const Real dlT = (lT_hi - lT_lo)/(nT - 1);
  const int nstates = nn*nY*nT;

  bool success = true;

  // ------------------------------------------------------------------------------
  // A. round trip on constructed exact equilibria
  // ------------------------------------------------------------------------------
  int n_fail = 0;
  int n_not_interior = 0;
  Real err_T_max = 0.0;
  Real err_Y_max = 0.0;
  Real res_max = 0.0;

  Kokkos::parallel_reduce("nueq_roundtrip",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nstates),
  KOKKOS_LAMBDA(const int &idx, int &nf, int &nni, Real &eT, Real &eY, Real &rmax) {
    const int in = idx/(nY*nT);
    const int iY = (idx - in*nY*nT)/nT;
    const int iT = idx - in*nY*nT - iY*nT;

    const Real n = logs.exp2_(ln_lo + in*dln);
    const Real T = logs.exp2_(lT_lo + iT*dlT);
    Real Y[MAX_SPECIES] = {0.0};
    Y[0] = Y_lo + iY*dY;

    // Build a state for which (T, Y[0]) is an exact weak equilibrium
    Real n_nu[3], e_nu[3];
    eos.GetTrappedNeutrinos(n, T, Y, n_nu, e_nu);
    Real Yl[MAX_SPECIES] = {0.0};
    Yl[0] = Y[0] + n_nu[0]/n;
    const Real e_tot = eos.GetEnergy(n, T, Y) + e_nu[0] + e_nu[1] + e_nu[2];

    // Start well away from the answer
    Real T_guess = 1.4*T;
    Real Y_guess[MAX_SPECIES] = {0.0};
    Y_guess[0] = 0.7*Y[0];

    Real T_eq = 0.0;
    Real Y_eq[MAX_SPECIES] = {0.0};
    int status = -1;
    bool ok = eos.GetBetaEquilibriumTrapped(n, e_tot, Yl, T_eq, Y_eq,
                                            T_guess, Y_guess, &status);

    if (!ok) {
      Kokkos::printf("GetBetaEquilibriumTrapped failed on an exact equilibrium:\n"
                     "  n = %20.17g\n  T = %20.17g\n  Ye = %20.17g\n"
                     "  status = %d\n", n, T, Y[0], status);
      nf += 1;
      return;
    }

    Real err_T = Kokkos::fabs(T_eq/T - 1.0);
    Real err_Y = Kokkos::fabs(Y_eq[0] - Y[0]);
    eT = (err_T > eT) ? err_T : eT;
    eY = (err_Y > eY) ? err_Y : eY;
    if (status != Primitive::EOSCompOSE<LogPolicy>::NUEQ_INTERIOR) {
      nni += 1;
    }

    // What the solver actually promises is a small residual, not a small error in T:
    // at low temperature the thermal part of the energy is a tiny fraction of the
    // total, so a converged energy residual still leaves a much larger error in T.
    // Rebuild the residual it converged on and check that instead.
    Real n_nu_eq[3], e_nu_eq[3];
    eos.GetTrappedNeutrinos(n, T_eq, Y_eq, n_nu_eq, e_nu_eq);
    Real res = Kokkos::fabs((Y_eq[0] + n_nu_eq[0]/n - Yl[0])/Yl[0]) +
               Kokkos::fabs((eos.GetEnergy(n, T_eq, Y_eq) + e_nu_eq[0] + e_nu_eq[1] +
                             e_nu_eq[2])/e_tot - 1.0);
    rmax = (res > rmax) ? res : rmax;
  }, Kokkos::Sum<int>(n_fail), Kokkos::Sum<int>(n_not_interior),
     Kokkos::Max<Real>(err_T_max), Kokkos::Max<Real>(err_Y_max),
     Kokkos::Max<Real>(res_max));

  // The solver's own convergence criterion is 1e-7 on this residual; allow a little
  // room for the round trip through code units.
  const Real tol_res = 1.0e-6;
  // Loose bounds on the recovered state, to catch a converged but wrong root.
  const Real tol_T = 1.0e-2;
  const Real tol_Y = 1.0e-4;

  std::cout << "Trapped weak equilibrium, " << nstates << " exact states:\n"
            << "  failures                : " << n_fail << "\n"
            << "  non-interior solutions  : " << n_not_interior << "\n"
            << "  max residual            : " << res_max << "\n"
            << "  max |T_eq/T - 1|        : " << err_T_max << "\n"
            << "  max |Y_eq - Y_e|        : " << err_Y_max << "\n";

  if (n_fail != 0 || n_not_interior != 0) {
    success = false;
  }
  if (!(res_max < tol_res)) {
    std::cout << "The solver returned points that do not satisfy its own convergence "
              << "criterion (tolerance " << tol_res << ").\n";
    success = false;
  }
  if (!(err_T_max < tol_T) || !(err_Y_max < tol_Y)) {
    std::cout << "The recovered equilibrium is too far from the state it was built "
              << "from (tolerances " << tol_T << " on T, " << tol_Y << " on Y_e).\n";
    success = false;
  }

  // ------------------------------------------------------------------------------
  // B. states for which no equilibrium exists
  // ------------------------------------------------------------------------------
  // Every one of these runs the restart ladder to exhaustion, so use a coarse subgrid
  // rather than the full one.
  const int nn_b = 2;
  const int nY_b = 3;
  const int nT_b = 4;
  const int nbad = 4;
  const int nstates_b = nn_b*nY_b*nT_b;
  const Real dln_b = (ln_hi - ln_lo)/(nn_b - 1);
  const Real dY_b = (Y_hi - Y_lo)/(nY_b - 1);
  const Real dlT_b = (lT_hi - lT_lo)/(nT_b - 1);

  int n_unusable = 0;
  int n_constrained = 0;
  int n_energy_only = 0;
  int n_Ye_moved = 0;
  Real res_e_max = 0.0;
  Real res_guess_min = 1.0e30;
  Real dT_max = 0.0;

  Kokkos::parallel_reduce("nueq_inconsistent",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nstates_b*nbad),
  KOKKOS_LAMBDA(const int &idx, int &nu, int &nc, int &neo, int &nym, Real &rmax,
                Real &rgmin, Real &dTmax) {
    const int ib = idx/nstates_b;
    const int is = idx - ib*nstates_b;
    const int in = is/(nY_b*nT_b);
    const int iY = (is - in*nY_b*nT_b)/nT_b;
    const int iT = is - in*nY_b*nT_b - iY*nT_b;

    const Real n = logs.exp2_(ln_lo + in*dln_b);
    const Real T = logs.exp2_(lT_lo + iT*dlT_b);
    Real Y[MAX_SPECIES] = {0.0};
    Y[0] = Y_lo + iY*dY_b;

    // A matter state carrying three times the equilibrium neutrino energy, so that the
    // guess (T, Y[0]) is *not* a solution of the energy equation either. This is the
    // situation in the M1 solver: the guess is the current matter temperature and the
    // energy includes whatever the radiation field actually holds.
    Real n_nu[3], e_nu[3];
    eos.GetTrappedNeutrinos(n, T, Y, n_nu, e_nu);
    const Real e_mat = eos.GetEnergy(n, T, Y);
    const Real e_tot = e_mat + 3.0*(e_nu[0] + e_nu[1] + e_nu[2]);

    // Y_le far out of reach of this energy, in both directions
    const Real Yl_bad[nbad] = {Y[0] + 1.0, Y[0] + 5.0, Y[0] - 1.0, Y[0] - 5.0};
    Real Yl[MAX_SPECIES] = {0.0};
    Yl[0] = Yl_bad[ib];

    Real T_eq = 0.0;
    Real Y_eq[MAX_SPECIES] = {0.0};
    Real Y_guess[MAX_SPECIES] = {0.0};
    Y_guess[0] = Y[0];
    int status = -1;
    bool ok = eos.GetBetaEquilibriumTrapped(n, e_tot, Yl, T_eq, Y_eq, T, Y_guess,
                                            &status);

    if (!ok) {
      nu += 1;
      return;
    }

    if (status == Primitive::EOSCompOSE<LogPolicy>::NUEQ_CONSTRAINED) {
      nc += 1;
    } else if (status == Primitive::EOSCompOSE<LogPolicy>::NUEQ_ENERGY_ONLY) {
      neo += 1;
      // the fallback must leave Y_e alone ...
      if (Y_eq[0] != Y_guess[0]) {
        nym += 1;
      }
      // ... and satisfy the energy equation it solved
      Real n_nu_eq[3], e_nu_eq[3];
      eos.GetTrappedNeutrinos(n, T_eq, Y_eq, n_nu_eq, e_nu_eq);
      Real res = Kokkos::fabs((eos.GetEnergy(n, T_eq, Y_eq) + e_nu_eq[0] + e_nu_eq[1] +
                               e_nu_eq[2])/e_tot - 1.0);
      rmax = (res > rmax) ? res : rmax;

      // For scale: the residual carried by the guess, which is what the solver used to
      // return here, and how far the equilibrium temperature actually is from it. Note
      // the residual can be small while the temperature shift is enormous -- the
      // thermal part of the energy is a small fraction of the total -- and that its
      // sign, hence the sign of the emissivity error, depends on whether the radiation
      // field holds more or less energy than equilibrium would.
      Real res_guess = Kokkos::fabs((e_mat + e_nu[0] + e_nu[1] + e_nu[2])/e_tot - 1.0);
      rgmin = (res_guess < rgmin) ? res_guess : rgmin;
      Real dT = Kokkos::fabs(T_eq/T - 1.0);
      dTmax = (dT > dTmax) ? dT : dTmax;
    }
  }, Kokkos::Sum<int>(n_unusable), Kokkos::Sum<int>(n_constrained),
     Kokkos::Sum<int>(n_energy_only), Kokkos::Sum<int>(n_Ye_moved),
     Kokkos::Max<Real>(res_e_max), Kokkos::Min<Real>(res_guess_min),
     Kokkos::Max<Real>(dT_max));

  std::cout << "Inconsistent states, " << nstates_b*nbad << " with Y_le out of reach:\n"
            << "  no usable answer        : " << n_unusable << "\n"
            << "  constrained solutions   : " << n_constrained << "\n"
            << "  energy-only fallbacks   : " << n_energy_only << "\n"
            << "  solved anyway           : "
            << nstates_b*nbad - n_unusable - n_constrained - n_energy_only << "\n"
            << "  Y_e moved by fallback   : " << n_Ye_moved << "\n"
            << "  max energy residual     : " << res_e_max << "\n"
            << "  min residual at guess   : " << res_guess_min << "\n"
            << "  max |T_eq/T_guess - 1|  : " << dT_max << "\n";

  if (n_unusable != 0) {
    std::cout << "Some inconsistent states returned no usable equilibrium at all; the "
              << "1D energy-only fallback should have caught them.\n";
    success = false;
  }
  if (n_Ye_moved != 0) {
    std::cout << "The energy-only fallback changed Y_e, which it must not do.\n";
    success = false;
  }
  if (n_energy_only == 0) {
    std::cout << "No state exercised the energy-only fallback, so this test is not "
              << "discriminating.\n";
    success = false;
  } else if (!(res_e_max < 1.0e-6)) {
    std::cout << "The energy-only fallback does not satisfy the energy equation.\n";
    success = false;
  } else if (!(dT_max > 0.1)) {
    // The point of the fallback is that the guess it replaces is nowhere near a
    // solution. Judge that on the temperature, not on the residual: because the thermal
    // part of the energy is a small fraction of the total, the guess can sit at a
    // residual of 1e-6 while its temperature is out by an order of magnitude, and it is
    // the temperature that sets the emissivity.
    std::cout << "The energy-only fallback returned essentially the temperature it was "
              << "given, so it is not being tested against anything.\n";
    success = false;
  }

  // ------------------------------------------------------------------------------
  // C. width of the Jacobian's temperature secant
  // ------------------------------------------------------------------------------
  const Real T_delta_narrow = 0.01;   // MeV: the width the solver used to use
  const int nscan = 2000;
  const Real n_scan = logs.exp2_(0.5*(ln_lo + ln_hi));
  const Real Y_scan = 0.5*(Y_lo + Y_hi);
  const Real dlT_scan = (lT_hi - lT_lo)/nscan;
  auto log_T_grid = eos.GetRawLogTemperature();

  Real jump_cell = 0.0;
  Real jump_narrow = 0.0;

  Kokkos::parallel_reduce("nueq_gradient",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nscan),
  KOKKOS_LAMBDA(const int &i, Real &jc, Real &jn) {
    Real Y[MAX_SPECIES] = {0.0};
    Y[0] = Y_scan;

    // half a table cell in log2(T)
    const Real T_fac = logs.exp2_(0.5*(log_T_grid(1) - log_T_grid(0)));

    Real cv_cell[2] = {0.0};
    Real cv_narrow[2] = {0.0};
    for (int s = 0; s < 2; ++s) {
      const Real T = logs.exp2_(lT_lo + (i + s)*dlT_scan);
      cv_cell[s] = (eos.GetEnergy(n_scan, T*T_fac, Y) -
                    eos.GetEnergy(n_scan, T/T_fac, Y))/(T*T_fac - T/T_fac);
      cv_narrow[s] = (eos.GetEnergy(n_scan, T + T_delta_narrow, Y) -
                      eos.GetEnergy(n_scan, T - T_delta_narrow, Y))/(2.0*T_delta_narrow);
    }

    Real dc = Kokkos::fabs(cv_cell[1]/cv_cell[0] - 1.0);
    Real dn = Kokkos::fabs(cv_narrow[1]/cv_narrow[0] - 1.0);
    jc = (dc > jc) ? dc : jc;
    jn = (dn > jn) ? dn : jn;
  }, Kokkos::Max<Real>(jump_cell), Kokkos::Max<Real>(jump_narrow));

  std::cout << "de_dT secant, T scan at n = " << n_scan << ", Y_e = " << Y_scan << ":\n"
            << "  max relative jump, one table cell : " << jump_cell << "\n"
            << "  max relative jump, 0.01 MeV step  : " << jump_narrow << "\n";

  if (!(jump_cell < 0.1*jump_narrow)) {
    std::cout << "A cell-wide secant is not much smoother in T than a sub-cell one, so "
              << "the premise of the eta_e_gradient step width does not hold for this "
              << "table.\n";
    success = false;
  }

  // ------------------------------------------------------------------------------
  // D. the two endpoints of the weight family
  // ------------------------------------------------------------------------------
  // BetaEquilibriumPartial carries one weight per neutrino channel and reduces to no
  // solve at all when they are 0: the right-hand sides are then the bare matter state
  // and the answer is that state, however poor the initial guess. That is the half of
  // this group with real content.
  //
  // The w = 1 half is narrow, and worth being explicit about: GetBetaEquilibriumTrapped
  // is implemented as GetBetaEquilibriumPartial with five unit weights, so comparing
  // them exercises the two wrappers' unit conversions and the defaulted weights, not
  // the arithmetic underneath. What pins the w = 1 limit to the trapped physics is
  // group A, which rebuilds the trapped residual from GetTrappedNeutrinos -- an
  // independent evaluation of the same closed forms -- and checks it vanishes.
  //
  // Both halves sit at equal weights within each pair, so this group says nothing about
  // the split electron-pair terms. Group F is the one that does.
  const Real tol_w1 = 1.0e-10;
  int n_w1_differs = 0;
  int n_w1_exact = 0;
  int n_w0_fail = 0;
  Real err_w0_T_max = 0.0;
  Real err_w0_Y_max = 0.0;
  Real res_w0_max = 0.0;

  Kokkos::parallel_reduce("nueq_weights",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nstates),
  KOKKOS_LAMBDA(const int &idx, int &nd, int &nx, int &nf, Real &eT, Real &eY,
                Real &rmax) {
    const int in = idx/(nY*nT);
    const int iY = (idx - in*nY*nT)/nT;
    const int iT = idx - in*nY*nT - iY*nT;

    const Real n = logs.exp2_(ln_lo + in*dln);
    const Real T = logs.exp2_(lT_lo + iT*dlT);
    Real Y[MAX_SPECIES] = {0.0};
    Y[0] = Y_lo + iY*dY;

    Real n_nu[3], e_nu[3];
    eos.GetTrappedNeutrinos(n, T, Y, n_nu, e_nu);
    Real Yl[MAX_SPECIES] = {0.0};
    Yl[0] = Y[0] + n_nu[0]/n;
    const Real e_mat = eos.GetEnergy(n, T, Y);
    const Real e_tot = e_mat + e_nu[0] + e_nu[1] + e_nu[2];

    const Real T_guess = 1.4*T;
    Real Y_guess[MAX_SPECIES] = {0.0};
    Y_guess[0] = 0.7*Y[0];

    Real T_ref = 0.0;
    Real T_w1 = 0.0;
    Real Y_ref[MAX_SPECIES] = {0.0};
    Real Y_w1[MAX_SPECIES] = {0.0};
    bool ok_ref = eos.GetBetaEquilibriumTrapped(n, e_tot, Yl, T_ref, Y_ref,
                                                T_guess, Y_guess);
    const Real w_one[PEQ_NWEIGHTS] = {1.0, 1.0, 1.0, 1.0, 1.0};
    bool ok_w1 = eos.GetBetaEquilibriumPartial(n, e_tot, Yl, w_one,
                                               T_w1, Y_w1, T_guess, Y_guess);
    if (ok_ref != ok_w1 ||
        !(Kokkos::fabs(T_w1/T_ref - 1.0) < tol_w1) ||
        !(Kokkos::fabs(Y_w1[0] - Y_ref[0]) < tol_w1)) {
      nd += 1;
    }
    if (T_w1 == T_ref && Y_w1[0] == Y_ref[0]) {
      nx += 1;
    }


    // All weights 0 removes every neutrino term from both residuals, so the right-hand
    // sides are the bare matter state and the solution is that state itself -- however
    // poor the initial guess.
    Real Yl0[MAX_SPECIES] = {0.0};
    Yl0[0] = Y[0];
    Real T_w0 = 0.0;
    Real Y_w0[MAX_SPECIES] = {0.0};
    const Real w_zero[PEQ_NWEIGHTS] = {0.0, 0.0, 0.0, 0.0, 0.0};
    bool ok_w0 = eos.GetBetaEquilibriumPartial(n, e_mat, Yl0, w_zero,
                                               T_w0, Y_w0, T_guess, Y_guess);
    if (!ok_w0) {
      nf += 1;
      return;
    }

    Real err_T = Kokkos::fabs(T_w0/T - 1.0);
    Real err_Y = Kokkos::fabs(Y_w0[0] - Y[0]);
    Real res = Kokkos::fabs(eos.GetEnergy(n, T_w0, Y_w0)/e_mat - 1.0);
    eT = (err_T > eT) ? err_T : eT;
    eY = (err_Y > eY) ? err_Y : eY;
    rmax = (res > rmax) ? res : rmax;
  }, Kokkos::Sum<int>(n_w1_differs), Kokkos::Sum<int>(n_w1_exact),
     Kokkos::Sum<int>(n_w0_fail),
     Kokkos::Max<Real>(err_w0_T_max), Kokkos::Max<Real>(err_w0_Y_max),
     Kokkos::Max<Real>(res_w0_max));

  std::cout << "Partial equilibrium endpoints, " << nstates << " states:\n"
            << "  w = 1 vs trapped entry pt  : " << n_w1_differs
            << " differ (tolerance " << tol_w1 << "), "
            << n_w1_exact << " of " << nstates << " identical\n"
            << "  w = 0 failures             : " << n_w0_fail << "\n"
            << "  w = 0 max residual         : " << res_w0_max << "\n"
            << "  w = 0 max |T_eq/T - 1|     : " << err_w0_T_max << "\n"
            << "  w = 0 max |Y_eq - Y_e|     : " << err_w0_Y_max << "\n";

  if (n_w1_differs != 0) {
    std::cout << "The two entry points disagree at unit weights, to " << tol_w1
              << ": one of the wrappers converts or defaults differently.\n";
    success = false;
  }

#if ENABLE_NURATES
  // ------------------------------------------------------------------------------
  // E. the Fermi-Dirac integrals the split weights are built on
  // ------------------------------------------------------------------------------
  // Pins bns_nurates' FDI_p1/p2/p3 to the mathematics rather than to functions.hpp,
  // in the two ways that matter to eos_compose.hpp: the closed-form values at eta = 0,
  // and the reflection identities. The reflections are the load-bearing ones -- the
  // decomposition in ps_types.hpp works only because FDI reflects on the *same*
  // polynomials func_eq_weak already carries, so a minimax evaluation contributes only
  // the decaying remainder F_k(-|eta|) and leaves the exact parts exact. The recurrence
  // F_k' = k F_{k-1} is checked too, since the Jacobian of the new terms rests on it.
  //
  // Absolute accuracy against a high-precision quadrature is not measured here: double
  // arithmetic cannot supply the reference. notes/sympy_ty_perspecies_weights.py does
  // that with mpmath.
  const Real pi_e = 3.14159265358979323846;
  const Real pi2_e = pi_e*pi_e;
  const Real pi4_e = pi2_e*pi2_e;
  const Real f1_0 = pi2_e/12.0;                    // pi^2/12
  const Real f2_0 = 1.80308535473939143;           // 3 zeta(3)/2
  const Real f3_0 = 7.0*pi4_e/120.0;               // 7 pi^4/120

  const int neta = 241;
  const Real eta_lo = -30.0;
  const Real eta_hi = 30.0;
  const Real deta = (eta_hi - eta_lo)/(neta - 1);

  Real err_refl = 0.0;
  Real err_recur = 0.0;

  Kokkos::parallel_reduce("nueq_fdi",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, neta),
  KOKKOS_LAMBDA(const int &i, Real &er, Real &ec) {
    const Real eta = eta_lo + i*deta;
    const Real e2 = eta*eta;

    const Real f1p = bns_nurates::FDI_p1(eta);
    const Real f1m = bns_nurates::FDI_p1(-eta);
    const Real f2p = bns_nurates::FDI_p2(eta);
    const Real f2m = bns_nurates::FDI_p2(-eta);
    const Real f3p = bns_nurates::FDI_p3(eta);
    const Real f3m = bns_nurates::FDI_p3(-eta);

    // F_1(eta) + F_1(-eta) = pi^2/6 + eta^2/2
    // F_2(eta) - F_2(-eta) = eta (pi^2 + eta^2)/3
    // F_3(eta) + F_3(-eta) = 7 pi^4/60 + eta^2/2 (pi^2 + eta^2/2)
    const Real r1 = pi2_e/6.0 + 0.5*e2;
    const Real r2 = eta*(pi2_e + e2)/3.0;
    const Real r3 = 7.0*pi4_e/60.0 + 0.5*e2*(pi2_e + 0.5*e2);
    const Real d1 = Kokkos::fabs((f1p + f1m)/r1 - 1.0);
    const Real d2 = (r2 != 0.0) ? Kokkos::fabs((f2p - f2m)/r2 - 1.0) : 0.0;
    const Real d3 = Kokkos::fabs((f3p + f3m)/r3 - 1.0);
    er = (d1 > er) ? d1 : er;
    er = (d2 > er) ? d2 : er;
    er = (d3 > er) ? d3 : er;

    // F_k'(eta) = k F_{k-1}(eta), by a central difference wide enough that
    // cancellation does not dominate. The heavies' eta = 0 point is included.
    const Real h = 1.0e-4;
    const Real c2 = Kokkos::fabs((bns_nurates::FDI_p2(eta + h) -
                                  bns_nurates::FDI_p2(eta - h))/(2.0*h*2.0*f1p) - 1.0);
    const Real c3 = Kokkos::fabs((bns_nurates::FDI_p3(eta + h) -
                                  bns_nurates::FDI_p3(eta - h))/(2.0*h*3.0*f2p) - 1.0);
    ec = (c2 > ec) ? c2 : ec;
    ec = (c3 > ec) ? c3 : ec;
  }, Kokkos::Max<Real>(err_refl), Kokkos::Max<Real>(err_recur));

  const Real err_f1_0 = std::abs(bns_nurates::FDI_p1(0.0)/f1_0 - 1.0);
  const Real err_f2_0 = std::abs(bns_nurates::FDI_p2(0.0)/f2_0 - 1.0);
  const Real err_f3_0 = std::abs(bns_nurates::FDI_p3(0.0)/f3_0 - 1.0);

  const Real tol_fdi = 1.0e-14;
  const Real tol_recur = 1.0e-8;   // limited by the central difference, not by FDI

  std::cout << "bns_nurates Fermi-Dirac integrals, eta in [" << eta_lo << ", "
            << eta_hi << "]:\n"
            << "  |F_1(0)/(pi^2/12) - 1|      : " << err_f1_0 << "\n"
            << "  |F_2(0)/(3 zeta(3)/2) - 1|  : " << err_f2_0 << "\n"
            << "  |F_3(0)/(7 pi^4/120) - 1|   : " << err_f3_0 << "\n"
            << "  worst reflection identity   : " << err_refl << "\n"
            << "  worst F_k' = k F_{k-1}      : " << err_recur << "\n";

  if (!(err_f1_0 < tol_fdi) || !(err_f2_0 < tol_fdi) || !(err_f3_0 < tol_fdi)) {
    std::cout << "FDI_p1/p2/p3 do not reproduce their closed-form values at eta = 0 "
              << "(tolerance " << tol_fdi << ").\n";
    success = false;
  }
  if (!(err_refl < tol_fdi)) {
    std::cout << "FDI_p1/p2/p3 do not reflect on the polynomials func_eq_weak carries, "
              << "so the split-weight decomposition would not leave the exact terms "
              << "exact (tolerance " << tol_fdi << ").\n";
    success = false;
  }
  if (!(err_recur < tol_recur)) {
    std::cout << "F_k' = k F_{k-1} does not hold, so the Jacobian of the split terms "
              << "is wrong (tolerance " << tol_recur << ").\n";
    success = false;
  }

  // ------------------------------------------------------------------------------
  // F. split electron-pair weights, dw != 0
  // ------------------------------------------------------------------------------
  // The only group that exercises the appended residual and Jacobian terms at all.
  // Same construction as group A -- build a state for which (T, Y_e) is the exact root,
  // then see whether the solver finds it from a displaced guess -- but with the two
  // electron-flavour weights deliberately unequal.
  //
  // GetTrappedNeutrinos supplies the two combinations the trapped solve uses, N+ - N-
  // and J+ + J-. Their partners come from dimensionless *ratios* rather than from
  // absolute densities, which keeps the whole construction free of unit conversions and
  // makes it an independent route to the same quantity:
  //
  //     (N+ + N-)/(N+ - N-) = 3 S_2(eta) / [eta (pi^2 + eta^2)]
  //     (J+ - J-)/(J+ + J-) = D_3(eta) / P_3(eta)
  //
  // eta itself is checked against a third units-free ratio, e_nu[0]/e_nu[1], since the
  // heavies sit at eta = 0 and so carry P_3(0) = 7 pi^4/60 alone. A wrong eta would
  // otherwise look exactly like a wrong residual.
  //
  // dw = 0.15 is a little above the 0.12 the single-zone run reaches; dw = 0.30
  // (opacity ratio 4 at the peak of the weight family) is a stress case.
  const Real pi_f = 3.14159265358979323846;
  const Real pi2_f = pi_f*pi_f;
  const Real pi4_f = pi2_f*pi2_f;
  const Real p3_zero = 7.0*pi4_f/60.0;

  Primitive::UnitSystem &cunits = eos.GetCodeUnitSystem();
  Primitive::UnitSystem &eunits = eos.GetEOSUnitSystem();
  const Real T_c2e = cunits.TemperatureConversion(eunits);
  const Real mu_c2e = cunits.ChemicalPotentialConversion(eunits);

  // Below this the lepton ratio above is 0/0: the sum stays finite but is recovered by
  // dividing by a vanishing difference. The grid is neutron rich by construction, so
  // nothing should land here -- the count is reported so an all-skipped pass is visible.
  const Real eta_floor = 1.0e-6;

  const int nsplit = 2;
  const Real w_split[nsplit][2] = {{0.95, 0.65}, {0.90, 0.30}};

  for (int iw = 0; iw < nsplit; ++iw) {
    const Real w_p = w_split[iw][0];
    const Real w_m = w_split[iw][1];
    const Real wbar = 0.5*(w_p + w_m);
    const Real dw = 0.5*(w_p - w_m);

    int nf_f = 0;
    int nskip_f = 0;
    Real eT_f = 0.0;
    Real eY_f = 0.0;
    Real eeta_f = 0.0;

    Kokkos::parallel_reduce("nueq_split",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nstates),
    KOKKOS_LAMBDA(const int &idx, int &nf, int &nsk, Real &eT, Real &eY, Real &ee) {
      const int in = idx/(nY*nT);
      const int iY = (idx - in*nY*nT)/nT;
      const int iT = idx - in*nY*nT - iY*nT;

      const Real n = logs.exp2_(ln_lo + in*dln);
      const Real T = logs.exp2_(lT_lo + iT*dlT);
      Real Y[MAX_SPECIES] = {0.0};
      Y[0] = Y_lo + iY*dY;

      const Real eta = eos.GetElectronLeptonChemicalPotential(n, T, Y)*mu_c2e/
                       (T*T_c2e);
      if (!(Kokkos::fabs(eta) > eta_floor)) {
        nsk += 1;
        return;
      }
      const Real a = Kokkos::fabs(eta);
      const Real a2 = a*a;
      const Real P3 = p3_zero + 0.5*a2*(pi2_f + 0.5*a2);
      const Real S2 = 2.0*bns_nurates::FDI_p2(-a) + a*(pi2_f + a2)/3.0;
      const Real D3_abs = P3 - 2.0*bns_nurates::FDI_p3(-a);
      const Real D3 = (eta < 0.0) ? -D3_abs : D3_abs;

      Real n_nu[3], e_nu[3];
      eos.GetTrappedNeutrinos(n, T, Y, n_nu, e_nu);

      // e_nu[1] is the mu pair at eta = 0, so this ratio is P_3(eta)/P_3(0) and pins
      // eta without any unit conversion entering.
      const Real err_eta = Kokkos::fabs(e_nu[0]/e_nu[1]*p3_zero/P3 - 1.0);
      ee = (err_eta > ee) ? err_eta : ee;

      const Real ratio_L = 3.0*S2/(eta*(pi2_f + eta*eta));
      const Real ratio_E = D3/P3;

      const Real w[PEQ_NWEIGHTS] = {w_p, w_m, 1.0, w_p, w_m};

      Real Yl[MAX_SPECIES] = {0.0};
      Yl[0] = Y[0] + (wbar*n_nu[0] + dw*n_nu[0]*ratio_L)/n;
      const Real e_rhs = eos.GetEnergy(n, T, Y) +
                         wbar*e_nu[0] + dw*e_nu[0]*ratio_E + e_nu[1] + e_nu[2];

      Real T_guess = 1.4*T;
      Real Y_guess[MAX_SPECIES] = {0.0};
      Y_guess[0] = 0.7*Y[0];

      Real T_eq = 0.0;
      Real Y_eq[MAX_SPECIES] = {0.0};
      int status = -1;
      bool ok = eos.GetBetaEquilibriumPartial(n, e_rhs, Yl, w, T_eq, Y_eq,
                                              T_guess, Y_guess, &status);

      if (!ok || status == Primitive::EOSCompOSE<LogPolicy>::NUEQ_UNSUPPORTED) {
        nf += 1;
        return;
      }

      Real err_T = Kokkos::fabs(T_eq/T - 1.0);
      Real err_Y = Kokkos::fabs(Y_eq[0] - Y[0]);
      eT = (err_T > eT) ? err_T : eT;
      eY = (err_Y > eY) ? err_Y : eY;
    }, Kokkos::Sum<int>(nf_f), Kokkos::Sum<int>(nskip_f),
       Kokkos::Max<Real>(eT_f), Kokkos::Max<Real>(eY_f), Kokkos::Max<Real>(eeta_f));

    std::cout << "Split electron-pair weights (" << w_p << ", " << w_m
              << "), dw = " << dw << ", " << nstates << " states:\n"
              << "  skipped, |eta| too small : " << nskip_f << "\n"
              << "  failures                 : " << nf_f << "\n"
              << "  max eta consistency err  : " << eeta_f << "\n"
              << "  max |T_eq/T - 1|         : " << eT_f << "\n"
              << "  max |Y_eq - Y_e|         : " << eY_f << "\n";

    if (!(eeta_f < 1.0e-12)) {
      std::cout << "eta reconstructed for the split-weight construction disagrees with "
                << "the one the solver sees, so group F is not testing what it says.\n";
      success = false;
    }
    if (nskip_f >= nstates || nf_f != 0 || !(eT_f < tol_T) || !(eY_f < tol_Y)) {
      std::cout << "The split-weight solve does not recover its own exact equilibrium "
                << "(tolerances " << tol_T << " on T, " << tol_Y << " on Y_e).\n";
      success = false;
    }
  }
#endif  // ENABLE_NURATES

  if (n_w0_fail != 0 || !(res_w0_max < tol_res) || !(err_w0_T_max < tol_T) ||
      !(err_w0_Y_max < tol_Y)) {
    std::cout << "Zero weights do not return the matter state (tolerances " << tol_res
              << " on the residual, " << tol_T << " on T, " << tol_Y << " on Y_e).\n";
    success = false;
  }

  if (!success) {
    std::cout << "The test was not successful...\n";
    exit(EXIT_FAILURE);
  }

  return;
}
