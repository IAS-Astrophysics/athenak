//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_helmholtz.cpp
//! \brief Unit test of the EOS<> wrapper getters added for the nuclear network coupling
//! with the Helmholtz policy: GetTemperatureFromEntropy, GetDensityFromPressure*,
//! GetSpecificHeat, and their unit conversion. Round trips over a (n, T, Y) grid inside
//! the table range, in geometric_solar units and on a cgs copy of the EOS.

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

void ProblemGenerator::EOSHelmholtz(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr ||
      pin->GetString("mhd", "dyn_eos").compare("helmholtz") != 0 ||
      pin->GetString("mhd", "dyn_error").compare("reset_floor") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSHelmholtz unit test needs DynGRMHD with mhd/dyn_eos = helmholtz "
              << "and mhd/dyn_error = reset_floor\n";
    exit(EXIT_FAILURE);
  }

  auto &eos = static_cast<dyngr::DynGRMHDPS<Primitive::EOSHelmholtz,
                                            Primitive::ResetFloor>*>(pmbp->pdyngr)
              ->eos.ps.GetEOSMutable();
  auto eos_cgs = eos;  // shallow copy, shares the table
  eos_cgs.SetCodeUnitSystem(Primitive::MakeCGS());

  // Conversion factors geometric_solar -> cgs for the unit check
  const Primitive::UnitSystem gs = Primitive::MakeGeometricSolar();
  const Primitive::UnitSystem cgs = Primitive::MakeCGS();
  const Real Tc = gs.TemperatureConversion(cgs);
  const Real EM = gs.EnergyConversion(cgs)/gs.MassConversion(cgs);
  const Real SM = gs.EntropyConversion(cgs)/gs.MassConversion(cgs);
  // Table temperatures are in EOS units; geometric_solar code temperature is MeV too
  const Real T_eos2code = Primitive::MakeNuclear().TemperatureConversion(gs);

  // Grid in the physically meaningful window of the table, <problem>/nmin..nmax [fm^-3]
  // (defaults ~1.7e3..1.7e11 g/cm^3) and Tmin..Tmax [code units, defaults 1e-5..10 MeV,
  // 1e5..1e11 K], shrunk 5% in ln n and ln T at both ends. Outside it the electron
  // table runs to supranuclear density and to 1e13 K, where nuclei have dissolved and
  // pairs or radiation make P independent of n. At low n and T the Coulomb positivity
  // guard makes P(n) non-monotonic; such points are accepted below via the inversion's
  // contract (P(n1) = p) and counted.
  const int nn = pin->GetOrAddInteger("problem", "nn", 64);
  const int nT = pin->GetOrAddInteger("problem", "nT", 64);
  const int nY = 4;
  const Real nmin = pin->GetOrAddReal("problem", "nmin", 1.0e-12);
  const Real nmax = pin->GetOrAddReal("problem", "nmax", 1.0e-4);
  const Real Tmin = pin->GetOrAddReal("problem", "Tmin", 1.0e-5);
  const Real Tmax = pin->GetOrAddReal("problem", "Tmax", 10.0);
  Real lnmin = log(fmax(eos.GetMinimumDensity(), nmin));
  Real lnmax = log(fmin(eos.GetMaximumDensity(), nmax));
  Real lTmin = log(fmax(eos.GetMinimumTemperature()*T_eos2code, Tmin));
  Real lTmax = log(fmin(eos.GetMaximumTemperature()*T_eos2code, Tmax));
  const Real shrink = 0.05;
  const Real dn_tot = lnmax - lnmin, dT_tot = lTmax - lTmin;
  lnmin += shrink*dn_tot; lnmax -= shrink*dn_tot;
  lTmin += shrink*dT_tot; lTmax -= shrink*dT_tot;
  const Real dln = (lnmax - lnmin)/(nn - 1);
  const Real dlT = (lTmax - lTmin)/(nT - 1);

  const Real tol = static_cast<Real>(std::numeric_limits<float>::epsilon());
  const Real tol_cv = 0.1;      // table derivative vs secant of the interpolant
  const Real tol_units = 1e-12;
  const Real tol_p = 1e-12;     // pressure residual of the density inversions

  bool success_all = true;
  Real max_cv_dev = 0.0;
  int n_contract = 0;  // n(p,.) points off the round trip but with P(n1) = p
  int n_unreach = 0;   // NaN results of the density inversions (see the getters)
  Kokkos::parallel_reduce("ut_helmholtz",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nn*nT*nY),
  KOKKOS_LAMBDA(const int &idx, bool &success, Real &cv_dev, int &ncon, int &nunr) {
    const int in = idx/(nT*nY);
    const int iT = (idx - in*nT*nY)/nY;
    const int iY = idx - in*nT*nY - iT*nY;

    Real Y[MAX_SPECIES] = {0.0};
    if (iY == 0) { Y[0] = 1.0; }                                   // pure He4
    else if (iY == 1) { Y[1] = 1.0; }                              // pure C12
    else if (iY == 2) { Y[2] = 1.0; }                              // pure O16 (eb < 0)
    else { Y[0] = 1.0/3.0; Y[1] = 1.0/3.0; Y[2] = 1.0/3.0; }        // equal thirds

    const Real n = exp(lnmin + in*dln);
    const Real T = exp(lTmin + iT*dlT);
    auto fail = [&](const char *what, Real val, Real err) {
      Kokkos::printf("%s failed at n = %.10e T = %.10e iY = %d: value %.10e err %.3e\n", // NOLINT
                     what, n, T, iY, val, err);
      success = false;
    };
    // The Coulomb entropy fit jumps at Gamma = 1 (Yakovlev & Shalybkov, 0.007 kB per
    // ion), so an s inside the gap has no exact T. The inversion is accepted when the
    // target lies between the entropies just below and just above the returned T.
    auto s_bracketed = [&](Real Tt, Real st) {
      Real s_lo = eos.GetEntropy(n, Tt*(1.0 - 1.0e-8), Y);
      Real s_hi = eos.GetEntropy(n, Tt*(1.0 + 1.0e-8), Y);
      Real width = fabs(s_hi - s_lo) + tol*fabs(st);
      return fabs(st - 0.5*(s_lo + s_hi)) <= 0.5*width;
    };

    // (a) T -> s -> T
    Real s = eos.GetEntropy(n, T, Y);
    Real T1 = eos.GetTemperatureFromEntropy(n, s, Y);
    Real err = fabs(T1/T - 1.0);
    if (!(err <= tol) && !s_bracketed(T1, s)) fail("T(n,s)", T1, err);

    // (b) n -> p -> n at fixed T. Off the round trip (ill-conditioned where radiation
    // dominates, non-monotonic P(n) where the Coulomb guard switches) the inversion
    // must still satisfy its contract P(n1, T) = p.
    Real p = eos.GetPressure(n, T, Y);
    Real n1 = eos.GetDensityFromPressureTemperature(p, T, Y);
    err = fabs(n1/n - 1.0);
    if (!isfinite(n1)) {
      nunr++;  // converged onto a jump of P(n), no solution in that cell
    } else if (!(err <= tol)) {
      Real errp = fabs(eos.GetPressure(n1, T, Y)/p - 1.0);
      if (!(errp <= tol_p)) fail("n(p,T): P(n1,T) != p", n1, errp);
      ncon++;
    }

    // (c) (p, s) -> (n, T). NaN means the isentrope leaves the table at the solution.
    // At the entropy jump the solution is n2 with P(n2, T2) = p and T2 at the jump, so
    // off the round trip check the contract.
    Real T2;
    Real n2 = eos.GetDensityFromPressureEntropy(p, s, Y, &T2);
    err = fabs(n2/n - 1.0);
    if (!isfinite(n2)) {
      nunr++;
    } else if (!(err <= tol)) {
      ncon++;
      Real p2 = eos.GetPressure(n2, T2, Y);
      Real errp = fabs(p2/p - 1.0);
      if (!(errp <= tol_p)) fail("n(p,s): P(n2,T2) != p", n2, errp);
      // the n used inside the inversion is n2, so bracket s at (n2, T2)
      Real s_lo = eos.GetEntropy(n2, T2*(1.0 - 1.0e-8), Y);
      Real s_hi = eos.GetEntropy(n2, T2*(1.0 + 1.0e-8), Y);
      if (!(fabs(s - 0.5*(s_lo + s_hi)) <= 0.5*(fabs(s_hi - s_lo) + tol*fabs(s)))) {
        fail("n(p,s): s not bracketed", T2, fabs(T2/T - 1.0));
      }
    } else {
      err = fabs(T2/T - 1.0);
      if (!(err <= tol) && !s_bracketed(T2, s)) fail("T(p,s)", T2, err);
    }

    // (d) cv vs central finite difference of eps in T
    Real cv = eos.GetSpecificHeat(n, T, Y);
    Real dT = 1.0e-3*T;
    Real cv_fd = (eos.GetSpecificInternalEnergy(n, T + dT, Y) -
                  eos.GetSpecificInternalEnergy(n, T - dT, Y))/(2.0*dT);
    err = fabs(cv/cv_fd - 1.0);
    cv_dev = fmax(cv_dev, err);
    if (!(err <= tol_cv) || !(cv > 0.0)) fail("cv vs FD", cv, err);

    // (e) cgs copy against the geometric_solar values through the unit factors
    Real cv_cgs = eos_cgs.GetSpecificHeat(n, T*Tc, Y);
    err = fabs(cv_cgs/(cv*EM/Tc) - 1.0);
    if (!(err <= tol_units)) fail("cv units", cv_cgs, err);
    Real T_cgs = eos_cgs.GetTemperatureFromEntropy(n, s*SM, Y);
    err = fabs(T_cgs/(T1*Tc) - 1.0);
    if (!(err <= tol_units)) fail("T(n,s) units", T_cgs, err);
  }, Kokkos::LAnd<bool>(success_all), Kokkos::Max<Real>(max_cv_dev),
     Kokkos::Sum<int>(n_contract), Kokkos::Sum<int>(n_unreach));

  if (global_variable::my_rank == 0) {
    std::cout << "EOSHelmholtz unit test: " << nn*nT*nY
              << " points, max |cv/cv_fd - 1| = " << max_cv_dev
              << ", n(p,.) accepted via P residual: " << n_contract
              << ", n(p,.) unreachable: " << n_unreach << std::endl;
  }
  // Every grid point is a solution with T inside the table, so NaN or contract-only
  // results must stay rare (multi-root Coulomb-guard points). A getter that always
  // returned NaN, or always a wrong branch, must not pass.
  const int max_rare = nn*nT*nY/100;
  if (n_unreach > max_rare || n_contract > max_rare) {
    std::cout << "Too many unreachable or contract-only points (limit " << max_rare
              << ")\n";
    success_all = false;
  }
  if (!success_all) {
    std::cout << "The test was not successful...\n";
    exit(EXIT_FAILURE);
  }
  std::cout << "Test Passed!\n";

  // Minkowski ADM variables, so the run does not end in c2p failures
  pmbp->padm->SetADMVariables(pmbp);
  return;
}
