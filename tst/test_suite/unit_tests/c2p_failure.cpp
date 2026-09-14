// Native C2P failure flags and returned-state checks, on host and execution space.
// Build/run with tst/test_c2p_failure.py and an existing CMake build.
#include <cmath>
#include <iostream>
#include <limits>

#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

using Solver = Primitive::PrimitiveSolver<Primitive::IdealGas, Primitive::ResetFloor>;
using Error = Primitive::Error;

struct CheckResult {
  Primitive::SolverResult status;
  Real state_error;
};

// Flat spatial metric; undensitized conserved moments; primitive velocity is Wv.
// Code and EOS units coincide, with baryon mass one. No table is required.
KOKKOS_INLINE_FUNCTION
CheckResult Evaluate(const Solver &solver, const int test, const Real nan) {
  Real gd[NSPMETRIC] = {1, 0, 0, 1, 0, 1};
  Real gu[NSPMETRIC] = {1, 0, 0, 1, 0, 1};
  Real b[NMAG] = {};
  Real initial[NPRIM] = {}, prim[NPRIM] = {}, cons[NCONS] = {};
  initial[PRH] = 1.0;
  initial[PVX] = test == 3 ? 2.0 : (test == 8 ? 1.0 : 1.0 / 8.0);
  initial[PTM] = 1.0 / 8.0;
  initial[PYF] = 1.0 / 4.0;
  const auto &eos = solver.GetEOS();
  initial[PPR] = eos.GetPressure(initial[PRH], initial[PTM], &initial[PYF]);
  if (test == 3 || test == 7) b[IBY] = 1.0;
  solver.PrimToCon(initial, cons, b, gd);
  if (test == 2) cons[CSX] = nan;
  if (test == 4) cons[CTA] = -1.0;
  if (test == 6) cons[CYD] = nan;

  CheckResult out{};
  out.status = solver.ConToPrim(prim, cons, b, gd, gu);
  // Independently construct the expected primitive state. On success it is the
  // initial state; on failure ResetFloor must return its configured atmosphere.
  Real expected[NPRIM] = {};
  if (test == 0) {
    for (int a = 0; a < NPRIM; ++a) expected[a] = initial[a];
  } else {
    expected[PRH] = eos.GetDensityFloor();
    expected[PTM] = eos.GetTemperatureFloor();
    expected[PYF] = eos.GetSpeciesAtmosphere(0);
    expected[PPR] = eos.GetPressure(expected[PRH], expected[PTM], &expected[PYF]);
  }
  Real expected_cons[NCONS] = {};
  solver.PrimToCon(expected, expected_cons, b, gd);
  // Check each primitive and conserved component at its own scale. The tests
  // exercise native point recovery, not the mesh wrapper's actual array writes.
  for (int a = 0; a < NPRIM; ++a) {
    const Real scale = fmax(fabs(expected[a]), eos.GetDensityFloor());
    const Real error = fabs(prim[a] - expected[a]) / scale;
    if (!isfinite(error)) { out.state_error = 1.0; return out; }
    out.state_error = fmax(out.state_error, error);
  }
  for (int a = 0; a < NCONS; ++a) {
    const Real scale = fmax(fabs(expected_cons[a]), eos.GetDensityFloor());
    const Real error = fabs(cons[a] - expected_cons[a]) / scale;
    if (!isfinite(error)) { out.state_error = 1.0; return out; }
    out.state_error = fmax(out.state_error, error);
  }
  return out;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  {
    constexpr int ntest = 9;
    const char *names[ntest] = {"success", "root-failure", "nonfinite-momentum",
        "bracketing-failure", "conserved-floor-failure", "root-failure-adjust-off",
        "nonfinite-species", "root-failure-after-magnetization", "primitive-floor-failure"};
    const Error errors[ntest] = {Error::SUCCESS, Error::NO_SOLUTION, Error::NANS_IN_CONS,
        Error::BRACKETING_FAILED, Error::CONS_FLOOR, Error::NO_SOLUTION,
        Error::NANS_IN_CONS, Error::NO_SOLUTION, Error::PRIM_FLOOR};
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real tol = 256 * std::numeric_limits<Real>::epsilon();
    for (int test = 0; test < ntest; ++test) {
      Solver solver;
      auto &eos = solver.GetEOSMutable();
      eos.SetNSpecies(1);
      eos.SetDensityFloor(1e-6);
      eos.SetTemperatureFloor(test == 8 ? 1.0 / 4.0 : 1e-6);
      eos.SetSpeciesAtmosphere(1.0 / 2.0, 0);
      eos.SetConservedFloorFailure(test == 4);
      eos.SetPrimitiveFloorFailure(test == 8);
      eos.SetAdjustConserved(test != 5);
      if (test == 7) eos.SetMaximumMagnetization(1.0 / 2.0);
      // Deliberately exhaust the selected solve, without changing its tolerance.
      if (test == 1 || test == 3 || test == 5 || test == 7) {
        solver.GetRootSolverMutable().iterations = 1;
      }
      const CheckResult host = Evaluate(solver, test, nan);
      Kokkos::View<CheckResult *> device("c2p_failure", 1);
      Kokkos::parallel_for("c2p_failure", Kokkos::RangePolicy<>(0, 1),
                          KOKKOS_LAMBDA(const int i) {
                            device(i) = Evaluate(solver, test, nan);
                          });
      const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
      const CheckResult results[2] = {host, mirror(0)};
      for (int side = 0; side < 2; ++side) {
        const auto &r = results[side];
        const bool pass = r.status.error == errors[test] &&
                          r.status.cons_floor == (test == 4) &&
                          r.status.prim_floor == (test == 8) &&
                          r.status.cons_adjusted == (test != 0) &&
                          std::isfinite(r.state_error) && r.state_error <= tol;
        std::cout << (pass ? "PASS " : "FAIL ") << names[test]
                  << " side=" << (side == 0 ? "host" : Kokkos::DefaultExecutionSpace::name())
                  << " error=" << static_cast<int>(r.status.error)
                  << " expected=" << static_cast<int>(errors[test])
                  << " cons_floor=" << r.status.cons_floor
                  << " prim_floor=" << r.status.prim_floor
                  << " cons_adjusted=" << r.status.cons_adjusted
                  << " state_error=" << r.state_error << '\n';
        failed += !pass;
      }
    }
  }
  Kokkos::finalize();
  return failed ? 1 : 0;
}
