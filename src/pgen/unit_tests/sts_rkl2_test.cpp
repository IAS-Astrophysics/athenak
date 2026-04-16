//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sts_rkl2_test.cpp
//! \brief Unit test for the host-side RKL2 STS math helpers.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "diffusion/sts_rkl2.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace {

void CheckEqual(const char* label, int actual, int expected) {
  if (actual != expected) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << label << " mismatch: expected " << expected << ", got "
              << actual << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void CheckNear(const char* label, Real actual, Real expected, Real tol) {
  if (std::abs(actual - expected) > tol) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << label << " mismatch: expected " << expected << ", got "
              << actual << " (tol=" << tol << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  (void) pin;
  (void) restart;

  CheckEqual("nstages(ratio=0)",
             parabolic::ComputeRKL2StageCount(0.0, 1.0), 3);
  CheckEqual("nstages(ratio=1)",
             parabolic::ComputeRKL2StageCount(1.0, 1.0), 3);
  CheckEqual("nstages(ratio=4)",
             parabolic::ComputeRKL2StageCount(4.0, 1.0), 5);
  CheckEqual("nstages(ratio=10)",
             parabolic::ComputeRKL2StageCount(10.0, 1.0), 7);

  constexpr int nstages = 5;
  const Real tol = static_cast<Real>(100.0)*std::numeric_limits<Real>::epsilon();

  const Real muj_ref[nstages] = {
    1.0,
    1.5,
    2.0833333333333335,
    1.89,
    1.8666666666666667
  };
  const Real nuj_ref[nstages] = {
    0.0,
    -0.5,
    -0.8333333333333334,
    -1.0125,
    -0.896
  };
  const Real muj_tilde_ref[nstages] = {
    0.047619047619047616,
    0.21428571428571427,
    0.29761904761904767,
    0.27,
    0.26666666666666666
  };
  const Real gamma_ref[nstages] = {
    0.0,
    -0.14285714285714288,
    -0.19841269841269848,
    -0.1575,
    -0.14666666666666667
  };

  for (int stage = 1; stage <= nstages; ++stage) {
    auto coeffs = parabolic::ComputeRKL2Coefficients(stage, nstages);
    CheckNear("muj", coeffs.muj, muj_ref[stage - 1], tol);
    CheckNear("nuj", coeffs.nuj, nuj_ref[stage - 1], tol);
    CheckNear("muj_tilde", coeffs.muj_tilde, muj_tilde_ref[stage - 1], tol);
    CheckNear("gammaj_tilde", coeffs.gammaj_tilde, gamma_ref[stage - 1], tol);
  }

  std::cout << "STS RKL2 helper test passed" << std::endl;
}
