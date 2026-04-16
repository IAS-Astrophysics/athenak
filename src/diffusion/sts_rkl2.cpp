//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sts_rkl2.cpp
//! \brief Host-side helpers for RKL2 super time stepping stage-count and coefficient
//! evaluation.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "diffusion/sts_rkl2.hpp"

namespace {

[[noreturn]] void STSFatalError(const char* file, int line, const std::string& msg) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

Real RawBj(int j) {
  return static_cast<Real>(j*j + j - 2) /
         static_cast<Real>(2*j*(j + 1));
}

} // namespace

namespace parabolic {

int ComputeRKL2StageCount(Real dt_sweep, Real dt_parabolic_min) {
  if (dt_sweep < 0.0) {
    STSFatalError(__FILE__, __LINE__, "dt_sweep must be non-negative in RKL2 stage-count "
                                      "calculation");
  }
  if (dt_parabolic_min <= 0.0) {
    STSFatalError(__FILE__, __LINE__, "dt_parabolic_min must be positive in RKL2 "
                                      "stage-count calculation");
  }

  Real ratio = dt_sweep/dt_parabolic_min;
  int nstages = static_cast<int>(
      std::floor(0.5*(-1.0 + std::sqrt(9.0 + 16.0*ratio))) + 1.0);

  if ((nstages % 2) == 0) {
    ++nstages;
  }
  return nstages;
}

RKL2Coefficients ComputeRKL2Coefficients(int stage, int nstages) {
  if (nstages < 2) {
    STSFatalError(__FILE__, __LINE__,
                  "nstages must be at least 2 in RKL2 coefficient calculation");
  }
  if (stage < 1 || stage > nstages) {
    STSFatalError(__FILE__, __LINE__,
                  "stage must satisfy 1 <= stage <= nstages in RKL2 coefficient "
                  "calculation");
  }

  Real bj = 0.0;
  Real bj_m1 = 0.0;
  Real bj_m2 = 0.0;

  if (stage == 1 || stage == 2) {
    bj = bj_m1 = bj_m2 = ONE_3RD;
  } else {
    bj = RawBj(stage);
    if (stage == 3) {
      bj_m1 = bj_m2 = ONE_3RD;
    } else {
      bj_m1 = RawBj(stage - 1);
      if (stage == 4) {
        bj_m2 = ONE_3RD;
      } else {
        bj_m2 = RawBj(stage - 2);
      }
    }
  }

  RKL2Coefficients coeffs;
  coeffs.muj = ((2.0*stage - 1.0)/stage)*bj/bj_m1;
  coeffs.nuj = -((stage - 1.0)/stage)*bj/bj_m2;

  Real denom = static_cast<Real>(nstages*nstages + nstages - 2);
  if (stage == 1) {
    coeffs.muj_tilde = bj*4.0/denom;
    coeffs.gammaj_tilde = 0.0;
  } else {
    coeffs.muj_tilde = coeffs.muj*4.0/denom;
    coeffs.gammaj_tilde = -(1.0 - bj_m1)*coeffs.muj_tilde;
  }

  return coeffs;
}

} // namespace parabolic
