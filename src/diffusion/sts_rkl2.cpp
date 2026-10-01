//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sts_rkl2.cpp
//! \brief Host-side RKL2 stage-count and coefficient helpers.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "diffusion/sts_rkl2.hpp"

namespace {

[[noreturn]] void STSFatalError(const char *file, int line, const std::string &msg) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

Real RawBj(int j) {
  Real jr = static_cast<Real>(j);
  return (jr*jr + jr - 2.0)/(2.0*jr*(jr + 1.0));
}

long double RKL2MaxRatio(int nstages) {
  long double sr = static_cast<long double>(nstages);
  return 0.25L*(sr*sr + sr - 2.0L);
}

} // namespace

namespace parabolic {

int ComputeRKL2StageCount(Real dt_sweep, Real dt_parabolic_min) {
  if (dt_sweep < 0.0) {
    STSFatalError(__FILE__, __LINE__,
                  "dt_sweep must be non-negative in RKL2 stage-count calculation");
  }
  if (dt_parabolic_min <= 0.0) {
    STSFatalError(__FILE__, __LINE__,
                  "dt_parabolic_min must be positive in RKL2 stage-count calculation");
  }

  long double ratio = static_cast<long double>(dt_sweep)/
                      static_cast<long double>(dt_parabolic_min);
  constexpr int min_nstages = 3;
  constexpr int max_nstages = std::numeric_limits<int>::max();
  if (!std::isfinite(ratio) || ratio > RKL2MaxRatio(max_nstages)) {
    STSFatalError(__FILE__, __LINE__,
                  "required RKL2 stage count exceeds the supported integer range");
  }

  long double estimate = 0.5L*(-1.0L + std::sqrt(9.0L + 16.0L*ratio));
  int nstages = static_cast<int>(std::ceil(estimate));
  nstages = std::max(nstages, min_nstages);
  if ((nstages % 2) == 0) {
    ++nstages;
  }

  // Correct roundoff in the inverse estimate and return the smallest sufficient odd s.
  while (RKL2MaxRatio(nstages) < ratio) {
    if (nstages > max_nstages - 2) {
      STSFatalError(__FILE__, __LINE__,
                    "required RKL2 stage count exceeds the supported integer range");
    }
    nstages += 2;
  }
  while (nstages > min_nstages && RKL2MaxRatio(nstages - 2) >= ratio) {
    nstages -= 2;
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

  constexpr Real one_third = 1.0/3.0;
  Real bj = 0.0;
  Real bj_m1 = 0.0;
  Real bj_m2 = 0.0;

  if (stage == 1 || stage == 2) {
    bj = bj_m1 = bj_m2 = one_third;
  } else {
    bj = RawBj(stage);
    if (stage == 3) {
      bj_m1 = bj_m2 = one_third;
    } else {
      bj_m1 = RawBj(stage - 1);
      bj_m2 = (stage == 4) ? one_third : RawBj(stage - 2);
    }
  }

  RKL2Coefficients coeffs;
  coeffs.muj = ((2.0*stage - 1.0)/stage)*bj/bj_m1;
  coeffs.nuj = -((stage - 1.0)/stage)*bj/bj_m2;

  Real nstages_r = static_cast<Real>(nstages);
  Real denom = nstages_r*nstages_r + nstages_r - 2.0;
  if (stage == 1) {
    coeffs.muj_tilde = bj*4.0/denom;
  } else {
    coeffs.muj_tilde = coeffs.muj*4.0/denom;
    coeffs.gammaj_tilde = -(1.0 - bj_m1)*coeffs.muj_tilde;
  }
  return coeffs;
}

} // namespace parabolic
