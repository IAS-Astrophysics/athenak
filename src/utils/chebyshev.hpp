#ifndef UTILS_CHEBYSHEV_HPP_
#define UTILS_CHEBYSHEV_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chebyshev.hpp
//  \brief functions related to chebyshev spectral method

#include "athena.hpp"
#include "globals.hpp"

// kth collocation point for chebyshev polynomials of the second kind
KOKKOS_INLINE_FUNCTION
Real ChebyshevSecondKindCollocationPoints(Real xmin, Real xmax, int N, int k) {
  // over the interval of -1 to 1
  Real x_k = std::cos( M_PI * (k + 1) / (N + 2));
  return  0.5 * ( (xmin - xmax) * x_k + (xmin + xmax));
}

// Function to compute Chebyshev polynomials of the second kind recursively
KOKKOS_INLINE_FUNCTION
Real ChebyshevSecondKindPolynomial(int k, Real x) {
  if (k == 0) return 1.0;
  if (k == 1) return 2.0 * x;
  Real Uk_2 = 1.0;           // U_0(x)
  Real Uk_1 = 2.0 * x;       // U_1(x)
  Real Uk = 0.0;
  for (int n = 2; n <= k; ++n) {
    Uk = 2.0 * x * Uk_1 - Uk_2;
    Uk_2 = Uk_1;
    Uk_1 = Uk;
  }
  return Uk;
}

#endif // UTILS_CHEBYSHEV_HPP_
