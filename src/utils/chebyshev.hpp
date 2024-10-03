#ifndef UTILS_CHEBYSHEV_HPP
#define UTILS_CHEBYSHEV_HPP

#include "athena.hpp"
#include "globals.hpp"
// TODO (HZ): Add function to compute the collocation points for Chebyshev polynomials of the second kind


// Function to compute Chebyshev polynomials of the second kind recursively
Real chebyshevSecondKindPolynomial(int k, Real x) {
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

#endif // UTILS_CHEBYSHEV_HPP
