#ifndef UTILS_LEGENDRE_ROOTS_HPP_
#define UTILS_LEGENDRE_ROOTS_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file legendre_roots.hpp
//! \brief find roots of legendre polynomial using Newton–Raphson with Chebyshev guesses

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <limits> // for std::numeric_limits
#include <utility>

#include "athena.hpp"


//----------------------------------------------------------------------------
// 1. Legendre polynomial Pn(x) via recurrence
//----------------------------------------------------------------------------
inline double Pn(int n, double x) {
  if (n == 0) return 1.0;
  if (n == 1) return x;

  double pnm2 = 1.0; // P0
  double pnm1 = x;   // P1
  double p     = 0.0;

  for (int k = 2; k <= n; ++k) {
    p = ((2.0 * k - 1.0) * x * pnm1 - (k - 1.0) * pnm2) / static_cast<double>(k);
    pnm2 = pnm1;
    pnm1 = p;
  }
  return p;
}

//----------------------------------------------------------------------------
// 2. Derivative Pn'(x):
//    (1 - x^2) Pn'(x) = n [P_{n-1}(x) - x Pn(x)]
//----------------------------------------------------------------------------
inline double PnPrime(int n, double x) {
  if (n == 0) return 0.0;
  double denom = 1.0 - x*x;
  if (std::fabs(denom) < 1.0e-30) {
    return 0.0;
  }
  double numerator = n * (Pn(n - 1, x) - x * Pn(n, x));
  return numerator / denom;
}

//----------------------------------------------------------------------------
// 3. Newton–Raphson iteration
//    x_{k+1} = x_k - Pn(x_k)/Pn'(x_k)
//    - Tolerance set to machine epsilon by default
//    - Max iteration ~20 is typically plenty (typically uses 2 or 3 iter)
//----------------------------------------------------------------------------
inline double NewtonLegendreRoot(
    int n,
    double x0,
    double tol = std::numeric_limits<double>::epsilon(),
    int max_iter = 20
) {
  for (int i = 0; i < max_iter; ++i) {
    double f  = Pn(n, x0);
    double df = PnPrime(n, x0);

    // avoid division by nearly zero
    if (std::fabs(df) < 1.0e-30) break;

    double dx = f / df;
    x0 -= dx;

    if (std::fabs(dx) < tol) {
      break;
    }

    if (i == max_iter-1) {
      std::cout << "Failed to Initialize the Gauss-Legendre Grid"<< std::endl;
      exit(EXIT_FAILURE);
    }
  }
  return x0;
}

//----------------------------------------------------------------------------
// 4. Main routine: compute n roots & weights for the Legendre polynomial
//
//    * Chebyshev initial guesses: x0_i = cos( (π*(i + 0.75)) / (n + 0.5) )
//    * Newton–Raphson to refine each guess
//    * Gauss–Legendre weight formula:  w_i = 2 / ( (1 - x_i^2) [Pn'(x_i)]^2 )
//    * Returns a 2D array [2][n]:  row 0 => roots, row 1 => weights
//----------------------------------------------------------------------------
std::vector<std::vector<Real>> RootsAndWeights(int n) {
  // temporary pair array for sorting by root
  std::vector<std::pair<double,double>> rw_pairs(n);

  for(int i = 0; i < n; ++i) {
    // Chebyshev guess
    double guess = std::cos(M_PI * (i + 0.75) / (n + 0.5));
    // Newton iteration
    double root = NewtonLegendreRoot(n, guess);

    // weight
    double dp = PnPrime(n, root);
    double weight = 2.0 / ((1.0 - root*root)*(dp*dp));

    rw_pairs[i] = std::make_pair(root, weight);
  }

  // Sort by ascending root
  std::sort(rw_pairs.begin(), rw_pairs.end(),
            [](auto &a, auto &b){ return a.first < b.first; });

  // Prepare return structure
  std::vector<std::vector<Real>> roots_weights(2, std::vector<Real>(n));
  for(int i = 0; i < n; ++i) {
    roots_weights[0][i] = rw_pairs[i].first;
    roots_weights[1][i] = rw_pairs[i].second;
  }

  return roots_weights;
}
#endif // UTILS_LEGENDRE_ROOTS_HPP_
