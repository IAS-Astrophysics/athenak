#ifndef UTILS_LEGENDRE_ROOTS_HPP_
#define UTILS_LEGENDRE_ROOTS_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file legendre_roots.hpp
//! \brief find roots of legendre polynomial using Newton–Raphson with Chebyshev guesses

#include <iostream>
#include <cmath>
#include <vector>

#include "athena.hpp"

//-------------------------------------------------------------------------
// 1. Basic Legendre polynomials
//-------------------------------------------------------------------------
/* P0(x) */
inline double P0(double x) {
  return 1.0;
}

/* P1(x) */
inline double P1(double x) {
  return x;
}

/* Nth Legendre Polynomial, Pn(x) */
double Pn(int n, double x) {
  if (n == 0) {
    return P0(x);
  } else if (n == 1) {
    return P1(x);
  } else {
    //Use the recurrence relation:
    //   P_n(x) = ( (2n-1)*x*P_{n-1}(x) - (n-1)*P_{n-2}(x) ) / n
    return ((2.0*n - 1.0)*x*Pn(n - 1, x) - (n - 1.0)*Pn(n - 2, x))/n;
  }
}

//-------------------------------------------------------------------------
// 2. Derivative of the Legendre polynomial, Pn'(x)
//
// A known useful identity is:
//   (1 - x^2) * Pn'(x) = n * [ P_{n-1}(x) - x * Pn(x) ]
//   => Pn'(x) = n/(1-x^2) * [P_{n-1}(x) - x*Pn(x)]
//-------------------------------------------------------------------------
double PnPrime(int n, double x) {
  if (n == 0) {
    return 0.0;
  } else {
    return (n/(1.0 - x*x)) * (Pn(n - 1, x) - x*Pn(n, x));
  }
}

//-------------------------------------------------------------------------
// 3. Newton–Raphson iteration for root of Pn(x)
//
//    x_{k+1} = x_k - Pn(n,x_k) / Pn'(n,x_k)
//-------------------------------------------------------------------------
double NewtonLegendreRoot(int n, double x0, double tol, int max_iter) {
  for (int i = 0; i < max_iter; ++i) {
    double f  = Pn(n, x0);
    double df = PnPrime(n, x0);

    // Avoid division by very small derivative
    if (std::fabs(df) < 1.0e-30) {
      break;  // or return x0 as is
    }

    double dx = f/df;
    x0 -= dx;

    if (std::fabs(dx) < tol) {
      break;
    }
  }
  return x0;
}

//-------------------------------------------------------------------------
// 4. Lagrange terms
//-------------------------------------------------------------------------
double Li(int n, const std::vector<double> &x, int i, double X) {
  double prod = 1.0;
  for (int j = 0; j <= n; j++) {
    if (j != i) {
      prod *= (X - x[j])/(x[i] - x[j]);
    }
  }
  return prod;
}

//-------------------------------------------------------------------------
// 5. Simpson's rule to compute the integral for weights via Lagrange
//    polynomials
//-------------------------------------------------------------------------
double Ci(int i, int n, const std::vector<double> &x,
          double a, double b, int N) {
  double h = (b - a) / N;
  double sum = 0.0;
  // Evaluate interior points
  for (int j = 1; j < N; j++) {
    double X = a + j * h;
    if (j % 2 == 0) {
      sum += 2.0 * Li(n - 1, x, i, X);
    } else {
      sum += 4.0 * Li(n - 1, x, i, X);
    }
  }
  // Endpoints
  double Fa = Li(n - 1, x, i, a);
  double Fb = Li(n - 1, x, i, b);

  return (h / 3.0) * (Fa + Fb + sum);
}

//-------------------------------------------------------------------------
// 6. Main routine: find roots & weights of the Legendre polynomial
//    using Newton–Raphson with Chebyshev initial guesses
//-------------------------------------------------------------------------
std::vector<std::vector<Real>> RootsAndWeights(int n) {
  // Storage: [0] = roots, [1] = weights
  std::vector<std::vector<Real>> roots_weights(2, std::vector<Real>(n));
  std::vector<double> xi(n);

  // Parameters for Newton iteration
  double tol      = 1e-14;
  int    max_iter = 100000;

  // Chebyshev-type initial guesses for the i-th root
  // Common choice:
  //    x0_i = cos( (pi*(i + 0.75)) / (n + 0.5) )
  // for i = 0..n-1
  for (int i = 0; i < n; ++i) {
    double guess = std::cos( M_PI * ( i + 0.75 ) / (n + 0.5) );
    double root  = NewtonLegendreRoot(n, guess, tol, max_iter);

    xi[i] = root;
    roots_weights[0][i] = root;
  }

  // Compute weights (using the original Lagrange-based Ci or direct formula).
  // Here we keep your original Lagrange-based approach:
  for (int i = 0; i < n; i++) {
    // Adjust N for integration as you see fit.
    // For a large n, you might not need extremely large N
    // if the function is well-behaved.
    // But here we keep your original 60000 intervals.
    roots_weights[1][i] = Ci(i, n, xi, -1.0, 1.0, 60000);
  }

  return roots_weights;
}

#endif // UTILS_LEGENDRE_ROOTS_HPP_
