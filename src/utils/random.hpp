#ifndef UTILS_RANDOM_HPP_
#define UTILS_RANDOM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file random.cpp
//  \brief Random number generators (that can be included in Kokkos parallel for regions)

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn Ran2
//! \brief  Extracted from the Numerical Recipes in C (version 2) code. Modified
//!  to use doubles instead of floats. -- T. A. Gardiner -- Aug. 12, 2003
//!
//! Long period (> 2 x 10^{18}) random number generator of L'Ecuyer with Bays-Durham
//! shuffle and added safeguards.  Returns a uniform random deviate between 0.0 and 1.0
//! (exclusive of the endpoint values).  Call with idum = a negative integer to
//! initialize; thereafter, do not alter idum between successive deviates in a sequence.
//! RNMX should appriximate the largest floating-point value that is less than 1.

#define NTAB 32

typedef struct RNG_State {
  int64_t idum;
  int64_t idum2;
  int64_t iy;
  int64_t iv[NTAB];
  // For Box-Mueller gaussian generation
  int iset;
  double gset;
} RNG_State;

#define IMR1 2147483563
#define IMR2 2147483399
#define AM (1.0/IMR1)
#define IMM1 (IMR1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NDIV (1+IMM1/NTAB)
#define RNMX (1.0-DBL_EPSILON)

KOKKOS_INLINE_FUNCTION
double RanSt(RNG_State *state) {
  int j;
  int64_t k;
  double temp;
  int64_t *idum;
  idum = &(state->idum);
  if (*idum <= 0) { // Initialize
    state->idum2 = 123456789;
    state->iy = 0;
    if (-(*idum) < 1) {
      *idum=1; // Be sure to prevent idum = 0
    } else {
      *idum = -(*idum);
    }
    state->idum2=(*idum);
    for (j=NTAB+7; j>=0; j--) { // Load the shuffle table (after 8 warm-ups)
      k=(*idum)/IQ1;
      *idum=IA1*(*idum-k*IQ1)-k*IR1;
      if (*idum < 0) *idum += IMR1;
      if (j < NTAB) state->iv[j] = *idum;
    }
    state->iy=state->iv[0];
  }
  k=(*idum)/IQ1;                               // Start here when not initializing
  *idum=IA1*(*idum-k*IQ1)-k*IR1;               // Compute idum=(IA1*idum) % IMR1 without
  if (*idum < 0) *idum += IMR1;                // overflows by Schrage's method
  k=state->idum2/IQ2;
  state->idum2=IA2*(state->idum2-k*IQ2)-k*IR2; // Compute idum2=(IA2*idum) % IMR2 likewise
  if (state->idum2 < 0) state->idum2 += IMR2;
  j=static_cast<int>(state->iy/NDIV);          // Will be in the range 0...NTAB-1
  state->iy=state->iv[j]-state->idum2;         // Here idum is shuffled, idum and idum2
  state->iv[j] = *idum;                        // are combined to generate output
  if (state->iy < 1) state->iy += IMM1;
  if ((temp=AM*state->iy) > RNMX) {
    return RNMX; // No endpoint values
  } else {
    return temp;
  }
}

KOKKOS_INLINE_FUNCTION
static Real Ran2(int64_t *idum) {
  int j;
  int64_t k;
  static int64_t idum2=123456789;
  static int64_t iy=0;
  static int64_t iv[NTAB];

  Real temp;

  if (*idum <= 0) { // Initialize
    if (-(*idum) < 1)
      *idum=1; // Be sure to prevent idum = 0
    else
      *idum = -(*idum);
    idum2=(*idum);
    for (j=NTAB+7; j>=0; j--) { // Load the shuffle table (after 8 warm-ups)
      k=(*idum)/IQ1;
      *idum=IA1*(*idum-k*IQ1)-k*IR1;
      if (*idum < 0) *idum += IMR1;
      if (j < NTAB) iv[j] = *idum;
    }
    iy=iv[0];
  }
  k=(*idum)/IQ1;                 // Start here when not initializing
  *idum=IA1*(*idum-k*IQ1)-k*IR1; // Compute idum=(IA1*idum) % IMR1 without
  if (*idum < 0) *idum += IMR1;   // overflows by Schrage's method
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2; // Compute idum2=(IA2*idum) % IMR2 likewise
  if (idum2 < 0) idum2 += IMR2;
  j=static_cast<int>(iy/NDIV);              // Will be in the range 0...NTAB-1
  iy=iv[j]-idum2;                // Here idum is shuffled, idum and idum2
  iv[j] = *idum;                 // are combined to generate output
  if (iy < 1)
    iy += IMM1;

  if ((temp=AM*iy) > RNMX)
    return RNMX; // No endpoint values
  else
    return temp;
}

#undef IMR1
#undef IMR2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NDIV
#undef RNMX

//----------------------------------------------------------------------------------------
//! \fn RanGaussian

KOKKOS_INLINE_FUNCTION
static Real RanGaussianSt(RNG_State *state) {
  static int iset = 0;
  static double gset;
  double fac, rsq, v1, v2;
  if (state->idum < 0) iset = 0;
  if (iset == 0) {
    do {
      v1 = 2.0 * RanSt(state) - 1.0;
      v2 = 2.0 * RanSt(state) - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >=1.0 || rsq == 0.0);
    fac = sqrt(-2.0*log(rsq)/rsq);
    gset = v1*fac;
    iset = 1;
    return v2*fac;
  } else {
    iset = 0;
    return gset;
  }
}

KOKKOS_INLINE_FUNCTION
static Real RanGaussian(int64_t *idum) {
  static int32_t iset = 0;
  static Real gset;
  Real fac, rsq, v1, v2;
  if (*idum < 0) iset = 0;
  if (iset == 0) {
    do {
      v1 = 2.0 * Ran2(idum) - 1.0;
      v2 = 2.0 * Ran2(idum) - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >=1.0 || rsq == 0.0);
    fac = sqrt(-2.0*log(rsq)/rsq);
    gset = v1 * fac;
    iset = 1;
    return v2*fac;
  } else {
    iset = 0;
    return gset;
  }
}

#endif // UTILS_RANDOM_HPP_
