//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ran2.cpp
//  \brief Random number generator with random state

// C headers

// C++ headers
#include <cfloat>
#include <iostream>

// Athena++ headers
#include "athena.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \fn double ran2(std::int64_t *idum)
//  \brief  Extracted from the Numerical Recipes in C (version 2) code. Modified
//   to use doubles instead of floats. -- T. A. Gardiner -- Aug. 12, 2003
//
// Long period (> 2 x 10^{18}) random number generator of L'Ecuyer with Bays-Durham
// shuffle and added safeguards.  Returns a uniform random deviate between 0.0 and 1.0
// (exclusive of the endpoint values).  Call with idum = a negative integer to
// initialize; thereafter, do not alter idum between successive deviates in a sequence.
// RNMX should appriximate the largest floating-point value that is less than 1.

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

double ran2st(RNG_State *state) {
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
#undef NTAB
#undef NDIV
#undef RNMX
