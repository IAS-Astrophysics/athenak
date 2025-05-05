#ifndef RADIATION_RADIATION_MULTI_FREQ_HPP_
#define RADIATION_RADIATION_MULTI_FREQ_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_tetrad.hpp
//  \brief helper functions for multi-frequency radiation

#include <math.h>

#include "athena.hpp"

// computes
KOKKOS_INLINE_FUNCTION
void Compute(Real nu_min, Real nu_max, int nfreq, string flag) {
  // if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);  // see cartesian_ks.hpp comments

  return;
}

#endif // RADIATION_RADIATION_MULTI_FREQ_HPP_
