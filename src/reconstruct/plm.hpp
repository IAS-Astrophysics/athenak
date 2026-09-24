#ifndef RECONSTRUCT_PLM_HPP_
#define RECONSTRUCT_PLM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file plm.hpp
//! \brief  piecewise linear reconstruction implemented as inline functions
//! This version only works with uniform mesh spacing

#include <math.h>
#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PLM()
//! \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1.

KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1,
         Real &ql_ip1, Real &qr_i) {
  // compute L/R slopes
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
  Real dq2 = dql*dqr;
  Real dqm = dq2/(dql + dqr);
  //Real dqm = 0.5*fmin(fabs(dql), fabs(dqr));
  if (dq2 <= 0.0) dqm = 0.0;

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + dqm;
  qr_i   = q_i - dqm;
  return;
}
#endif // RECONSTRUCT_PLM_HPP_
