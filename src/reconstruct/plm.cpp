//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file plm.cpp
//  \brief  piecewise linear reconstruction implemented as inline function.
//  This version only works with uniform mesh spacing

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinear()
//  \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i) over [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinear(TeamMember_t const &member, const int il, const int iu,
                     const AthenaArray2DSlice<Real> &q,
                     AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr)
{
  int nvar = q.extent_int(0);
  for (int n=0; n<nvar; ++n) {
    // compute L/R slopes for each variable
    par_for_inner(member, il, iu, [&](const int i)
    { 
      Real dql = (q(n,i  ) - q(n,i-1));
      Real dqr = (q(n,i+1) - q(n,i  ));

      // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
      Real dq2 = dql*dqr;
      Real dqm = 0.0;
      if (dq2 > 0.0) dqm = dq2/(dql + dqr);

      // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
      ql(n,i+1) = q(n,i) + dqm;
      qr(n,i  ) = q(n,i) - dqm;
    });
  }
  return;
}
