//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file plm.cpp
//  \brief  piecewise linear reconstruction implemented as inline functions
//  This version only works with uniform mesh spacing

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX1()
//  \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i) over [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX1(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    // compute L/R slopes for each variable
    par_for_inner(member, il, iu, [&](const int i)
    { 
      Real dql = (q(m,n,k,j,i  ) - q(m,n,k,j,i-1));
      Real dqr = (q(m,n,k,j,i+1) - q(m,n,k,j,i  ));

      // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
      Real dq2 = dql*dqr;
      Real dqm = dq2/(dql + dqr);
      if (dq2 <= 0.0) dqm = 0.0;

      // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
      ql(n,i+1) = q(m,n,k,j,i) + dqm;
      qr(n,i  ) = q(m,n,k,j,i) - dqm;
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX2()
//  \brief Reconstructs linear slope in cell j to cmpute ql(j+1) and qr(j) over [il,iu]
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX2(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    // compute L/R slopes for each variable
    par_for_inner(member, il, iu, [&](const int i)
    { 
      Real dql = (q(m,n,k,j  ,i) - q(m,n,k,j-1,i));
      Real dqr = (q(m,n,k,j+1,i) - q(m,n,k,j  ,i));

      // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
      Real dq2 = dql*dqr;
      Real dqm = dq2/(dql + dqr);
      if (dq2 <= 0.0) dqm = 0.0;

      // compute ql_(j+1/2) and qr_(j-1/2) using limited slopes
      ql_jp1(n,i) = q(m,n,k,j,i) + dqm;
      qr_j  (n,i) = q(m,n,k,j,i) - dqm;
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX3()
//  \brief Reconstructs linear slope in cell k to cmpute ql(k+1) and qr(k) over [il,iu]
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX3(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    // compute L/R slopes for each variable
    par_for_inner(member, il, iu, [&](const int i)
    { 
      Real dql = (q(m,n,k  ,j,i) - q(m,n,k-1,j,i));
      Real dqr = (q(m,n,k+1,j,i) - q(m,n,k  ,j,i));

      // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
      Real dq2 = dql*dqr;
      Real dqm = dq2/(dql + dqr);
      if (dq2 <= 0.0) dqm = 0.0;

      // compute ql_(k+1/2) and qr_(k-1/2) using limited slopes
      ql_kp1(n,i) = q(m,n,k,j,i) + dqm;
      qr_k  (n,i) = q(m,n,k,j,i) - dqm;
    });
  }
  return;
}
