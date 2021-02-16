//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ppm.cpp
//  \brief WENO-Z reconstruction for a Cartesian-like coordinate with uniform spacing.
//
// REFERENCES:
// Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
// non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)

#include <algorithm>    // max()
#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn WENOZ()
//  \brief Reconstructs 5th-order polynomial in cell i to compute ql(i+1) and qr(i).
//  Works for any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void WENOZ(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
           const Real &q_ip2, Real &ql_ip1, Real &qr_i) noexcept 
{
  // Smooth WENO weights: Note that these are from Del Zanna et al. 2007 (A.18)
  const Real beta_coeff[2]{13. / 12., 0.25};

  Real beta[3];
  beta[0] = beta_coeff[0] * SQR(q_im2 +     q_i - 2.0*q_im1) +
            beta_coeff[1] * SQR(q_im2 + 3.0*q_i - 4.0*q_im1);

  beta[1] = beta_coeff[0] * SQR(q_im1 + q_ip1 - 2.0*q_i) +
            beta_coeff[1] * SQR(q_im1 - q_ip1);

  beta[2] = beta_coeff[0] * SQR(q_ip2 +      q_i - 2.0*q_ip1) +
            beta_coeff[1] * SQR(q_ip2 + 3.0* q_i - 4.0*q_ip1);

  // Rescale epsilon
  const Real epsL = 1.0e-42;

  // WENO-Z+: Acker et al. 2016
  const Real tau_5 = fabs(beta[0] - beta[2]);

  Real indicator[3];
  indicator[0] = tau_5 / (beta[0] + epsL);
  indicator[1] = tau_5 / (beta[1] + epsL);
  indicator[2] = tau_5 / (beta[2] + epsL);

  // compute qL_ip1
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  Real f[3];
  f[0] = ( 2.0*q_im2 - 7.0*q_im1 + 11.0*q_i  );
  f[1] = (-1.0*q_im1 + 5.0*q_i   + 2.0 *q_ip1);
  f[2] = ( 2.0*q_i   + 5.0*q_ip1 -      q_ip2);

  Real alpha[3];
  alpha[0] = 0.1*(1.0 + SQR(indicator[0]));
  alpha[1] = 0.6*(1.0 + SQR(indicator[1]));
  alpha[2] = 0.3*(1.0 + SQR(indicator[2]));
  Real alpha_sum = 6.0*(alpha[0] + alpha[1] + alpha[2]);

  ql_ip1 = (f[0]*alpha[0] + f[1]*alpha[1] + f[2]*alpha[2])/alpha_sum;

  // compute qR_i
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] = ( 2.0*q_ip2 - 7.0*q_ip1 + 11.0*q_i  );
  f[1] = (-1.0*q_ip1 + 5.0*q_i   + 2.0 *q_im1);
  f[2] = ( 2.0*q_i   + 5.0*q_im1 -      q_im2);

  alpha[0] = 0.1*(1.0 + SQR(indicator[2]));
  alpha[1] = 0.6*(1.0 + SQR(indicator[1]));
  alpha[2] = 0.3*(1.0 + SQR(indicator[0]));
  alpha_sum = 6.0*(alpha[0] + alpha[1] + alpha[2]);

  qr_i = (f[0]*alpha[0] + f[1]*alpha[1] + f[2]*alpha[2])/alpha_sum;

  return;
}


//----------------------------------------------------------------------------------------
//! \fn WENOZ
//  \brief Wrapper function for WENOZ reconstruction in x1-direction.
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void WENOZX1(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 
      WENOZ(q(m,n,k,j,i-2), q(m,n,k,j,i-1), q(m,n,k,j,i), q(m,n,k,j,i+1),
            q(m,n,k,j,i+2), ql(n,i+1), qr(n,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZX2
//  \brief Wrapper function for WENOZ reconstruction in x1-direction.
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void WENOZX2(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 
      WENOZ(q(m,n,k,j-2,i), q(m,n,k,j-1,i), q(m,n,k,j,i), q(m,n,k,j+1,i),
            q(m,n,k,j+2,i), ql_jp1(n,i), qr_j(n,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZX3
//  \brief Wrapper function for WENOZ reconstruction in x1-direction.
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void WENOZX3(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 
      WENOZ(q(m,n,k-2,j,i), q(m,n,k-1,j,i), q(m,n,k,j,i), q(m,n,k+1,j,i),
            q(m,n,k+2,j,i), ql_kp1(n,i), qr_k(n,i));
    });
  }
  return;
}
