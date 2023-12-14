#ifndef RECONSTRUCT_WENOZ_HPP_
#define RECONSTRUCT_WENOZ_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file wenoz.hpp
//! \brief WENO-Z reconstruction for a Cartesian-like coordinate with uniform spacing.
//!
//! REFERENCES:
//! Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
//! non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)

#include <math.h>
#include <algorithm>    // max()

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn WENOZ()
//! \brief Reconstructs 5th-order polynomial in cell i to compute ql(i+1) and qr(i).
//! Works for any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void WENOZ(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
           const Real &q_ip2, Real &ql_ip1, Real &qr_i) noexcept  {
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
//! \brief Wrapper function for WENOZ reconstruction in x1-direction.
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void WENOZX1(TeamMember_t const &member, const EOS_Data &eos, const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      Real &qim2 = q(m,n,k,j,i-2);
      Real &qim1 = q(m,n,k,j,i-1);
      Real &qi   = q(m,n,k,j,i  );
      Real &qip1 = q(m,n,k,j,i+1);
      Real &qip2 = q(m,n,k,j,i+2);
      WENOZ(qim2, qim1, qi, qip1, qip2, ql(n,i+1), qr(n,i));
      if (apply_floors) {
        if (n==IDN) {
          ql(IDN,i+1) = fmax(ql(IDN,i+1), dfloor_);
          qr(IDN,i  ) = fmax(qr(IDN,i  ), dfloor_);
        }
        if (n==IEN) {
          ql(IEN,i+1) = fmax(ql(IEN,i+1), efloor_);
          qr(IEN,i  ) = fmax(qr(IEN,i  ), efloor_);
        }
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZX2
//! \brief Wrapper function for WENOZ reconstruction in x1-direction.
//! This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void WENOZX2(TeamMember_t const &member, const EOS_Data &eos, const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      Real &qjm2 = q(m,n,k,j-2,i);
      Real &qjm1 = q(m,n,k,j-1,i);
      Real &qj   = q(m,n,k,j  ,i);
      Real &qjp1 = q(m,n,k,j+1,i);
      Real &qjp2 = q(m,n,k,j+2,i);
      WENOZ(qjm2, qjm1, qj, qjp1, qjp2, ql_jp1(n,i), qr_j(n,i));
      if (apply_floors) {
        if (n==IDN) {
          ql_jp1(IDN,i) = fmax(ql_jp1(IDN,i), dfloor_);
          qr_j  (IDN,i) = fmax(qr_j  (IDN,i), dfloor_);
        }
        if (n==IEN) {
          ql_jp1(IEN,i) = fmax(ql_jp1(IEN,i), efloor_);
          qr_j  (IEN,i) = fmax(qr_j  (IEN,i), efloor_);
        }
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZX3
//! \brief Wrapper function for WENOZ reconstruction in x1-direction.
//! This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void WENOZX3(TeamMember_t const &member, const EOS_Data &eos, const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      Real &qkm2 = q(m,n,k-2,j,i);
      Real &qkm1 = q(m,n,k-1,j,i);
      Real &qk   = q(m,n,k  ,j,i);
      Real &qkp1 = q(m,n,k+1,j,i);
      Real &qkp2 = q(m,n,k+2,j,i);
      WENOZ(qkm2, qkm1, qk, qkp1, qkp2, ql_kp1(n,i), qr_k(n,i));
      if (apply_floors) {
        if (n==IDN) {
          ql_kp1(IDN,i) = fmax(ql_kp1(IDN,i), dfloor_);
          qr_k  (IDN,i) = fmax(qr_k  (IDN,i), dfloor_);
        }
        if (n==IEN) {
          ql_kp1(IEN,i) = fmax(ql_kp1(IEN,i), efloor_);
          qr_k  (IEN,i) = fmax(qr_k  (IEN,i), efloor_);
        }
      }
    });
  }
  return;
}
#endif // RECONSTRUCT_WENOZ_HPP_
