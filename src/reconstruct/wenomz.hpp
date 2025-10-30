#ifndef RECONSTRUCT_WENOMZ_HPP_
#define RECONSTRUCT_WENOMZ_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file wenomz.hpp
//! \brief WENO-MZ reconstruction for a Cartesian-like coordinate with uniform spacing.
//!
//! REFERENCES:
//! Wang Y., Zhao K., Yuan L., "A modified fifth-order WENO-Z scheme based on the
//! weights of the reformulated adaptive order WENO scheme"
//! Int J Numer Meth Fluids. 2024;96:1631–1652
//!


#include <math.h>
#include <algorithm>    // max()

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn WENOMZ()
//! \brief Reconstructs 5th-order polynomial in cell i to compute ql(i+1) and qr(i).
//! Works for any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void WENOMZ(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
           const Real &q_ip2, Real &ql_ip1, Real &qr_i) noexcept  {
  // Smooth WENO weights: See Jiang & Shu 1996

  constexpr Real beta_coeff0 = 13.0/12.0, beta_coeff1 = 0.25, beta_coeff4 = 1.0/12.0;
  constexpr Real epsL = 1.0e-40;
  constexpr Real t1 = 2.0;

  Real beta0 = beta_coeff0 * SQR(q_im2 +     q_i - 2.0*q_im1) +
          beta_coeff1 * SQR(q_im2 + 3.0*q_i - 4.0*q_im1);

  Real beta1 = beta_coeff0 * SQR(q_im1 + q_ip1 - 2.0*q_i) +
          beta_coeff1 * SQR(q_im1 - q_ip1);

  Real beta2 = beta_coeff0 * SQR(q_ip2 +     q_i - 2.0*q_ip1) +
          beta_coeff1 * SQR(q_ip2 + 3.0*q_i - 4.0*q_ip1);

  Real beta4 = beta_coeff4 * SQR(q_im1 - 2.0*q_i + q_ip1);

  Real tau_5 = fabs(beta0 - beta2);
  Real r = (fabs(beta2 - beta1) + epsL) / (fabs(beta0 - beta1) + epsL);
  Real t0 = 1.0 + r;
  Real t2 = 1.0 + 1.0/r;
  Real eta = tau_5*SQR(SQR(tau_5/(fmax(beta0, beta2) + epsL) ));

  Real weight_arg[3];
  weight_arg[0] = eta/(beta0+epsL) + (tau_5 - eta)/(t0*beta4 + epsL);
  weight_arg[1] = eta/(beta1+epsL) + (tau_5 - eta)/(t1*beta4 + epsL);
  weight_arg[2] = eta/(beta2+epsL) + (tau_5 - eta)/(t2*beta4 + epsL);

  // compute qL_ip1
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  Real f[3];
  f[0] = ( 2.0*q_im2 - 7.0*q_im1 + 11.0*q_i  );
  f[1] = (-1.0*q_im1 + 5.0*q_i   + 2.0 *q_ip1);
  f[2] = ( 2.0*q_i   + 5.0*q_ip1 -      q_ip2);

  Real alpha[3];
  alpha[0] = 0.1 + 0.1*weight_arg[0];
  alpha[1] = 0.6 + 0.6*weight_arg[1];
  alpha[2] = 0.3 + 0.3*weight_arg[2];

  Real alpha_sum = 6.0*(alpha[0] + alpha[1] + alpha[2]);

  ql_ip1 = (f[0]*alpha[0] + f[1]*alpha[1] + f[2]*alpha[2])/alpha_sum;

  // compute qR_i
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] = ( 2.0*q_ip2 - 7.0*q_ip1 + 11.0*q_i  );
  f[1] = (-1.0*q_ip1 + 5.0*q_i   + 2.0 *q_im1);
  f[2] = ( 2.0*q_i   + 5.0*q_im1 -      q_im2);

  alpha[0] = 0.1 + 0.1*weight_arg[2];
  alpha[2] = 0.3 + 0.3*weight_arg[0];

  alpha_sum = 6.0*(alpha[0] + alpha[1] + alpha[2]);

  qr_i = (f[0]*alpha[0] + f[1]*alpha[1] + f[2]*alpha[2])/alpha_sum;

  return;
}


//----------------------------------------------------------------------------------------
//! \fn WENOMZ
//! \brief Wrapper function for WENOMZ reconstruction in x1-direction.
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void WENOMZX1(TeamMember_t const &member, const EOS_Data &eos, const bool apply_floors,
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
      WENOMZ(qim2, qim1, qi, qip1, qip2, ql(n,i+1), qr(n,i));
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
//! \fn WENOMZX2
//! \brief Wrapper function for WENOMZ reconstruction in x1-direction.
//! This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void WENOMZX2(TeamMember_t const &member, const EOS_Data &eos, const bool apply_floors,
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
      WENOMZ(qjm2, qjm1, qj, qjp1, qjp2, ql_jp1(n,i), qr_j(n,i));
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
//! \fn WENOMZX3
//! \brief Wrapper function for WENOMZ reconstruction in x1-direction.
//! This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void WENOMZX3(TeamMember_t const &member, const EOS_Data &eos, const bool apply_floors,
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
      WENOMZ(qkm2, qkm1, qk, qkp1, qkp2, ql_kp1(n,i), qr_k(n,i));
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
#endif // RECONSTRUCT_WENOMZ_HPP_
