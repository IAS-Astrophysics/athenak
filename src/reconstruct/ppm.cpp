//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ppm.cpp
//  \brief piecewise parabolic reconstruction with Collela-Sekora extremum preserving
//  limiters for a Cartesian-like coordinate with uniform spacing.
//
// This version does not include the extensions to the CS limiters described by
// McCorquodale et al. and as implemented in Athena++ by K. Felker.  This is to keep the
// code simple, because Kyle found these extensions did not improve the solution very
// much in practice, and because they can break monotonicity.
//
// REFERENCES:
// (CW) P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for Gas-Dynamical
// Simulations", JCP, 54, 174 (1984)
//
// (CS) P. Colella & M. Sekora, "A limiter for PPM that preserves accuracy at smooth
// extrema", JCP, 227, 7069 (2008)
//
// (MC) P. McCorquodale & P. Colella,  "A high-order finite-volume method for conservation
// laws on locally refined grids", CAMCoS, 6, 1 (2011)
//
// (PH) L. Peterson & G.W. Hammett, "Positivity preservation and advection algorithms
// with application to edge plasma turbulence", SIAM J. Sci. Com, 35, B576 (2013)

#include <math.h>

#include <algorithm>    // max()

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PPM()
//  \brief Reconstructs parabolic slope in cell i to compute ql(i+1) and qr(i). Works for
//  reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void PPM(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
         const Real &q_ip2, Real &ql_ip1, Real &qr_i) {
  //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
  // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
  // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
  Real qlv = (7.*(q_i + q_im1) - (q_im2 + q_ip1))/12.0;
  Real qrv = (7.*(q_i + q_ip1) - (q_im1 + q_ip2))/12.0;

  //---- Apply CS monotonicity limiters to qrv and qlv ----
  // approximate second derivatives at i-1/2 (PH 3.35)
  // KGF: add the off-center quantities first to preserve FP symmetry
  Real d2qc = 3.0*((q_im1 + q_i) - 2.0*qlv);
  Real d2ql = (q_im2 + q_i  ) - 2.0*q_im1;
  Real d2qr = (q_im1 + q_ip1) - 2.0*q_i;

  // limit second derivative (PH 3.36)
  Real d2qlim = 0.0;
  Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
  if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  // compute limited value for qlv (PH 3.33 and 3.34)
  if (((q_im1 - qlv)*(q_i - qlv)) > 0.0) {
    qlv = 0.5*(q_i + q_im1) - d2qlim/6.0;
  }

  // approximate second derivatives at i+1/2 (PH 3.35)
  // KGF: add the off-center quantities first to preserve FP symmetry
  d2qc = 3.0*((q_i + q_ip1) - 2.0*qrv);
  d2ql = d2qr;
  d2qr = (q_i + q_ip2) - 2.0*q_ip1;

  // limit second derivative (PH 3.36)
  d2qlim = 0.0;
  lim_slope = fmin(fabs(d2ql),fabs(d2qr));
  if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  // compute limited value for qrv (PH 3.33 and 3.34)
  if (((q_i - qrv)*(q_ip1 - qrv)) > 0.0) {
    qrv = 0.5*(q_i + q_ip1) - d2qlim/6.0;
  }

  //---- identify extrema, use smooth extremum limiter ----
  // CS 20 (missing "OR"), and PH 3.31
  Real qa = (qrv - q_i)*(q_i - qlv);
  Real qb = (q_im1 - q_i)*(q_i - q_ip1);
  if (qa <= 0.0 || qb <= 0.0) {
    // approximate secnd derivates (PH 3.37)
    // KGF: add the off-center quantities first to preserve FP symmetry
    Real d2q  = 6.0*(qlv + qrv - 2.0*q_i);
    Real d2qc = (q_im1 + q_ip1) - 2.0*q_i;
    Real d2ql = (q_im2 + q_i  ) - 2.0*q_im1;
    Real d2qr = (q_i   + q_ip2) - 2.0*q_ip1;

    // limit second derivatives (PH 3.38)
    d2qlim = 0.0;
    lim_slope = fmin(fabs(d2ql),fabs(d2qr));
    lim_slope = fmin(fabs(d2qc),lim_slope);
    if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
      d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
    }
    if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
      d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
    }

    // limit L/R states at extrema (PH 3.39)
    Real rho = 0.0;
    if ( fabs(d2q) > (1.0e-12)*fmax( fabs(q_im1), fmax(fabs(q_i),fabs(q_ip1))) ) {
      // Limiter is not sensitive to round-off error.  Use limited slope
      rho = d2qlim/d2q;
    }
    qlv = q_i + (qlv - q_i)*rho;
    qrv = q_i + (qrv - q_i)*rho;
  } else {
    // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
    Real qc = qrv - q_i;
    Real qd = qlv - q_i;
    if (fabs(qc) >= 2.0*fabs(qd)) {
      qrv = q_i - 2.0*qd;
    }
    if (fabs(qd) >= 2.0*fabs(qc)) {
      qlv = q_i - 2.0*qc;
    }
  }

  //---- set L/R states ----
  ql_ip1 = qrv;
  qr_i   = qlv;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseParabolicX1()
//  \brief Wrapper function for PPM reconstruction in x1-direction.
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX1(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      PPM(q(m,n,k,j,i-2), q(m,n,k,j,i-1), q(m,n,k,j,i), q(m,n,k,j,i+1),
          q(m,n,k,j,i+2), ql(n,i+1), qr(n,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseParabolicX2()
//  \brief Wrapper function for PPM reconstruction in x2-direction.
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX2(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      PPM(q(m,n,k,j-2,i), q(m,n,k,j-1,i), q(m,n,k,j,i), q(m,n,k,j+1,i),
          q(m,n,k,j+2,i), ql_jp1(n,i), qr_j(n,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseParabolicX3()
//  \brief Wrapper function for PPM reconstruction in x3-direction.
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX3(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      PPM(q(m,n,k-2,j,i), q(m,n,k-1,j,i), q(m,n,k,j,i), q(m,n,k+1,j,i),
          q(m,n,k+2,j,i), ql_kp1(n,i), qr_k(n,i));
    });
  }
  return;
}
