#ifndef RECONSTRUCT_PPM_HPP_
#define RECONSTRUCT_PPM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ppm.hpp
//! \brief piecewise parabolic reconstruction with both Collela-Woodward (CW) limiters
//! (implemented in the PPM4 inline function) and Collela-Sekora (CS) extremum preserving
//! limiters (implemented in the PPMX inline function) for a Cartesian-like coordinates
//! with uniform spacing.
//!
//! This version does not include the extensions to the CS limiters described by
//! McCorquodale et al. and as implemented in Athena++ by K. Felker.  This is to keep the
//! code simple, because Kyle found these extensions did not improve the solution very
//! much in practice, and because they can break monotonicity.
//!
//! REFERENCES:
//! (CW) P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for Gas-Dynamical
//! Simulations", JCP, 54, 174 (1984)
//!
//! (CS) P. Colella & M. Sekora, "A limiter for PPM that preserves accuracy at smooth
//! extrema", JCP, 227, 7069 (2008)
//!
//! (MC) P. McCorquodale & P. Colella, "A high-order finite-volume method for conservation
//! laws on locally refined grids", CAMCoS, 6, 1 (2011)
//!
//! (PH) L. Peterson & G.W. Hammett, "Positivity preservation and advection algorithms
//! with application to edge plasma turbulence", SIAM J. Sci. Com, 35, B576 (2013)

#include <math.h>
#include <algorithm>    // max()

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PPM4()
//! \brief Original PPM (Colella & Woodward) parabolic reconstruction.  Returns
//! interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void PPM4(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
          const Real &q_ip2, Real &ql_ip1, Real &qr_i) {
  //---- Interpolate L/R values (CS eqn 16, PH 3.26 and 3.27) ----
  // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
  // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
  Real qlv = (7.*(q_i + q_im1) - (q_im2 + q_ip1))/12.0;
  Real qrv = (7.*(q_i + q_ip1) - (q_im1 + q_ip2))/12.0;

  //---- limit qrv and qlv to neighboring cell-centered values (CS eqn 13) ----
  qlv = fmax(qlv, fmin(q_i, q_im1));
  qlv = fmin(qlv, fmax(q_i, q_im1));
  qrv = fmax(qrv, fmin(q_i, q_ip1));
  qrv = fmin(qrv, fmax(q_i, q_ip1));

  //--- monotonize interpolated L/R states (CS eqns 14, 15) ---
  Real qc = qrv - q_i;
  Real qd = qlv - q_i;
  if ((qc*qd) >= 0.0) {
    qlv = q_i;
    qrv = q_i;
  } else {
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
//! \fn PPMX()
//! \brief PPM parabolic reconstruction with Colella & Sekora limiters.  Returns
//! interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void PPMX(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
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
//! \brief Wrapper function for PPM reconstruction in x1-direction.
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX1(TeamMember_t const &member,
     const EOS_Data &eos, const bool extremum_preserving, const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    if (extremum_preserving) {
      par_for_inner(member, il, iu, [&](const int i) {
        Real &qim2 = q(m,n,k,j,i-2);
        Real &qim1 = q(m,n,k,j,i-1);
        Real &qi   = q(m,n,k,j,i  );
        Real &qip1 = q(m,n,k,j,i+1);
        Real &qip2 = q(m,n,k,j,i+2);
        PPMX(qim2, qim1, qi, qip1, qip2, ql(n,i+1), qr(n,i));
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
    } else {
      par_for_inner(member, il, iu, [&](const int i) {
        Real &qim2 = q(m,n,k,j,i-2);
        Real &qim1 = q(m,n,k,j,i-1);
        Real &qi   = q(m,n,k,j,i  );
        Real &qip1 = q(m,n,k,j,i+1);
        Real &qip2 = q(m,n,k,j,i+2);
        PPM4(qim2, qim1, qi, qip1, qip2, ql(n,i+1), qr(n,i));
      });
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseParabolicX2()
//! \brief Wrapper function for PPM reconstruction in x2-direction.
//! This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX2(TeamMember_t const &member,
     const EOS_Data &eos, const bool extremum_preserving, const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    if (extremum_preserving) {
      par_for_inner(member, il, iu, [&](const int i) {
        Real &qjm2 = q(m,n,k,j-2,i);
        Real &qjm1 = q(m,n,k,j-1,i);
        Real &qj   = q(m,n,k,j  ,i);
        Real &qjp1 = q(m,n,k,j+1,i);
        Real &qjp2 = q(m,n,k,j+2,i);
        PPMX(qjm2, qjm1, qj, qjp1, qjp2, ql_jp1(n,i), qr_j(n,i));
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
    } else {
      par_for_inner(member, il, iu, [&](const int i) {
        Real &qjm2 = q(m,n,k,j-2,i);
        Real &qjm1 = q(m,n,k,j-1,i);
        Real &qj   = q(m,n,k,j  ,i);
        Real &qjp1 = q(m,n,k,j+1,i);
        Real &qjp2 = q(m,n,k,j+2,i);
        PPM4(qjm2, qjm1, qj, qjp1, qjp2, ql_jp1(n,i), qr_j(n,i));
      });
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseParabolicX3()
//! \brief Wrapper function for PPM reconstruction in x3-direction.
//! This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX3(TeamMember_t const &member,
     const EOS_Data &eos, const bool extremum_preserving, const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    if (extremum_preserving) {
      par_for_inner(member, il, iu, [&](const int i) {
        Real &qkm2 = q(m,n,k-2,j,i);
        Real &qkm1 = q(m,n,k-1,j,i);
        Real &qk   = q(m,n,k  ,j,i);
        Real &qkp1 = q(m,n,k+1,j,i);
        Real &qkp2 = q(m,n,k+2,j,i);
        PPMX(qkm2, qkm1, qk, qkp1, qkp2, ql_kp1(n,i), qr_k(n,i));
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
    } else {
      par_for_inner(member, il, iu, [&](const int i) {
        Real &qkm2 = q(m,n,k-2,j,i);
        Real &qkm1 = q(m,n,k-1,j,i);
        Real &qk   = q(m,n,k  ,j,i);
        Real &qkp1 = q(m,n,k+1,j,i);
        Real &qkp2 = q(m,n,k+2,j,i);
        PPM4(qkm2, qkm1, qk, qkp1, qkp2, ql_kp1(n,i), qr_k(n,i));
      });
    }
  }
  return;
}
#endif // RECONSTRUCT_PPM_HPP_
