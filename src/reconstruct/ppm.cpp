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

#include <algorithm>    // max()
#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PiecewiseParabolicX1()
//  \brief Reconstructs parabolic slope in cell i to compute ql(i+1) and qr(i) in [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX1(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const AthenaArray5D<Real> &q,
     AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 
      //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
      // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
      // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
      Real qlv = (7.*(q(m,n,k,j,i)+q(m,n,k,j,i-1)) - (q(m,n,k,j,i-2)+q(m,n,k,j,i+1)))/12.;
      Real qrv = (7.*(q(m,n,k,j,i)+q(m,n,k,j,i+1)) - (q(m,n,k,j,i-1)+q(m,n,k,j,i+2)))/12.;

      //---- Apply CS monotonicity limiters to qrv and qlv ----
      // approximate second derivatives at i-1/2 (PH 3.35) 
      Real d2qc = 3.0*(q(m,n,k,j,i-1) - 2.0*qlv + q(m,n,k,j,i));
      Real d2ql = (q(m,n,k,j,i-2) - 2.0*q(m,n,k,j,i-1) + q(m,n,k,j,i  ));
      Real d2qr = (q(m,n,k,j,i-1) - 2.0*q(m,n,k,j,i  ) + q(m,n,k,j,i+1));
    
      // limit second derivative (PH 3.36)
      Real d2qlim = 0.0;
      Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      } 
      // compute limited value for qlv (PH 3.34)
      qlv = 0.5*(q(m,n,k,j,i) + q(m,n,k,j,i-1)) - d2qlim/6.0;

      // approximate second derivatives at i+1/2 (PH 3.35) 
      d2qc = 3.0*(q(m,n,k,j,i) - 2.0*qrv + q(m,n,k,j,i+1));
      d2ql = d2qr;
      d2qr = (q(m,n,k,j,i  ) - 2.0*q(m,n,k,j,i+1) + q(m,n,k,j,i+2));

      // limit second derivative (PH 3.36)
      d2qlim = 0.0;
      lim_slope = fmin(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      // compute limited value for qrv (PH 3.33)
      qrv = 0.5*(q(m,n,k,j,i) + q(m,n,k,j,i+1)) - d2qlim/6.0;

      //---- identify extrema, use smooth extremum limiter ----
      // CS 20 (missing "OR"), and PH 3.31
      Real qa = (qrv - q(m,n,k,j,i))*(q(m,n,k,j,i) - qlv);
      Real qb = (q(m,n,k,j,i-1) - q(m,n,k,j,i))*(q(m,n,k,j,i) - q(m,n,k,j,i+1));
      if (qa <= 0.0 || qb <= 0.0) {
        // approximate secnd derivates (PH 3.37)
        Real d2q  = 6.0*(qlv - 2.0*q(m,n,k,j,i) + qrv);
        Real d2qc = (q(m,n,k,j,i-1) - 2.0*q(m,n,k,j,i  ) + q(m,n,k,j,i+1));
        Real d2ql = (q(m,n,k,j,i-2) - 2.0*q(m,n,k,j,i-1) + q(m,n,k,j,i  ));
        Real d2qr = (q(m,n,k,j,i  ) - 2.0*q(m,n,k,j,i+1) + q(m,n,k,j,i+2));

        // limit second derivatives (PH 3.38)
        Real d2qlim = 0.0;
        Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
        lim_slope = fmin(fabs(d2qc),lim_slope);
        if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
          d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
        }
        if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
          d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
        }

        // limit L/R states at extrema (PH 3.39)
        if (d2q == 0.0) {  // revert to donor cell
          qlv = q(m,n,k,j,i);
          qrv = q(m,n,k,j,i);
        } else {  // add limited slope (PH 3.39)
          qlv = q(m,n,k,j,i) + (qlv - q(m,n,k,j,i))*d2qlim/d2q;
          qrv = q(m,n,k,j,i) + (qrv - q(m,n,k,j,i))*d2qlim/d2q;
        }
      } else {
        // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
        Real qc = qrv - q(m,n,k,j,i);
        Real qd = qlv - q(m,n,k,j,i);
        if (fabs(qc) >= 2.0*fabs(qd)) {
          qrv = q(m,n,k,j,i) - 2.0*qd;
        }
        if (fabs(qd) >= 2.0*fabs(qc)) {
          qlv = q(m,n,k,j,i) - 2.0*qc;
        }
      }

      //---- set L/R states ----
      ql(n,i+1) = qrv;
      qr(n,i  ) = qlv;
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseParabolicX2()
//  \brief Reconstructs linear slope in cell j to cmpute ql(j+1) and qr(j) over [il,iu]
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX2(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const AthenaArray5D<Real> &q,
     AthenaScratch2D<Real> &ql_jp1, AthenaScratch2D<Real> &qr_j)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 
      //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
      // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
      // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
      Real qlv = (7.*(q(m,n,k,j,i)+q(m,n,k,j-1,i)) - (q(m,n,k,j-2,i)+q(m,n,k,j+1,i)))/12.;
      Real qrv = (7.*(q(m,n,k,j,i)+q(m,n,k,j+1,i)) - (q(m,n,k,j-1,i)+q(m,n,k,j+2,i)))/12.;

      // Apply CS monotonicity limiters to qrv and qlv
      // approximate second derivatives at i-1/2 (PH 3.35) 
      Real d2qc = 3.0*(q(m,n,k,j-1,i) - 2.0*qlv + q(m,n,k,j,i));
      Real d2ql = (q(m,n,k,j-2,i) - 2.0*q(m,n,k,j-1,i) + q(m,n,k,j  ,i));
      Real d2qr = (q(m,n,k,j-1,i) - 2.0*q(m,n,k,j  ,i) + q(m,n,k,j+1,i));
    
      // limit second derivative (PH 3.36)
      Real d2qlim = 0.0;
      Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      } 
      // compute limited value for qlv (PH 3.34)
      qlv = 0.5*(q(m,n,k,j,i) + q(m,n,k,j-1,i)) - d2qlim/6.0;

      // approximate second derivatives at i+1/2 (PH 3.35) 
      d2qc = 3.0*(q(m,n,k,j,i) - 2.0*qrv + q(m,n,k,j+1,i));
      d2ql = d2qr;
      d2qr = (q(m,n,k,j,i  ) - 2.0*q(m,n,k,j+1,i) + q(m,n,k,j+2,i));

      // limit second derivative (PH 3.36)
      d2qlim = 0.0;
      lim_slope = fmin(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      // compute limited value for qrv (PH 3.33)
      qrv = 0.5*(q(m,n,k,j,i) + q(m,n,k,j+1,i)) - d2qlim/6.0;

      //---- identify extrema, use smooth extremum limiter ----
      // CS 20 (missing "OR"), and PH 3.31
      Real qa = (qrv - q(m,n,k,j,i))*(q(m,n,k,j,i) - qlv);
      Real qb = (q(m,n,k,j-1,i) - q(m,n,k,j,i))*(q(m,n,k,j,i) - q(m,n,k,j+1,i));
      if (qa <= 0.0 || qb <= 0.0) {
        // approximate secnd derivates (PH 3.37)
        Real d2q  = 6.0*(qlv - 2.0*q(m,n,k,j,i) + qrv);
        Real d2qc = (q(m,n,k,j-1,i) - 2.0*q(m,n,k,j  ,i) + q(m,n,k,j+1,i));
        Real d2ql = (q(m,n,k,j-2,i) - 2.0*q(m,n,k,j-1,i) + q(m,n,k,j  ,i));
        Real d2qr = (q(m,n,k,j  ,i) - 2.0*q(m,n,k,j+1,i) + q(m,n,k,j+2,i));

        // limit second derivatives (PH 3.38)
        Real d2qlim = 0.0;
        Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
        lim_slope = fmin(fabs(d2qc),lim_slope);
        if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
          d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
        }
        if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
          d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
        }

        // limit L/R states at extrema (PH 3.39)
        if (d2q == 0.0) {  // revert to donor cell
          qlv = q(m,n,k,j,i);
          qrv = q(m,n,k,j,i);
        } else {  // add limited slope (PH 3.39)
          qlv = q(m,n,k,j,i) + (qlv - q(m,n,k,j,i))*d2qlim/d2q;
          qrv = q(m,n,k,j,i) + (qrv - q(m,n,k,j,i))*d2qlim/d2q;
        }
      } else {
        // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
        Real qc = qrv - q(m,n,k,j,i);
        Real qd = qlv - q(m,n,k,j,i);
        if (fabs(qc) >= 2.0*fabs(qd)) {
          qrv = q(m,n,k,j,i) - 2.0*qd;
        }
        if (fabs(qd) >= 2.0*fabs(qc)) {
          qlv = q(m,n,k,j,i) - 2.0*qc;
        }
      }

      //---- set L/R states ----
      ql_jp1(n,i) = qrv;
      qr_j  (n,i) = qlv;
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseParabolicX3()
//  \brief Reconstructs linear slope in cell k to cmpute ql(k+1) and qr(k) over [il,iu]
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX3(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const AthenaArray5D<Real> &q,
     AthenaScratch2D<Real> &ql_kp1, AthenaScratch2D<Real> &qr_k)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
    // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
    // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
    par_for_inner(member, il, iu, [&](const int i)
    { 
      Real qlv = (7.*(q(m,n,k,j,i)+q(m,n,k-1,j,i)) - (q(m,n,k-2,j,i)+q(m,n,k+1,j,i)))/12.;
      Real qrv = (7.*(q(m,n,k,j,i)+q(m,n,k+1,j,i)) - (q(m,n,k-1,j,i)+q(m,n,k+2,j,i)))/12.;

      //---- Apply CS monotonicity limiters to qrv and qlv ----
      // approximate second derivatives at i-1/2 (PH 3.35) 
      Real d2qc = 3.0*(q(m,n,k-1,j,i) - 2.0*qlv + q(m,n,k,j,i));
      Real d2ql = (q(m,n,k-2,j,i) - 2.0*q(m,n,k-1,j,i) + q(m,n,k  ,j,i));
      Real d2qr = (q(m,n,k-1,j,i) - 2.0*q(m,n,k  ,j,i) + q(m,n,k+1,j,i));
    
      // limit second derivative (PH 3.36)
      Real d2qlim = 0.0;
      Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      } 
      // compute limited value for qlv (PH 3.34)
      qlv = 0.5*(q(m,n,k,j,i) + q(m,n,k-1,j,i)) - d2qlim/6.0;

      // approximate second derivatives at i+1/2 (PH 3.35) 
      d2qc = 3.0*(q(m,n,k,j,i) - 2.0*qrv + q(m,n,k+1,j,i));
      d2ql = d2qr;
      d2qr = (q(m,n,k,j,i  ) - 2.0*q(m,n,k+1,j,i) + q(m,n,k+2,j,i));

      // limit second derivative (PH 3.36)
      d2qlim = 0.0;
      lim_slope = fmin(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
      }
      // compute limited value for qrv (PH 3.33)
      qrv = 0.5*(q(m,n,k,j,i) + q(m,n,k+1,j,i)) - d2qlim/6.0;

      //---- identify extrema, use smooth extremum limiter ----
      // CS 20 (missing "OR"), and PH 3.31
      Real qa = (qrv - q(m,n,k,j,i))*(q(m,n,k,j,i) - qlv);
      Real qb = (q(m,n,k-1,j,i) - q(m,n,k,j,i))*(q(m,n,k,j,i) - q(m,n,k+1,j,i));
      if (qa <= 0.0 || qb <= 0.0) {
        // approximate secnd derivates (PH 3.37)
        Real d2q  = 6.0*(qlv - 2.0*q(m,n,k,j,i) + qrv);
        Real d2qc = (q(m,n,k-1,j,i) - 2.0*q(m,n,k  ,j,i) + q(m,n,k+1,j,i));
        Real d2ql = (q(m,n,k-2,j,i) - 2.0*q(m,n,k-1,j,i) + q(m,n,k  ,j,i));
        Real d2qr = (q(m,n,k  ,j,i) - 2.0*q(m,n,k+1,j,i) + q(m,n,k+2,j,i));

        // limit second derivatives (PH 3.38)
        Real d2qlim = 0.0;
        Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
        lim_slope = fmin(fabs(d2qc),lim_slope);
        if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
          d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
        }
        if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
          d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
        }

        // limit L/R states at extrema (PH 3.39)
        if (d2q == 0.0) {  // revert to donor cell
          qlv = q(m,n,k,j,i);
          qrv = q(m,n,k,j,i);
        } else {  // add limited slope (PH 3.39)
          qlv = q(m,n,k,j,i) + (qlv - q(m,n,k,j,i))*d2qlim/d2q;
          qrv = q(m,n,k,j,i) + (qrv - q(m,n,k,j,i))*d2qlim/d2q;
        }
      } else {
        // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
        Real qc = qrv - q(m,n,k,j,i);
        Real qd = qlv - q(m,n,k,j,i);
        if (fabs(qc) >= 2.0*fabs(qd)) {
          qrv = q(m,n,k,j,i) - 2.0*qd;
        }
        if (fabs(qd) >= 2.0*fabs(qc)) {
          qlv = q(m,n,k,j,i) - 2.0*qc;
        }
      }

      //---- set L/R states ----
      ql_kp1(n,i) = qrv;
      qr_k  (n,i) = qlv;
    });
  }
  return;
}
