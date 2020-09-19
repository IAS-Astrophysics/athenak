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
#include "reconstruct.hpp"

//----------------------------------------------------------------------------------------
// PiecewiseLinear constructor

PiecewiseParabolic::PiecewiseParabolic(ParameterInput *pin, int nvar, int ncells1) :
  Reconstruction(pin, nvar, ncells1)
{
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseParabolicX1()
//  \brief Reconstructs parabolic slope in cell i to compute ql(i+1) and qr(i) in [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

void PiecewiseParabolic::ReconstructX1(const int k,const int j,const int il,const int iu,
    const AthenaArray4D<Real> &q, AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr)
{
  int nvar = q.extent_int(0);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
      // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
      // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
      Real qlv_ = (7.0*(q(n,k,j,i) + q(n,k,j,i-1)) - (q(n,k,j,i-2) + q(n,k,j,i+1)))/12.0;
      Real qrv_ = (7.0*(q(n,k,j,i) + q(n,k,j,i+1)) - (q(n,k,j,i-1) + q(n,k,j,i+2)))/12.0;

      //---- Apply CS monotonicity limiters to qrv_ and qlv_ ----
      // approximate second derivatives at i-1/2 (PH 3.35) 
      Real d2qc = 3.0*(q(n,k,j,i-1) - 2.0*qlv_ + q(n,k,j,i));
      Real d2ql = (q(n,k,j,i-2) - 2.0*q(n,k,j,i-1) + q(n,k,j,i  ));
      Real d2qr = (q(n,k,j,i-1) - 2.0*q(n,k,j,i  ) + q(n,k,j,i+1));
    
      // limit second derivative (PH 3.36)
      Real d2qlim = 0.0;
      Real lim_slope = std::min(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      } 
      // compute limited value for qlv_ (PH 3.34)
      qlv_ = 0.5*(q(n,k,j,i) + q(n,k,j,i-1)) - d2qlim/6.0;

      // approximate second derivatives at i+1/2 (PH 3.35) 
      d2qc = 3.0*(q(n,k,j,i) - 2.0*qrv_ + q(n,k,j,i+1));
      d2ql = d2qr;
      d2qr = (q(n,k,j,i  ) - 2.0*q(n,k,j,i+1) + q(n,k,j,i+2));

      // limit second derivative (PH 3.36)
      d2qlim = 0.0;
      lim_slope = std::min(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      // compute limited value for qrv_ (PH 3.33)
      qrv_ = 0.5*(q(n,k,j,i) + q(n,k,j,i+1)) - d2qlim/6.0;

      //---- identify extrema, use smooth extremum limiter ----
      // CS 20 (missing "OR"), and PH 3.31
      Real qa = (qrv_ - q(n,k,j,i))*(q(n,k,j,i) - qlv_);
      Real qb = (q(n,k,j,i-1) - q(n,k,j,i))*(q(n,k,j,i) - q(n,k,j,i+1));
      if (qa <= 0.0 || qb <= 0.0) {
        // approximate secnd derivates (PH 3.37)
        Real d2q  = 6.0*(qlv_ - 2.0*q(n,k,j,i) + qrv_);
        Real d2qc = (q(n,k,j,i-1) - 2.0*q(n,k,j,i  ) + q(n,k,j,i+1));
        Real d2ql = (q(n,k,j,i-2) - 2.0*q(n,k,j,i-1) + q(n,k,j,i  ));
        Real d2qr = (q(n,k,j,i  ) - 2.0*q(n,k,j,i+1) + q(n,k,j,i+2));

        // limit second derivatives (PH 3.38)
        Real d2qlim = 0.0;
        Real lim_slope = std::min(fabs(d2ql),fabs(d2qr));
        lim_slope = std::min(fabs(d2qc),lim_slope);
        if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
          d2qlim = SIGN(d2q)*std::min(1.25*lim_slope,fabs(d2q));
        }
        if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
          d2qlim = SIGN(d2q)*std::min(1.25*lim_slope,fabs(d2q));
        }

        // limit L/R states at extrema (PH 3.39)
        if (d2q == 0.0) {  // revert to donor cell
          qlv_ = q(n,k,j,i);
          qrv_ = q(n,k,j,i);
        } else {  // add limited slope (PH 3.39)
          qlv_ = q(n,k,j,i) + (qlv_ - q(n,k,j,i))*d2qlim/d2q;
          qrv_ = q(n,k,j,i) + (qrv_ - q(n,k,j,i))*d2qlim/d2q;
        }
      } else {
        // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
        Real qc = qrv_ - q(n,k,j,i);
        Real qd = qlv_ - q(n,k,j,i);
        if (fabs(qc) >= 2.0*fabs(qd)) {
          qrv_ = q(n,k,j,i) - 2.0*qd;
        }
        if (fabs(qd) >= 2.0*fabs(qc)) {
          qlv_ = q(n,k,j,i) - 2.0*qc;
        }
      }

      //---- set L/R states ----
      ql(n,i+1) = qrv_;
      qr(n,i  ) = qlv_;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief Reconstructs linear slope in cell j to cmpute ql(j+1) and qr(j) over [il,iu]
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

void PiecewiseParabolic::ReconstructX2(const int k,const int j,const int il,const int iu,
    const AthenaArray4D<Real> &q, AthenaArray2D<Real> &ql_jp1, AthenaArray2D<Real> &qr_j)
{
  int nvar = q.extent_int(0);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
      // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
      // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
      Real qlv_ = (7.0*(q(n,k,j,i) + q(n,k,j-1,i)) - (q(n,k,j-2,i) + q(n,k,j+1,i)))/12.0;
      Real qrv_ = (7.0*(q(n,k,j,i) + q(n,k,j+1,i)) - (q(n,k,j-1,i) + q(n,k,j+2,i)))/12.0;

      // Apply CS monotonicity limiters to qrv_ and qlv_
      // approximate second derivatives at i-1/2 (PH 3.35) 
      Real d2qc = 3.0*(q(n,k,j-1,i) - 2.0*qlv_ + q(n,k,j,i));
      Real d2ql = (q(n,k,j-2,i) - 2.0*q(n,k,j-1,i) + q(n,k,j  ,i));
      Real d2qr = (q(n,k,j-1,i) - 2.0*q(n,k,j  ,i) + q(n,k,j+1,i));
    
      // limit second derivative (PH 3.36)
      Real d2qlim = 0.0;
      Real lim_slope = std::min(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      } 
      // compute limited value for qlv_ (PH 3.34)
      qlv_ = 0.5*(q(n,k,j,i) + q(n,k,j-1,i)) - d2qlim/6.0;

      // approximate second derivatives at i+1/2 (PH 3.35) 
      d2qc = 3.0*(q(n,k,j,i) - 2.0*qrv_ + q(n,k,j+1,i));
      d2ql = d2qr;
      d2qr = (q(n,k,j,i  ) - 2.0*q(n,k,j+1,i) + q(n,k,j+2,i));

      // limit second derivative (PH 3.36)
      d2qlim = 0.0;
      lim_slope = std::min(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      // compute limited value for qrv_ (PH 3.33)
      qrv_ = 0.5*(q(n,k,j,i) + q(n,k,j+1,i)) - d2qlim/6.0;

      //---- identify extrema, use smooth extremum limiter ----
      // CS 20 (missing "OR"), and PH 3.31
      Real qa = (qrv_ - q(n,k,j,i))*(q(n,k,j,i) - qlv_);
      Real qb = (q(n,k,j-1,i) - q(n,k,j,i))*(q(n,k,j,i) - q(n,k,j+1,i));
      if (qa <= 0.0 || qb <= 0.0) {
        // approximate secnd derivates (PH 3.37)
        Real d2q  = 6.0*(qlv_ - 2.0*q(n,k,j,i) + qrv_);
        Real d2qc = (q(n,k,j-1,i) - 2.0*q(n,k,j  ,i) + q(n,k,j+1,i));
        Real d2ql = (q(n,k,j-2,i) - 2.0*q(n,k,j-1,i) + q(n,k,j  ,i));
        Real d2qr = (q(n,k,j  ,i) - 2.0*q(n,k,j+1,i) + q(n,k,j+2,i));

        // limit second derivatives (PH 3.38)
        Real d2qlim = 0.0;
        Real lim_slope = std::min(fabs(d2ql),fabs(d2qr));
        lim_slope = std::min(fabs(d2qc),lim_slope);
        if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
          d2qlim = SIGN(d2q)*std::min(1.25*lim_slope,fabs(d2q));
        }
        if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
          d2qlim = SIGN(d2q)*std::min(1.25*lim_slope,fabs(d2q));
        }

        // limit L/R states at extrema (PH 3.39)
        if (d2q == 0.0) {  // revert to donor cell
          qlv_ = q(n,k,j,i);
          qrv_ = q(n,k,j,i);
        } else {  // add limited slope (PH 3.39)
          qlv_ = q(n,k,j,i) + (qlv_ - q(n,k,j,i))*d2qlim/d2q;
          qrv_ = q(n,k,j,i) + (qrv_ - q(n,k,j,i))*d2qlim/d2q;
        }
      } else {
        // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
        Real qc = qrv_ - q(n,k,j,i);
        Real qd = qlv_ - q(n,k,j,i);
        if (fabs(qc) >= 2.0*fabs(qd)) {
          qrv_ = q(n,k,j,i) - 2.0*qd;
        }
        if (fabs(qd) >= 2.0*fabs(qc)) {
          qlv_ = q(n,k,j,i) - 2.0*qc;
        }
      }

      //---- set L/R states ----
      ql_jp1(n,i) = qrv_;
      qr_j  (n,i) = qlv_;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX3()
//  \brief Reconstructs linear slope in cell k to cmpute ql(k+1) and qr(k) over [il,iu]
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

void PiecewiseParabolic::ReconstructX3(const int k,const int j,const int il,const int iu,
    const AthenaArray4D<Real> &q, AthenaArray2D<Real> &ql_kp1, AthenaArray2D<Real> &qr_k)
{
  int nvar = q.extent_int(0);
  for (int n=0; n<nvar; ++n) {
    //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
    // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
    // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
    for (int i=il; i<=iu; ++i) {
      Real qlv_ = (7.0*(q(n,k,j,i) + q(n,k-1,j,i)) - (q(n,k-2,j,i) + q(n,k+1,j,i)))/12.0;
      Real qrv_ = (7.0*(q(n,k,j,i) + q(n,k+1,j,i)) - (q(n,k-1,j,i) + q(n,k+2,j,i)))/12.0;

      //---- Apply CS monotonicity limiters to qrv_ and qlv_ ----
      // approximate second derivatives at i-1/2 (PH 3.35) 
      Real d2qc = 3.0*(q(n,k-1,j,i) - 2.0*qlv_ + q(n,k,j,i));
      Real d2ql = (q(n,k-2,j,i) - 2.0*q(n,k-1,j,i) + q(n,k  ,j,i));
      Real d2qr = (q(n,k-1,j,i) - 2.0*q(n,k  ,j,i) + q(n,k+1,j,i));
    
      // limit second derivative (PH 3.36)
      Real d2qlim = 0.0;
      Real lim_slope = std::min(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      } 
      // compute limited value for qlv_ (PH 3.34)
      qlv_ = 0.5*(q(n,k,j,i) + q(n,k-1,j,i)) - d2qlim/6.0;

      // approximate second derivatives at i+1/2 (PH 3.35) 
      d2qc = 3.0*(q(n,k,j,i) - 2.0*qrv_ + q(n,k+1,j,i));
      d2ql = d2qr;
      d2qr = (q(n,k,j,i  ) - 2.0*q(n,k+1,j,i) + q(n,k+2,j,i));

      // limit second derivative (PH 3.36)
      d2qlim = 0.0;
      lim_slope = std::min(fabs(d2ql),fabs(d2qr));
      if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
        d2qlim = SIGN(d2qc)*std::min(1.25*lim_slope,fabs(d2qc));
      }
      // compute limited value for qrv_ (PH 3.33)
      qrv_ = 0.5*(q(n,k,j,i) + q(n,k+1,j,i)) - d2qlim/6.0;

      //---- identify extrema, use smooth extremum limiter ----
      // CS 20 (missing "OR"), and PH 3.31
      Real qa = (qrv_ - q(n,k,j,i))*(q(n,k,j,i) - qlv_);
      Real qb = (q(n,k-1,j,i) - q(n,k,j,i))*(q(n,k,j,i) - q(n,k+1,j,i));
      if (qa <= 0.0 || qb <= 0.0) {
        // approximate secnd derivates (PH 3.37)
        Real d2q  = 6.0*(qlv_ - 2.0*q(n,k,j,i) + qrv_);
        Real d2qc = (q(n,k-1,j,i) - 2.0*q(n,k  ,j,i) + q(n,k+1,j,i));
        Real d2ql = (q(n,k-2,j,i) - 2.0*q(n,k-1,j,i) + q(n,k  ,j,i));
        Real d2qr = (q(n,k  ,j,i) - 2.0*q(n,k+1,j,i) + q(n,k+2,j,i));

        // limit second derivatives (PH 3.38)
        Real d2qlim = 0.0;
        Real lim_slope = std::min(fabs(d2ql),fabs(d2qr));
        lim_slope = std::min(fabs(d2qc),lim_slope);
        if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
          d2qlim = SIGN(d2q)*std::min(1.25*lim_slope,fabs(d2q));
        }
        if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
          d2qlim = SIGN(d2q)*std::min(1.25*lim_slope,fabs(d2q));
        }

        // limit L/R states at extrema (PH 3.39)
        if (d2q == 0.0) {  // revert to donor cell
          qlv_ = q(n,k,j,i);
          qrv_ = q(n,k,j,i);
        } else {  // add limited slope (PH 3.39)
          qlv_ = q(n,k,j,i) + (qlv_ - q(n,k,j,i))*d2qlim/d2q;
          qrv_ = q(n,k,j,i) + (qrv_ - q(n,k,j,i))*d2qlim/d2q;
        }
      } else {
        // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
        Real qc = qrv_ - q(n,k,j,i);
        Real qd = qlv_ - q(n,k,j,i);
        if (fabs(qc) >= 2.0*fabs(qd)) {
          qrv_ = q(n,k,j,i) - 2.0*qd;
        }
        if (fabs(qd) >= 2.0*fabs(qc)) {
          qlv_ = q(n,k,j,i) - 2.0*qc;
        }
      }

      //---- set L/R states ----
      ql_kp1(n,i) = qrv_;
      qr_k  (n,i) = qlv_;
    }
  }
  return;
}
