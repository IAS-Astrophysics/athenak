#ifndef RECONSTRUCT_PPMEP_HPP_
#define RECONSTRUCT_PPMEP_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ppmep.hpp
//! \brief WENO-Z/PPM4 hybrid, aiming for stable extremum- and monotonicity preservation.
//! REFERENCES:
//!
//! Rider, Greenough, Kamm, "Accurate monotonicity- and xtrema-preserving methods through
//! adaptive nonlinear hybridizations
//!
//! P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for Gas-Dynamical
//! Simulations", JCP, 54, 174 (1984)
//!
//! Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
//! non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)
//!
//! Castro M., Costa B., Don W.W. , "High order weighted essentially non-oscillatory
//! WENO-Z schemes for hyperbolic conservation laws" , JCP, 230, 1766 (2011)

#include <math.h>
#include <algorithm>    // max()

#include "athena.hpp"

#include "ppm.hpp"
#include "wenoz.hpp"
//----------------------------------------------------------------------------------------
//! \fn WENOZ()
//! \brief Reconstructs 5th-order polynomial in cell i to compute ql(i+1) and qr(i).
//! Works for any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
Real MINMOD(const Real a, const Real b){
  Real c = 0.0;
  if(a*b >0.0) {
    c = ( fabs(a) < fabs(b) ? a : b);
  }
  return(c);
}

KOKKOS_INLINE_FUNCTION
Real MEDIAN(const Real a, const Real b, const Real c){
  return a + MINMOD(b-a, c-a);
}

KOKKOS_INLINE_FUNCTION
void MONOTONIZE_EDGES(const Real qL_in, const Real qR_in, const Real q, const Real qm, const Real qp, Real &qL_mon, Real &qR_mon){
  Real qLs = MEDIAN(q, qL_in, qm);
  Real qRs = MEDIAN(q, qR_in, qp);
  qL_mon = MEDIAN(q, qLs, 3*q - 2*qRs);
  qR_mon = MEDIAN(q, qRs, 3*q - 2*qLs);
  return;
}


KOKKOS_INLINE_FUNCTION
void PPMEP(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
           const Real &q_ip2, Real &ql_ip1, Real &qr_i) {

  // 5th-order upwind reconstruction
  const Real c1 = 2. / 60.;
  const Real c2 = -13. / 60.;
  const Real c3 = 47. / 60.;
  const Real c4 = 27. / 60.;
  const Real c5 = -3. / 60.;

  const Real q_minus = c1*q_ip2 + c2*q_ip1 + c3*q_i + c4*q_im1 + c5*q_im2;
  const Real q_plus  = c1*q_im2 + c2*q_im1 + c3*q_i + c4*q_ip1 + c5*q_ip2;

  Real new_q_minus;
  Real new_q_plus;

  //monotonize
  MONOTONIZE_EDGES(q_minus, q_plus, q_i, q_im1, q_ip1, new_q_minus, new_q_plus);
  const Real q_mean = (fabs(q_im1) + fabs(q_i) + fabs(q_ip1))/3.0;
  const Real eps = 1.0e-14 * q_mean;

  // check if limiter was triggered
  if ( fabs( new_q_minus - q_minus) > eps || fabs(new_q_plus - q_plus) > eps ) {
    Real q_minus_weno;
    Real q_plus_weno;
    WENOZ(q_im2, q_im1, q_i, q_ip1, q_ip2, q_minus_weno, q_plus_weno);

    if (new_q_minus == q_i || new_q_plus == q_i) {
      //to avoid clipping at extrema, use WENO value
      q_minus_weno = MEDIAN(q_i, q_minus_weno, q_minus);
      q_plus_weno = MEDIAN(q_i, q_plus_weno, q_plus);

      MONOTONIZE_EDGES(q_minus_weno, q_plus_weno, q_i, q_im1, q_ip1, new_q_minus, new_q_plus);
      new_q_minus = MEDIAN(q_minus_weno, new_q_minus, q_minus);
      new_q_plus = MEDIAN(q_plus_weno, new_q_plus, q_plus);
    } else {
      // gradient is too steep, use PPM4
      Real q_minus_ppm;
      Real q_plus_ppm;
      PPM4(q_im2, q_im1, q_i, q_ip1, q_ip2, q_minus_ppm, q_plus_ppm);
      q_minus_ppm = MEDIAN(q_minus, q_minus_ppm, q_minus_weno);
      q_plus_ppm = MEDIAN(q_plus, q_plus_ppm, q_plus_weno);

      MONOTONIZE_EDGES(q_minus_ppm, q_plus_ppm, q_i, q_im1, q_ip1, new_q_minus, new_q_plus);
      new_q_minus = MEDIAN(new_q_minus, q_minus_weno, q_minus);
      new_q_plus = MEDIAN(new_q_plus, q_plus_weno, q_plus);
    }
  }

  ql_ip1 = new_q_plus;
  qr_i = new_q_minus;
  return;
}
#endif // RECONSTRUCT_PPMEP_HPP_
