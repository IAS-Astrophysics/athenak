#ifndef RADIATION_RADIATION_MULTI_FREQ_HPP_
#define RADIATION_RADIATION_MULTI_FREQ_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_tetrad.hpp
//  \brief helper functions for multi-frequency radiation

#include <math.h>

#include "athena.hpp"






KOKKOS_INLINE_FUNCTION
void getFreqAngIndices(const int &ifr_ang, const int &nang, int &ifr, int &iang) {
  ifr  = ifr_ang / nang;
  iang = ifr_ang - ifr*nang;
  return;
}


KOKKOS_INLINE_FUNCTION
int getFreqAngIndex(const int &ifr, const int &iang, const int &nang) {
  int ret = iang + ifr*nang;
  return ret;
}

// nu is defined in simulation unit
KOKKOS_INLINE_FUNCTION
Real BBSpectrum(const Real &nu, const Real &temp, const Real &a_rad) {
  Real ret = 15./SQR(SQR(M_PI)) * a_rad;
  ret *= nu*SQR(nu) / (exp(nu/temp)-1.);
  return ret;
}


KOKKOS_INLINE_FUNCTION
Real HolBBIntSmall(const Real &a) {
  Real a3  = a*SQR(a);
  Real a4  = a*a3;
  Real a5  = a*a4;
  Real a7  = SQR(a)*a5;
  Real a9  = SQR(a)*a7;
  Real a11 = SQR(a)*a9;

  Real ret = a3/3. - a4/8. + a5/60. - a7/5040. + a9/272160. - a11/13305600.;
  ret *= 15./SQR(SQR(M_PI));

  return ret;
}


KOKKOS_INLINE_FUNCTION
Real dHolBBIntSmall(const Real &nu_f, const Real &temp) {
  Real a = nu_f/temp;
  Real da = -nu_f/SQR(temp);

  Real a2  = SQR(a);
  Real a3  = a*a2;
  Real a4  = a*a3;
  Real a6  = a2*a4;
  Real a8  = a2*a6;
  Real a10 = a2*a8;

  Real ret = a2 - a3/2. + a4/12. - a6/720. + a8/30240. - a10/1209600.;
  ret *= 15./SQR(SQR(M_PI)) * da;

  return ret;
}


KOKKOS_INLINE_FUNCTION
Real HolBBIntLarge(const Real &a) {
  int num_itr_max = 100;
  Real err = 1e-12;
  Real tol = err / (15./SQR(SQR(M_PI)));

  Real term1 = a*SQR(a) * log(1.-exp(-a));
  Real term2=0.0, term3=0.0, term4=0.0;

  for (int k=1; k <= num_itr_max; ++k) {
    Real dterm2 = exp(-k*a)/SQR(k);
    Real dterm3 = dterm2/k;
    Real dterm4 = dterm3/k;
    dterm2 = -3*SQR(a) * dterm2;
    dterm3 = -6*a * dterm3;
    dterm4 = -6 * dterm4;
    if (fabs(dterm2) > tol) term2+=dterm2;
    if (fabs(dterm3) > tol) term3+=dterm3;
    if (fabs(dterm4) > tol) term4+=dterm4;
    if ((fabs(dterm2) <= tol) && (fabs(dterm3) <= tol) && (fabs(dterm4) <= tol))
      break;
  }

  Real ret = term1 + term2 + term3 + term4;
  ret = (1. + 15./SQR(SQR(M_PI))*ret);
  return ret;
}


KOKKOS_INLINE_FUNCTION
Real dHolBBIntLarge(const Real &nu_f, const Real &temp) {
  Real a = nu_f/temp;
  Real da = -nu_f/SQR(temp);

  int num_itr_max = 100;
  Real err = 1e-12;
  Real tol = err / (15./SQR(SQR(M_PI))*fabs(da));

  Real term1 = 3*SQR(a) * log(1.-exp(-a));
  Real term2 = a*SQR(a) / (exp(a)-1);
  Real term3=0.0;
  for (int k=1; k <= num_itr_max; ++k) {
    Real dterm3 = 3*SQR(a) * exp(-k*a)/k;
    if (fabs(dterm3) > tol) term3+=dterm3;
    if (fabs(dterm3) <= tol) break;
  }

  Real ret = term1 + term2 + term3;
  ret = 15./SQR(SQR(M_PI)) * ret * da;
  return ret;
}


KOKKOS_INLINE_FUNCTION
Real BBIntegral(const Real &nu_min, const Real &nu_max, const Real &temp, const Real &a_rad) {
  Real nu_T_min = nu_min/temp;
  Real nu_T_max = nu_max/temp;

  Real BBInt_0_min = (nu_T_min <= 0.5) ? HolBBIntSmall(nu_T_min) : HolBBIntLarge(nu_T_min);
  Real BBInt_0_max = (nu_T_max <= 0.5) ? HolBBIntSmall(nu_T_max) : HolBBIntLarge(nu_T_max);

  Real ret = (BBInt_0_max - BBInt_0_min) * a_rad*SQR(SQR(temp));
  return ret;
}


KOKKOS_INLINE_FUNCTION
Real GetEffTemperature(const Real &ir_cm_e, const Real &nu_e, const Real &a_rad) {

  // estimate temp_old first;
  Real temp_old = 1; // TODO: estimate

  Real temp_new = temp_old;
  {
    Real nu_T_e = nu_e/temp_old;
    Real holB  = (nu_T_e <= 0.5) ? HolBBIntSmall(nu_T_e) : HolBBIntLarge(nu_T_e);
    Real dholB = (nu_T_e <= 0.5) ? dHolBBIntSmall(nu_e, temp_old) : dHolBBIntLarge(nu_e, temp_old);

    Real func = (1-holB) * a_rad*SQR(SQR(temp_old))/(4*M_PI) - ir_cm_e;
    Real dfunc = (4./temp_old * (1-holB) - dholB) * a_rad*SQR(SQR(temp_old))/(4*M_PI);

    Real dtemp = -func/dfunc;
    temp_new = temp_old + dtemp;
  }

  return temp_new;
}










//----------------------------------------------------------------------------------------
//! \fn Real PLMRadFreq
//  \brief PLM for radiation reconstruction in frequency domain
//         Default slope is in second-order with central differencing
//         order == 0: zeroth-order slope, return 0
//         order == 1: first-order slope, only use inu_l(nu_l) and inu1(nu1)
//         limiter == 1: minmod limiter
//         limiter == 2: van Leer limiter
//         boundary == 1: use left boundary  (nu_l = nu_cm[f]   = nu_cm[0])
//         boundary == 2: use right boundary (nu_r = nu_cm[f+1] = nu_cm[N-1])
//
// inputs:     (inu0)   (inu_l)  (inu1)   (inu_r)  (inu2)
//      |        |        |        *        |        |        |
//      |        |        |                 |        *        |
//      |        *        |                 |                 |
//      |   nu_cm[f-1/2]  |   nu_cm[f+1/2]  |   nu_cm[f+3/2]  |
// -----|--------|--------|--------|--------|--------|--------|----------->
//   nu_cm[f-1]  |     nu_cm[f]    |     nu_cm[f+1]  |     nu_cm[f+2]
// inputs:     (nu0)    (nu_l)   (nu1)    (nu_r)   (nu2)

KOKKOS_INLINE_FUNCTION
Real GetMultiFreqRadSlope(const Real  &nu0, const Real  &nu_l, const Real  &nu1, const Real  &nu_r, const Real  &nu2,
                          const Real &inu0, const Real &inu_l, const Real &inu1, const Real &inu_r, const Real &inu2,
                          int order, int limiter, int boundary) {
  // Zeroth-order
  if (order == 0)
    return 0;

  // First-order
  if (order == 1) {
    // if ()
    // inu_l, inu1, nu_l, nu1
    return (inu1-inu_l)/(nu1-nu_l);
  }


  // Second-order
  // without limiter
  Real k_ret = (inu2 - inu0) / (nu2 - nu0);
  // left and right boundaries (downgrade to first-order evaluation)
  if (boundary == 1) k_ret = (inu1 - inu_l) / (nu1 - nu_l);
  if (boundary == 2) k_ret = (inu_r - inu1) / (nu_r - nu1);

  // Apply limiters
  if (limiter > 0) {
    // compute L/R slopes
    Real ka = (inu1 - inu0) / (nu1 - nu0);
    Real kb = (inu2 - inu1) / (nu2 - nu1);

    // left and right boundaries
    if (boundary == 1) ka = (inu1 - inu_l) / (nu1 - nu_l);
    if (boundary == 2) kb = (inu_r - inu1) / (nu_r - nu1);

    // minmod limiter
    if (limiter == 1) {
      if (ka*kb > 0) k_ret = SIGN(ka) * fmin(fabs(ka), fabs(kb));
      else k_ret = 0;
    } // endif (limiter == 1)

    // van Leer limiter
    if (limiter == 2) {
      if ((fabs(ka+kb) > FLT_MIN) && (ka*kb > 0))
        k_ret = 2*ka*kb/(ka+kb);
      else k_ret = 0;
    } // endif (limiter == 2)
  } // endif (limiter > 0)

  return k_ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real IntensityFraction
// //  \brief PLM for radiation reconstruction in frequency domain
//

//                                (nu0)                                                                  (nu1)
//                              nu_tet[f]                                                              nu_tet[f+1]
// ............................ ---|----------------------------------------------------------------------|--- ............................ --->
//                           (ir_cm_star_l)                                                        (ir_cm_star_r)
//                            I_cm_star[L]    I_cm_star[L+1]    I_cm_star[...]    I_cm_star[R-1]    I_cm_star[R]
// ... ---|----------------|--------========|================|=== .......... ===|================|========--------|----------------|--- ... --->
//     nu_cm[L-1]       nu_cm[L]         nu_cm[L+1]       nu_cm[L+2]         nu_cm[R-1]       nu_cm[R]         nu_cm[R+1]       nu_cm[R+2]
//     (nu_lm1)         (nu_l)           (nu_lp1)         (nu_lp2)           (nu_rm1)         (nu_r)           (nu_rp1)         (nu_rp2)

KOKKOS_INLINE_FUNCTION
Real IntensityFraction(const Real  &nu_f, const Real  &nu_fp1,
                       const Real  &nu1h, const Real  &nu1, const Real  &nu3h, const Real  &nu2, const Real  &nu5h,
                       const Real &inu1h, const Real &inu1, const Real &inu3h, const Real &inu2, const Real &inu5h,
                       int order, int limiter, int boundary, bool leftbin) {
  // Left Case:                       (nu_f)
  //                                 nu_tet[f]
  // ................................ ---|--- ......................... --->
  //             (inu1h)   (inu1) (inu3h)|  (inu2) (inu5h)
  //        |                |           |    |                |
  //        |       *        |       *   |    |                |
  //        |                |           |    |       *        |
  //        | I_cm_star[L-1] |  I_cm_star[L]  | I_cm_star[L+1] |
  // ... ---|----------------|------------====|----------------|--- ... --->
  //     nu_cm[L-1]       nu_cm[L]         nu_cm[L+1]       nu_cm[L+2]
  //             (nu1h)    (nu1)   (nu3h)   (nu2)   (nu5h)
  //
  // Right Case:                     (nu_fp1)
  //                                nu_tet[f+1]
  // ................................ ---|--- ......................... --->
  //             (inu1h)   (inu1) (inu3h)|  (inu2) (inu5h)
  //        |                |           |    |                |
  //        |       *        |       *   |    |                |
  //        |                |           |    |       *        |
  //        | I_cm_star[R-1] |  I_cm_star[R]  | I_cm_star[R+1] |
  // ... ---|----------------|===========-----|----------------|--- ... --->
  //     nu_cm[R-1]       nu_cm[R]         nu_cm[R+1]       nu_cm[R+2]
  //             (nu1h)    (nu1)   (nu3h)   (nu2)   (nu5h)

  Real k_slope = GetMultiFreqRadSlope(nu1h, nu1, nu3h, nu2, nu5h,
                                      inu1h, inu1, inu3h, inu2, inu5h,
                                      order, limiter, boundary);

  Real nu_min = (leftbin) ? nu_f : fmax(nu1, nu_f) ;
  Real nu_max = (leftbin) ? fmin(nu2, nu_fp1) : nu_fp1 ;
  Real inu_min = inu3h + k_slope*(nu_min - nu3h);
  Real inu_max = inu3h + k_slope*(nu_max - nu3h);
  Real frac_ret = 0.5*(inu_min+inu_max)*(nu_max-nu_min);

  return frac_ret;
}










//
//             (inu1h)  (inu1)  (inu3h)  (inu2)  (inu5h)
//        |                |                |       *        |
//        |       *        |       *        |                |
//        |  ir_cm_star_0  |  ir_cm_star_1  |  ir_cm_star_2  |
//        | I_cm_star[f-1] |  I_cm_star[f]  | I_cm_star[f+1] |
// ... ---|----------------|----------------|----------------|--- ... --->
//     nu_cm[f-1]       nu_cm[f]         nu_cm[f+1]       nu_cm[f+2]
//       nu0    (nu1h)   (nu1)   (nu3h)   (nu2)   (nu5h)    nu3
//
// Note: nu_cm[f] = n0_cm * nu_tet(f)
//       I_cm_star[f] = (n0_cm^4)/(n0*n_0) * i0(m,n,k,j,i);

KOKKOS_INLINE_FUNCTION
bool AssignFreqIntensity(const int &fIdx, const DvceArray1D<Real> &nu_tet, const DvceArray5D<Real> &i0,
                         const int &m, const int &k, const int &j, const int &i, const int &iang,
                         const int &nang, const Real &nfreq1, const Real &n0_cm, const Real &n0, const Real &n_0,
                         const Real &a_rad, const Real &temp,
                         Real &nu0, Real &nu1, Real &nu2, Real &nu3, Real &nu1h, Real &nu3h, Real &nu5h,
                         Real &inu1h, Real &inu3h, Real &inu5h, Real &inu1, Real &inu2,
                         Real &ir_cm_star_1, int &boundary) {

  Real ir_cm_star_0=0, ir_cm_star_2=0;

  nu0 = -1;
  if ((fIdx-1 >= 0) && (fIdx-1 <= nfreq1)) {
    nu0 = n0_cm*nu_tet(fIdx-1);
    int n = getFreqAngIndex(fIdx-1, iang, nang);
    ir_cm_star_0 = SQR(SQR(n0_cm))*i0(m,n,k,j,i)/(n0*n_0);
  }

  nu1 = -1;
  ir_cm_star_1=0;
  if ((fIdx >= 0) && (fIdx <= nfreq1)) {
    nu1 = n0_cm*nu_tet(fIdx);
    int n = getFreqAngIndex(fIdx, iang, nang);
    ir_cm_star_1 = SQR(SQR(n0_cm))*i0(m,n,k,j,i)/(n0*n_0);
  }

  nu2 = -1;
  if ((fIdx+1 >= 0) && (fIdx+1 <= nfreq1)) {
    nu2 = n0_cm*nu_tet(fIdx+1);
    int n = getFreqAngIndex(fIdx+1, iang, nang);
    ir_cm_star_2 = SQR(SQR(n0_cm))*i0(m,n,k,j,i)/(n0*n_0);
  }

  nu3 = ((fIdx+2 >= 0) && (fIdx+2 <= nfreq1)) ? n0_cm*nu_tet(fIdx+2) : -1;

  nu1h  = ((nu0!=-1) && (nu1!=-1)) ? (nu0+nu1)/2 : -1;
  nu3h  = ((nu1!=-1) && (nu2!=-1)) ? (nu1+nu2)/2 : -1;
  nu5h  = ((nu2!=-1) && (nu3!=-1)) ? (nu2+nu3)/2 : -1;

  inu1h = ((nu0!=-1) && (nu1!=-1)) ? ir_cm_star_0 / (nu1-nu0) : 0;
  inu3h = ((nu1!=-1) && (nu2!=-1)) ? ir_cm_star_1 / (nu2-nu1) : 0;
  inu5h = ((nu2!=-1) && (nu3!=-1)) ? ir_cm_star_2 / (nu3-nu2) : 0;

  inu1 = -1; inu2 = -1; boundary = 0;
  if (fIdx == 0) {
    inu1 = 0;
    boundary = 1;
  } else if (fIdx+1 == nfreq1) {
    inu2 = BBSpectrum(nu2, temp, a_rad)/(4*M_PI);
    boundary = 2;
  }

  if ((fIdx < 0) && (fIdx >= nfreq1)) return false;
  else return true;
}














































// computes
// KOKKOS_INLINE_FUNCTION
// void Compute(Real nu_min, Real nu_max, int nfreq, string flag) {
//   // if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);  // see cartesian_ks.hpp comments
//
//   return;
// }






#endif // RADIATION_RADIATION_MULTI_FREQ_HPP_
