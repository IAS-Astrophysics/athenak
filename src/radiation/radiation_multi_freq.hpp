#ifndef RADIATION_RADIATION_MULTI_FREQ_HPP_
#define RADIATION_RADIATION_MULTI_FREQ_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_multi_freq.hpp
//  \brief helper functions for multi-frequency radiation

#include <math.h>
#include "athena.hpp"

//=================================== Index Operations ===================================
//----------------------------------------------------------------------------------------
//! \fn void getFreqAngIndices
//  \brief Exact frequency index (ifr) and angular index (iang) given
//         the frequency-angular index (ifr_ang).
KOKKOS_INLINE_FUNCTION
void getFreqAngIndices(const int &ifr_ang, const int &nang, int &ifr, int &iang) {
  ifr  = ifr_ang / nang;
  iang = ifr_ang - ifr*nang;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn int getFreqAngIndex
//  \brief Return frequency-angular index (ifr_ang) given
//         frequency index (ifr) and angular index (iang).
KOKKOS_INLINE_FUNCTION
int getFreqAngIndex(const int &ifr, const int &iang, const int &nang) {
  int ret = iang + ifr*nang;
  return ret;
}

//============================== Blackbody Helper Functions ==============================
//----------------------------------------------------------------------------------------
//! \fn Real BBSpectrum
//  \brief Compute blackbody spectrum given frequency (nu).
//         All inputs are given in simulation units.
//         B_{nu} = 15*arad/pi^4 * nu^3/(exp(nu/temp)-1)
KOKKOS_INLINE_FUNCTION
Real BBSpectrum(const Real &nu, const Real &temp, const Real &a_rad) {
  Real ret = 15./SQR(SQR(M_PI)) * a_rad;
  ret *= nu*SQR(nu) / (exp(nu/temp)-1.);
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real HolBBIntSmall
//  \brief Compute the integration of dimensionless blackbody spectrum from
//         0 to small a, where a is the ratio of frequency and temperature.
//         This is approximated by the expansion near a=0.
//         All inputs are given in simulation units.
//         Return 1/(arad*temp^4) \int_0^a B_{nu/temp} d(nu/temp)
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

//----------------------------------------------------------------------------------------
//! \fn Real HolBBIntLarge
//  \brief Compute the integration of dimensionless blackbody spectrum from
//         0 to large a, where a is the ratio of frequency and temperature.
//         This is approximated by the polylogarithm functions, which
//         converges fast for large a.
//         All inputs are given in simulation units.
//         Return 1/(arad*temp^4) \int_0^a B_{nu/temp} d(nu/temp)
KOKKOS_INLINE_FUNCTION
Real HolBBIntLarge(const Real &a) {
  int num_itr_max = 50;
  Real err = 1e-12;
  Real tol = err / (15./SQR(SQR(M_PI)));

  Real a2  = a*a;
  Real a3  = a*a2;
  Real term1 = a3 * log(1.-exp(-a));
  Real term2=0.0, term3=0.0, term4=0.0;

  for (int k=1; k <= num_itr_max; ++k) {
    Real dterm2 = exp(-k*a)/SQR(k);
    Real dterm3 = dterm2/k;
    Real dterm4 = dterm3/k;
    dterm2 = -3*a2 * dterm2;
    dterm3 = -6*a * dterm3;
    dterm4 = -6 * dterm4;
    term2 += dterm2;
    term3 += dterm3;
    term4 += dterm4;
    if ((fabs(dterm2) < tol) && (fabs(dterm3) < tol) && (fabs(dterm4) < tol))
      break;
  }

  Real ret = term1 + term2 + term3 + term4;
  ret = (1. + 15./SQR(SQR(M_PI))*ret);
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real HolBBInt
//  \brief Compute the integration of dimensionless blackbody spectrum from
//         0 to a, where a is the ratio of frequency and temperature.
//         This is defined as:
//                    = HolBBIntSmall(a), when a <= 0.5
//                    = HolBBIntLarge(a), when a >  0.5
//         All inputs are given in simulation units.
//         Return 1/(arad*temp^4) \int_0^a B_{nu/temp} d(nu/temp)
KOKKOS_INLINE_FUNCTION
Real HolBBInt(const Real &a) {
  Real ret = (a <= 0.5) ? HolBBIntSmall(a) : HolBBIntLarge(a);
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real BBIntegral
//  \brief Compute the integration of blackbody spectrum from nu_min to nu_max.
//         All inputs and outputs are in simulation units.
//         Return \int_{nu_min}^{nu_max} B_{nu} d(nu)
KOKKOS_INLINE_FUNCTION
Real BBIntegral(const Real &nu_min, const Real &nu_max, const Real &temp, const Real &a_rad) {
  Real ret = HolBBInt(nu_max/temp) - HolBBInt(nu_min/temp);
  ret *= a_rad*SQR(SQR(temp));
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real dBcapSmall
//  \brief Compute the derivative of the dimensionless blackbody integration
//         (from 0 to small a, where a is the ratio of frequency and temperature).
//         This function is used in Newton-Raphson method.
//         All inputs and outputs are in simulation units.
//         Return d\mathbb{B}_{small}/da
KOKKOS_INLINE_FUNCTION
Real dBcapSmall(const Real &a) {
  Real a2  = a*a;
  Real a3  = a*a2;
  Real a4  = a*a3;
  Real a6  = a2*a4;
  Real a8  = a2*a6;
  Real a10 = a2*a8;

  Real ret = a2 - a3/2. + a4/12. - a6/720. + a8/30240. - a10/1209600.;
  ret *= 15./SQR(SQR(M_PI));

  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real dBcapLarge
//  \brief Compute the derivative of the dimensionless blackbody integration
//         (from 0 to large a, where a is the ratio of frequency and temperature).
//         This function is used in Newton-Raphson method.
//         All inputs and outputs are in simulation units.
//         Return d\mathbb{B}_{large}/da
KOKKOS_INLINE_FUNCTION
Real dBcapLarge(const Real &a) {
  int num_itr_max = 50;
  Real err = 1e-12;
  Real tol = err / (15./SQR(SQR(M_PI)));

  Real a2  = a*a;
  Real a3  = a*a2;
  Real term1 = 3*a2 * log(1.-exp(-a));
  Real term2 = a3 / (exp(a)-1.);
  Real term3=0.0;
  for (int k=1; k <= num_itr_max; ++k) {
    Real dterm3 = 3*a2 * exp(-k*a)/k;
    term3 += dterm3;
    if (fabs(dterm3) < tol) break;
  }

  Real ret = term1 + term2 + term3;
  ret = 15./SQR(SQR(M_PI)) * ret;
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real dBcapHuge
//  \brief Compute the derivative of the dimensionless blackbody integration
//         (from 0 to huge a, where a is the ratio of frequency and temperature),
//         since dBcapLarge fails to capture correct derivative when a is huge.
//         This function is used in Newton-Raphson method.
//         All inputs and outputs are in simulation units.
//         Return d\mathbb{B}_{huge}/da
KOKKOS_INLINE_FUNCTION
Real dBcapHuge(const Real &a) {
  return 15./SQR(SQR(M_PI)) * a*SQR(a) * exp(-a);
}

//----------------------------------------------------------------------------------------
//! \fn Real dBcapTemp
//  \brief Compute the derivative of the dimensionless blackbody
//         integration (from 0 to any nu/temp).
//         This function is used in Newton-Raphson method.
//         All inputs and outputs are in simulation units.
//         Return d\mathbb{B}/dtemp = d\mathbb{B}/da * da/dtemp
KOKKOS_INLINE_FUNCTION
Real dBcapTemp(const Real &temp, const Real &nu_f) {
  Real a = nu_f/temp;
  Real da_ = -nu_f/SQR(temp);
  if (a <= 0.5)
    return dBcapSmall(a)*da_;
  else if (a <= 31.609864819846077)
    return dBcapLarge(a)*da_;
  else
    return dBcapHuge(a)*da_;
}

//============================= Mathmatical Helper Functions =============================
//----------------------------------------------------------------------------------------
//! \fn bool FourthPolyRoot
//  \brief Exact solution for fourth order polynomial of
//         the form coeff4 * x^4 + x + coeff0 = 0.
KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coeff4, const Real coeff0, Real &root) {
  // Calculate real root of z^3 - 4*coeff0/coeff4 * z - 1/coeff4^2 = 0
  Real ccubic = coeff0 * coeff0 * coeff0;
  Real delta1 = 0.25 - 64.0 * ccubic * coeff4 / 27.0;
  if (delta1 < 0.0) {
    return false;
  }
  delta1 = sqrt(delta1);
  if (delta1 < 0.5) {
    return false;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = pow(delta1, -2.0/3.0) / 3.0;
  } else {
    zroot = pow(0.5 + delta1, 1.0/3.0) - pow(-0.5 + delta1, 1.0/3.0);
  }
  if (zroot < 0.0) {
    return false;
  }
  zroot *= pow(coeff4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coeff4 * rcoef);
  if (delta2 < 0.0) {
    return false;
  }
  delta2 = sqrt(delta2);
  root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool InverseMatrix
//  \brief Inverse square matrix A with NxN elements.
//         Augmented matrix aug in Nx2N is provided for Gauss-Jordan elimination.
//         Return matrix inverse inv if A is not singular
KOKKOS_INLINE_FUNCTION
bool InverseMatrix(const int N, const ScrArray2D<Real> &A, ScrArray2D<Real> &aug, ScrArray2D<Real> &inv) {
  bool doublecheck = true;
  Real tol = 1e-2;

  // Create augmented matrix [A | I]
  // Real aug[N][2*N];
  for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
          aug(i,j) = A(i,j);
      }
      for (int j=N; j<2*N; j++) {
          aug(i,j) = (i==(j-N)) ? 1 : 0;  // identity matrix
      }
  } // endfor i

  // Gauss-Jordan elimination
  for (int k=0; k<N; k++) {
      // Partial pivoting
      int pivotRow = k;
      Real maxVal = fabs(aug(k,k));
      for (int i=k+1; i<N; i++) {
          if (fabs(aug(i,k)) > maxVal) {
              maxVal = fabs(aug(i,k));
              pivotRow = i;
          }
      }

      // If pivot is zero, matrix is singular
      if (fabs(aug(pivotRow,k)) < 1e-20) {
          return false; // Singular matrix
      }

      // Swap rows if necessary
      if (pivotRow != k) {
          for (int j = 0; j < 2*N; j++) {
              Real tmp = aug(k,j);
              aug(k,j) = aug(pivotRow,j);
              aug(pivotRow,j) = tmp;
          }
      }

      // Normalize pivot row
      Real pivot = aug(k,k);
      for (int j=0; j<2*N; j++) {
          aug(k,j) /= pivot;
      }

      // Eliminate other rows
      for (int i=0; i<N; i++) {
          if (i != k) {
              Real factor = aug(i,k);
              for (int j=0; j<2*N; j++) {
                  aug(i,j) -= factor * aug(k,j);
              }
          }
      }
  } // endfor k

  // Extract inverse
  for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
          inv(i,j) = aug(i,j+N);
      }
  } // endfor i

  // Double check
  if (doublecheck) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
          Real sum = 0.0;
          for (int k=0; k<N; k++) {
            sum += A(i,k) * inv(k,j);
          }
          Real err = (i==j) ? fabs(1-sum) : fabs(sum);
          if (err > tol) return false; // matrix A is ill-conditioned
        } // endfor j
    } // endfor i
  } // endif doublecheck

  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool UpdateFluidFrameIntensity
//  \brief Given triangular linear system Ax=b, where
//         mapping matrix A is either upper- or lower-triangular
//         and b is updated fluid-frame intensity defined in tetrad-frame frequency bins,
//         solve updated fluid-frame intensity x defined in fluid-frame frequency bins
//         by backward or forward substitution.
//         Return false if A is singular or any intensity x is negative
KOKKOS_INLINE_FUNCTION
bool SolveTriLinearSystem(const int N, const ScrArray2D<Real> &A, const ScrArray2D<Real> &ir_cm,
                          const int &iang, const Real &n0_cm, ScrArray2D<Real> &ir_cm_star) {
  if (n0_cm >= 1) {
    // lower-triangular, forward substitution
    for (int i=0; i < N; ++i) {
      Real sum = ir_cm(iang,i);
      for (int j=0; j < i; ++j) {
        sum -= A(i,j) * ir_cm_star(iang,j);
      } // endfor j
      if (A(i,i) == 0.0) return false; // singular
      Real ir_update = sum / A(i,i);
      if (ir_update < 0) return false; // negative intensity
      ir_cm_star(iang,i) = ir_update;
    } // endfor i
  } else {
    // upper-triangular, backward substitution
    for (int i=N-1; i >= 0; --i) {
      Real sum = ir_cm(iang,i);
      for (int j=i+1; j < N; ++j) {
        sum -= A(i,j) * ir_cm_star(iang,j);
      } // endfor j
      if (A(i,i) == 0.0) return false; // singular
      Real ir_update = sum / A(i,i);
      if (ir_update < 0) return false; // negative intensity
      ir_cm_star(iang,i) = ir_update;
    } // endfor i
  } // endelse

  return true;
}

//============================= Effective Temperature Functions =============================
//----------------------------------------------------------------------------------------
//! \fn Real GuessEffTemperature
//  \brief Make an initial guess of the effective temperature in terms of a (nu/temp)
//         given Acap, which is defined as Acap = 4*pi/arad * ir_cm/nu_cm^4.
//         The estimation is based on the regime of Acap, which is
//         negatively correlated with a.
KOKKOS_INLINE_FUNCTION
Real GuessEffTemperature(const Real &Acap) {
  // parameters
  Real Acap_mid  = 0.9653823091764577; // Acap_mid = A(1)
  Real Acap_tiny = 1e-16; // Acap_tiny = A(31.609864819846077)
  Real a = 0;

  if (Acap >= Acap_mid) {
    // small a (a <= 1)
    Real coeff4 = -SQR(SQR(M_PI))/5;
    Real coeff0 = SQR(SQR(M_PI))/5*Acap - 3./8;
    Real a_inv;
    bool flag = FourthPolyRoot(coeff4, coeff0, a_inv);
    if (!(flag) || !(isfinite(a)))
      a_inv = pow(Acap - 15./(8*SQR(SQR(M_PI))), 0.25);
    a = 1./a_inv;

  } else if (Acap >= Acap_tiny) {
    // large a (1 < a <= 31.609864819846077)
    Real c0 = 0.964272215486288;
    Real c1 = -0.529732261375545;
    Real c2 = 0.24130597940731527;
    Real c3 = 0.01573219167683236;
    Real c4 = 0.0003799940235742786;
    Real lg_Acap = log10(Acap);
    Real lg_Acap2 = SQR(lg_Acap);
    Real lg_Acap3 = lg_Acap2*lg_Acap;
    Real lg_Acap4 = lg_Acap3*lg_Acap;
    a = c4*lg_Acap4 + c3*lg_Acap3 + c2*lg_Acap2 + c1*lg_Acap + c0;

  } else {
    // huge a
    a = 31.609864819846077;
  } // endelse

  return a;
}

//----------------------------------------------------------------------------------------
//! \fn Real DelNuTInvNR
//  \brief Compute the difference for the update of effective temperature
//         in terms of 1/a (a=nu/temp) given initial guess a
//         and target Acap (Acap = 4*pi/arad * ir_cm/nu_cm^4).
//         This function is used in Newton-Raphson method,
//         where (1/a_new) = 1/a_old + DelNuTInvNR(a,Acap)
KOKKOS_INLINE_FUNCTION
Real DelNuTInvNR(const Real &a, const Real &Acap) {
  Real num_itr_max = 50;
  Real err_rel = 1e-12;

  Real a2 = SQR(a);
  Real a3 = a*a2;
  Real a4 = a*a3;

  Real Bcap = HolBBInt(a);
  Real g_tar = (1.-Bcap)/a4 - Acap;
  Real dg_tar = 4*(1.-Bcap)/a3;

  if (a <= 0.5) {
    // small a
    dg_tar += dBcapSmall(a) / a2;

  } else if (a > 31.609864819846077) {
    // huge a
    dg_tar += dBcapHuge(a) / a2;

  } else {
    // large a
    dg_tar += (3*log(1.-exp(-a)) + a/(exp(a)-1.)) * 15./SQR(SQR(M_PI));
    for (int k=1; k <= num_itr_max; ++k) {
      Real diff = 3*exp(-k*a)/k * 15./SQR(SQR(M_PI));
      dg_tar += diff;
      if (fabs(diff/dg_tar) < err_rel) break;
    } // endfor k
  } // endelse

  return -g_tar/dg_tar;
}

//----------------------------------------------------------------------------------------
//! \fn Real DelNuTNR
//  \brief Compute the difference for the update of effective temperature
//         in terms of a (a=nu/temp) given initial guess a
//         and target Acap (Acap = 4*pi/arad * ir_cm/nu_cm^4).
//         This function is used in Newton-Raphson method,
//         where a_new = a_old + DelNuTNR(a,Acap)
KOKKOS_INLINE_FUNCTION
Real DelNuTNR(const Real &a, const Real &Acap) {
  Real num_itr_max = 50;
  Real err_rel = 1e-12;

  Real a2 = SQR(a);
  Real a3 = a*a2;
  Real a4 = a*a3;

  Real h_tar = Acap*a4 + HolBBInt(a) - 1.;
  Real dh_tar = 4*Acap*a3;

  if (a <= 0.5) {
    // small a
    dh_tar += dBcapSmall(a);

  } else if (a > 31.609864819846077) {
    // huge a
    dh_tar += dBcapHuge(a);

  } else {
    // large a
    dh_tar += (3*a2*log(1.-exp(-a)) + a3/(exp(a)-1.)) * 15./SQR(SQR(M_PI));
    for (int k=1; k <= num_itr_max; ++k) {
      Real diff = 3*a2*exp(-k*a)/k * 15./SQR(SQR(M_PI));
      dh_tar += diff;
      if (fabs(diff/dh_tar) < err_rel) break;
    } // endfor k
  } // endelse

  return -h_tar/dh_tar;
}

//----------------------------------------------------------------------------------------
//! \fn Real EffTempTarFunc
//  \brief Compute the target function given the effective temperature in
//         terms of a and target Acap (Acap = 4*pi/arad * ir_cm/nu_cm^4).
//         The target effective temperature is found when the target function
//         is zero. Target function F = A*a^4 + mathbb{B} - 1
//         This function is used in bisection iteration.
//         Note that we adapt this function when a is huge (i.e., Acap is tiny).
KOKKOS_INLINE_FUNCTION
Real EffTempTarFunc(const Real &a, const Real &Acap) {
  Real Acap_tiny = 1e-16; // Acap_tiny = A(31.609864819846077)
  Real a2 = SQR(a);
  Real a3 = a*a2;
  Real a4 = a*a3;

  Real ret = Acap*a4;
  if (Acap > Acap_tiny)
    ret += HolBBInt(a) - 1;
  else
    ret += -15./SQR(SQR(M_PI)) * (a3+3*a2+6*a+6) * exp(-a);
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real GetEffTemperature
//  \brief Compute the effective temperature iteratively given the fluid-frame
//         radiation intensity (ir_cm_e) in the last frequency bin [nu_cm_e, +inf].
//         When a is small or large, Newton-Raphson method is adopted. When a is
//         huge, we use bisection method as the derivatives used in Newton-Raphson
//         is too small. This can guarantee the root finding within 50 iterations with
//         accuracy below 1e-12 in target function.
KOKKOS_INLINE_FUNCTION
Real GetEffTemperature(const Real &ir_cm_e, const Real &nu_cm_e, const Real &a_rad) {
  Real tol = 1e-12;
  int num_itr_max = 50;
  Real lg_a_step = 0.05;
  int num_itr_max_search = 100;

  // parameters
  Real Acap_mid = 0.9653823091764577; // Acap_mid = A(1)
  Real Acap_tiny = 1e-16; // Acap_tiny = A(31.609864819846077)

  // estimate temperature;
  Real Acap = 4*M_PI/a_rad * ir_cm_e/SQR(SQR(nu_cm_e));
  bool useBisection = (Acap <= Acap_tiny) ? true : false;
  Real a_ini = GuessEffTemperature(Acap);
  Real lg_a_ini = log10(a_ini);

  // initialize left and right for bisection (if applicable)
  Real lg_a0=lg_a_ini, lg_a1=lg_a_ini;
  if (useBisection) { // a >= 31.609864819846077
    for (int n=1; n<=num_itr_max_search; ++n) { // most finished within 10 iterations
      if (EffTempTarFunc(a_ini, Acap) >= 0) {
        lg_a0 -= lg_a_step;
        if (EffTempTarFunc(pow(10.,lg_a0), Acap) < 0) break;
      } else {
        lg_a1 += lg_a_step;
        if (EffTempTarFunc(pow(10.,lg_a1), Acap) >= 0) break;
      } // endelse
      if (n == num_itr_max_search) useBisection = false;
    } // endfor n
  } // endif useBisection

  // solve effective temperature
  Real lg_a_new = (lg_a0+lg_a1)/2 - 1.; // del_lg_a in first iteration is 1
  Real a_new = a_ini;
  for (int m=1; m<=num_itr_max; ++m) {
    if (useBisection) {
      // use bisection
      Real del_lg_a = (lg_a0+lg_a1)/2 - lg_a_new;
      lg_a_new = (lg_a0+lg_a1)/2;
      a_new = pow(10., lg_a_new);

      if (EffTempTarFunc(a_new, Acap) < 0)
        lg_a0 = lg_a_new;
      else
        lg_a1 = lg_a_new;

      if (fabs(del_lg_a) < tol) break;

    } else {
      // use Newton-Raphson
      Real del_a = 0.0;
      if (Acap >= Acap_mid) {
        // a is small
        Real del_a_inv = DelNuTInvNR(a_new, Acap);
        del_a = 1./(1./a_new + del_a_inv) - a_new;

      } else {
        // a is large
        del_a = DelNuTNR(a_new, Acap);

      } // endelse

      a_new += del_a;
      if (fabs(del_a) < tol) break;

    } // endelse !useBisection
  } // endfor m

  return nu_cm_e/a_new;
}

//================================= Emissivity Functions =================================
//----------------------------------------------------------------------------------------
//! \fn Real ComputeEmissivity
//  \brief Compute the thermal emissivity given frequency bin [nu_f, nu_fp1].
//         eps_f = arad*tgas^4/(4*pi) * ( \mathbb{B}(nu_fp1) - \mathbb{B}(nu_f) )
KOKKOS_INLINE_FUNCTION
Real ComputeEmissivity(const DvceArray1D<Real> &nu_tet, const int &ifr, const Real &tgas, const Real &a_rad) {
  int nfreq1 = nu_tet.extent_int(0)-1;

  Real eps_f = (ifr < nfreq1) ? BBIntegral(0, nu_tet(ifr+1), tgas, a_rad)
                              : a_rad*SQR(SQR(tgas));
  eps_f -= BBIntegral(0, nu_tet(ifr), tgas, a_rad);
  eps_f = 1./(4*M_PI) * fmax(0., eps_f);

  return eps_f;
}

//----------------------------------------------------------------------------------------
//! \fn Real ComputeEmDerivative
//  \brief Compute the derivative of thermal emissivity as a function of gas temperature
//         given frequency bin [nu_f, nu_fp1].
//         deps_f/dtgas = arad*tgas^4/(4*pi) * [
//                          4/tgas * ( \mathbb{B}(nu_fp1) - \mathbb{B}(nu_f) )
//                        + ( (d\mathbb{B}/dtgas)(nu_fp1) - (d\mathbb{B}/dtgas)(nu_f) ) ]
//         This function is used in updating gas temperature in gas-radiation coupling.
KOKKOS_INLINE_FUNCTION
Real ComputeEmDerivative(const DvceArray1D<Real> &nu_tet, const int &ifr, const Real &tgas, const Real &a_rad) {
  int nfreq1 = nu_tet.extent_int(0)-1;

  Real holB_fp1  = (ifr < nfreq1) ? HolBBInt(nu_tet(ifr+1)/tgas)   : 1;
  Real dholB_fp1 = (ifr < nfreq1) ? dBcapTemp(tgas, nu_tet(ifr+1)) : 0;

  Real deps_f = 4./tgas*(holB_fp1 - HolBBInt(nu_tet(ifr)/tgas));
  deps_f += dholB_fp1 - dBcapTemp(tgas, nu_tet(ifr));
  deps_f *= a_rad*SQR(SQR(tgas))/(4*M_PI);

  return deps_f;
}

//============================= Fluid-Frame Intensity Mapping ============================
//----------------------------------------------------------------------------------------
//! \fn bool AssignFreqIntensity
//  \brief Assign the intensities and corresponding frequency bins to prepare
//         intensity reconstruction from fluid frame to tetrad frame.
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

  if ((fIdx < 0) || (fIdx >= nfreq1)) return false;
  else return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool AssignFreqIntensityInv
//  \brief Assign the intensities and corresponding frequency bins to prepare
//         intensity reconstruction from tetrad frame to fluid frame.
//
//             (inu1h)  (inu1)  (inu3h)  (inu2)  (inu5h)
//        |                |                |       *        |
//        |       *        |       *        |                |
//        |    ir_cm_0     |    ir_cm_1     |    ir_cm_2     |
//        |    I_cm[f-1]   |    I_cm[f]     |    I_cm[f+1]   |
// ... ---|----------------|----------------|----------------|--- ... --->
//     nu_tet[f-1]      nu_tet[f]        nu_tet[f+1]      nu_tet[f+2]
//       nu0    (nu1h)   (nu1)   (nu3h)   (nu2)   (nu5h)    nu3
//
// Note: nu_cm[f] = n0_cm * nu_tet(f)
KOKKOS_INLINE_FUNCTION
bool AssignFreqIntensityInv(const int &fIdx, const DvceArray1D<Real> &nu_tet, const ScrArray2D<Real> &ir_cm_update,
                            const int &iang, const Real &nfreq1, const Real &a_rad, const Real &temp,
                            Real &nu0, Real &nu1, Real &nu2, Real &nu3, Real &nu1h, Real &nu3h, Real &nu5h,
                            Real &inu1h, Real &inu3h, Real &inu5h, Real &inu1, Real &inu2,
                            Real &ir_cm_1, int &boundary) {

  Real ir_cm_0=0, ir_cm_2=0;

  nu0 = -1;
  if ((fIdx-1 >= 0) && (fIdx-1 <= nfreq1)) {
    nu0 = nu_tet(fIdx-1);
    ir_cm_0 = ir_cm_update(iang, fIdx-1);
  }

  nu1 = -1;
  ir_cm_1=0;
  if ((fIdx >= 0) && (fIdx <= nfreq1)) {
    nu1 = nu_tet(fIdx);
    ir_cm_1 = ir_cm_update(iang, fIdx);
  }

  nu2 = -1;
  if ((fIdx+1 >= 0) && (fIdx+1 <= nfreq1)) {
    nu2 = nu_tet(fIdx+1);
    ir_cm_2 = ir_cm_update(iang, fIdx+1);
  }

  nu3 = ((fIdx+2 >= 0) && (fIdx+2 <= nfreq1)) ? nu_tet(fIdx+2) : -1;

  nu1h  = ((nu0!=-1) && (nu1!=-1)) ? (nu0+nu1)/2 : -1;
  nu3h  = ((nu1!=-1) && (nu2!=-1)) ? (nu1+nu2)/2 : -1;
  nu5h  = ((nu2!=-1) && (nu3!=-1)) ? (nu2+nu3)/2 : -1;

  inu1h = ((nu0!=-1) && (nu1!=-1)) ? ir_cm_0 / (nu1-nu0) : 0;
  inu3h = ((nu1!=-1) && (nu2!=-1)) ? ir_cm_1 / (nu2-nu1) : 0;
  inu5h = ((nu2!=-1) && (nu3!=-1)) ? ir_cm_2 / (nu3-nu2) : 0;

  inu1 = -1; inu2 = -1; boundary = 0;
  if (fIdx == 0) {
    inu1 = 0;
    boundary = 1;
  } else if (fIdx+1 == nfreq1) {
    inu2 = BBSpectrum(nu2, temp, a_rad)/(4*M_PI);
    boundary = 2;
  }

  if ((fIdx < 0) || (fIdx >= nfreq1)) return false;
  else return true;
}

//----------------------------------------------------------------------------------------
//! \fn Real GetMultiFreqRadSlope
//  \brief Compute the slope from the PLM reconstruction for radiation intensity
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
    Real ka = (boundary == 1) ? (inu1-inu_l) / (nu1-nu_l)
                              : (inu1-inu0)  / (nu1-nu0);
    Real kb = (boundary == 2) ? (inu_r-inu1) / (nu_r-nu1)
                              : (inu2-inu1)  / (nu2-nu1);
    // forward
    Real k_ret = ka;

    // minmod limiter
    if (limiter == 1)
      k_ret = (ka*kb > 0) ? SIGN(ka)*fmin(fabs(ka),fabs(kb)) : 0;

    // van Leer limiter
    if (limiter == 2)
      k_ret = ((fabs(ka+kb) > FLT_MIN) && (ka*kb > 0))
            ? 2*ka*kb/(ka+kb) : 0;

    return k_ret;
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
//  \brief Compute the left/right fraction of intensity according to PLM reconstruction
//         for radiation intensity given frequency bins in different frames
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
  int limiter_all[3];
  limiter_all[0] = limiter;
  limiter_all[1] = 2; limiter_all[2] = 1; // (limiter == 0)
  if (limiter == 1) {
    limiter_all[1] = 2;
    limiter_all[2] = 0;
  } else if (limiter == 2) {
    limiter_all[1] = 1;
    limiter_all[2] = 0;
  }

  Real ifrac_ret = 0;
  for (int n_lim=0; n_lim < 3; ++n_lim) {
    int limiter_use = limiter_all[n_lim];
    Real k_slope = GetMultiFreqRadSlope(nu1h, nu1, nu3h, nu2, nu5h,
                                        inu1h, inu1, inu3h, inu2, inu5h,
                                        order, limiter_use, boundary);
    Real nu_min = (leftbin) ? nu_f : fmax(nu1, nu_f) ;
    Real nu_max = (leftbin) ? fmin(nu2, nu_fp1) : nu_fp1 ;
    Real inu_min = inu3h + k_slope*(nu_min - nu3h);
    Real inu_max = inu3h + k_slope*(nu_max - nu3h);
    ifrac_ret = 0.5*(inu_min+inu_max)*(nu_max-nu_min);
    if (ifrac_ret > 0) break;
  } // endfor n_lim

  // backup routine (0th order)
  if (ifrac_ret < 0) {
    Real nu_min = (leftbin) ? nu_f : fmax(nu1, nu_f) ;
    Real nu_max = (leftbin) ? fmin(nu2, nu_fp1) : nu_fp1 ;
    ifrac_ret = fmax(0, inu3h*(nu_max-nu_min));
  }

  return ifrac_ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real MapIntensity
//  \brief Map the radiation intensity from fluid-frame frequency bins to
//         tetrad-frame frequency bins, with the radiation intensity at ifr-th
//         frequency bin and iang-th angular bin, and the option to output
//         ifr-th row of mapping matrix by setting update_matrix_row=true.
KOKKOS_INLINE_FUNCTION
Real MapIntensity(const int &ifr, const DvceArray1D<Real> &nu_tet, const DvceArray5D<Real> &i0,
                  const int &m, const int &k, const int &j, const int &i, const int &iang,
                  const Real &n0_cm, const Real &n0, const Real &n_0, const Real &a_rad,
                  int order, int limiter, ScrArray2D<Real> &matrix_imap, bool update_matrix_row) {
  // target frequency and intensity
  Real ir_cm_f = 0.0; // value to be assigned
  Real &nu_f = nu_tet(ifr);

  // parameters
  int nfr_ang = i0.extent_int(1);
  int nfrq = nu_tet.extent_int(0);
  int nang = nfr_ang/nfrq;
  int nfreq1 = nfrq-1;
  Real &nu_e = nu_tet(nfreq1);

  // auxiliary variables for intensity mapping
  Real nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h;
  Real inu1, inu2, inu1h, inu3h, inu5h;
  Real ir_cm_star_1; int boundary;

  // get effective temperature at last frequency bin
  int ne = getFreqAngIndex(nfreq1, iang, nang);
  Real ir_cm_star_e = SQR(SQR(n0_cm))*i0(m,ne,k,j,i)/(n0*n_0);
  Real teff = GetEffTemperature(ir_cm_star_e, n0_cm*nu_e, a_rad);

  // locate left & right fluid-frame frequency bins (if exist)
  // -1 <= idx_l   <= N-2
  //  0 <= idx_l+1 <= N-1
  int idx_lp1=0;
  while ((n0_cm*nu_tet(idx_lp1) < nu_f) && (idx_lp1 <= nfreq1)) idx_lp1++;
  int idx_l = idx_lp1-1;
  // -1 <= idx_r   <= N-2
  //  0 <= idx_r+1 <= N-1
  int idx_r=-1, idx_rp1=-1;
  if (ifr+1 <= nfreq1) {
    idx_rp1=fmax(idx_l,0);
    Real &nu_fp1 = nu_tet(ifr+1);
    while ((n0_cm*nu_tet(idx_rp1) <= nu_fp1) && (idx_rp1 <= nfreq1)) idx_rp1++;
    idx_r = idx_rp1-1;
  }

  // get mapping coefficients
  if (nu_f >= n0_cm*nu_e) { // only happen when (n0_cm < 1)
    // This covers the rightmost corner case (f = N-1) && (n0_cm < 1)
    //
    // Corner Case 4: Rightmost (nu_tet[f] >= nu_cm[N-1] && n0_cm < 1)
    //
    // When f <= N-2,                        When f = N-1,
    //       (nu_f)          (nu_fp1)               (nu_f)             (nu_fp1)
    //     nu_tet[N-2]      nu_tet[N-1]           nu_tet[N-1]
    // --------|----------------|---> inf     --------|----------------> inf
    //         | (ir_cm_star_e) |                     | (ir_cm_star_e)
    //         | I_cm_star[N-1] |                     | I_cm_star[N-1]
    // -----|---================----> inf     -----|---================> inf
    //  nu_cm[N-1]                             nu_cm[N-1]
    //    (nu1)                                  (nu1)
    Real integral_f = 0.0;
    if (ifr+1 <= nfreq1) { // ifr <= N-2
      Real &nu_fp1 = nu_tet(ifr+1);
      integral_f = 1./(4*M_PI) * BBIntegral(nu_f, nu_fp1, teff, a_rad);
    } else { // ifr == N-1
      integral_f = 1./(4*M_PI) * a_rad*SQR(SQR(teff)); // from 0 to inf
      integral_f = integral_f - 1./(4*M_PI) * BBIntegral(0, nu_f, teff, a_rad);
    }
    integral_f = fmax(0, integral_f); // avoid negative number in machine precision
    Real frac_f = integral_f/ir_cm_star_e;
    ir_cm_f += integral_f;
    if (update_matrix_row) matrix_imap(ifr,nfreq1) = frac_f;

  } else if ((ifr == nfreq1) && (n0_cm > 1)) { // (f = N-1) && (n0_cm > 1)
    // (f = N-1) && (n0_cm < 1) is covered at the beginning of the if statement
    //
    // Corner Case 3: Rightmost (ifr=N-1 && n0_cm > 1)
    //                          0 <= L <= N-2
    //                           (nu_f)
    //         nu_tet[N-2]     nu_tet[N-1]
    // ------------|---------------|--------------------------------------------------------> inf
    //           (inu1h)     (inu1)   (inu3h)     (inu2)   (inu5h)
    //    |         *          |                    |                    |
    //    |                    |         *          |         *          |
    //    |                    |                    |                    |         *
    //    |   I_cm_star[N-4]   |   I_cm_star[N-3]   |   I_cm_star[N-2]   |   I_cm_star[N-1]
    // ---|--------------------|----================|====================|==================> inf
    // nu_cm[N-4]           nu_cm[N-3]           nu_cm[N-2]           nu_cm[N-1]
    //    nu0    (nu1h)      (nu1)    (nu3h)      (nu2)     (nu5h)      nu3

    // prepare left frequency and intensity
    bool left_bin_assigned = AssignFreqIntensity(idx_l, nu_tet, i0, m, k, j, i, iang,
                                                 nang, nfreq1, n0_cm, n0, n_0, a_rad, teff,
                                                 nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                 inu1h, inu3h, inu5h, inu1, inu2,
                                                 ir_cm_star_1, boundary);
    // compute left fractional contribution
    Real ifrac_l = 0;
    if (left_bin_assigned) {
      bool leftbin = true;
      Real nu_fp1 = nu2 * 1e12; // set nu_fp1 in this way so it is not invoked in 'IntensityFraction'
      ifrac_l = IntensityFraction(nu_f, nu_fp1,
                                  nu1h, nu1, nu3h, nu2, nu5h,
                                  inu1h, inu1, inu3h, inu2, inu5h,
                                  order, limiter, boundary, leftbin);
      ir_cm_f += ifrac_l;
      if (update_matrix_row) matrix_imap(ifr,idx_l) = ifrac_l/ir_cm_star_1;
    }

    // add the rest intensity contribution
    for (int f=idx_l+1; f<=nfreq1; ++f) {
      int nf = getFreqAngIndex(f, iang, nang);
      ir_cm_f += SQR(SQR(n0_cm))*i0(m,nf,k,j,i)/(n0*n_0);
      if (update_matrix_row) matrix_imap(ifr,f) = 1;
    }

  } else { // (f <= N-2)

    Real &nu_fp1 = nu_tet(ifr+1);
    // General Case:
    // Given frequency bin [nu_tet(f), nu_tet(f+1)],
    // we can always find nu_cm(L) <  nu_tet(f)   <= nu_cm(L+1)
    //                and nu_cm(R) <= nu_tet(f+1) <  nu_cm(R+1)
    //
    //                               (nu_f)                                                                (nu_fp1)
    //                              nu_tet[f]                                                             nu_tet[f+1]
    // ............................ ---|----------------------------------------------------------------------|--- ............................ --->
    //                            I_cm_star[L]    I_cm_star[L+1]    I_cm_star[...]    I_cm_star[R-1]    I_cm_star[R]
    // ... ---|----------------|--------========|================|=== .......... ===|================|========--------|----------------|--- ... --->
    //     nu_cm[L-1]       nu_cm[L]         nu_cm[L+1]       nu_cm[L+2]         nu_cm[R-1]       nu_cm[R]         nu_cm[R+1]       nu_cm[R+2]

    // add the contribution from L+1 to R-1
    for (int f=idx_l+1; f<=idx_r-1; ++f) {
      int nf = getFreqAngIndex(f, iang, nang);
      ir_cm_f += SQR(SQR(n0_cm))*i0(m,nf,k,j,i)/(n0*n_0);
      if (update_matrix_row) matrix_imap(ifr,f) = 1;
    }

    // Left:                            (nu_f)
    //                                 nu_tet[f]
    // ................................ ---|--- ......................... --->
    //             (inu1h)   (inu1) (inu3h)|  (inu2) (inu5h)
    //        |       *        |       *   |    |                |
    //        |                |           |    |       *        |
    //        | I_cm_star[L-1] |  I_cm_star[L]  | I_cm_star[L+1] |
    // ... ---|----------------|------------====|----------------|--- ... --->
    //     nu_cm[L-1]       nu_cm[L]         nu_cm[L+1]       nu_cm[L+2]
    //       nu0   (nu1h)    (nu1)   (nu3h)   (nu2)   (nu5h)    nu3

    // prepare left frequency and intensity
    bool left_bin_assigned = AssignFreqIntensity(idx_l, nu_tet, i0, m, k, j, i, iang,
                                                 nang, nfreq1, n0_cm, n0, n_0, a_rad, teff,
                                                 nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                 inu1h, inu3h, inu5h, inu1, inu2,
                                                 ir_cm_star_1, boundary);
    // compute left fractional contribution
    Real ifrac_l = 0;
    if (left_bin_assigned) {
      bool leftbin = true;
      ifrac_l = IntensityFraction(nu_f, nu_fp1,
                                  nu1h, nu1, nu3h, nu2, nu5h,
                                  inu1h, inu1, inu3h, inu2, inu5h,
                                  order, limiter, boundary, leftbin);
      ir_cm_f += ifrac_l;
      if (update_matrix_row) matrix_imap(ifr,idx_l) = ifrac_l/ir_cm_star_1;
    }

    // Right:                          (nu_fp1)
    //                                nu_tet[f+1]
    // ................................ ---|--- ......................... --->
    //             (inu1h)   (inu1) (inu3h)|  (inu2) (inu5h)
    //        |       *        |       *   |    |                |
    //        |                |           |    |       *        |
    //        | I_cm_star[R-1] |  I_cm_star[R]  | I_cm_star[R+1] |
    // ... ---|----------------|===========-----|----------------|--- ... --->
    //     nu_cm[R-1]       nu_cm[R]         nu_cm[R+1]       nu_cm[R+2]
    //             (nu1h)    (nu1)   (nu3h)   (nu2)   (nu5h)

    // prepare right frequency and intensity
    bool right_bin_assigned = AssignFreqIntensity(idx_r, nu_tet, i0, m, k, j, i, iang,
                                                  nang, nfreq1, n0_cm, n0, n_0, a_rad, teff,
                                                  nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                  inu1h, inu3h, inu5h, inu1, inu2,
                                                  ir_cm_star_1, boundary);
    // compute right fractional contribution
    Real ifrac_r = 0;
    if ((right_bin_assigned) && (idx_r > idx_l)) {
      // note: if idx_r=idx_l, right fraction is equivalent to left bin, which would double count the contribution.
      bool leftbin = false;
      ifrac_r = IntensityFraction(nu_f, nu_fp1,
                                  nu1h, nu1, nu3h, nu2, nu5h,
                                  inu1h, inu1, inu3h, inu2, inu5h,
                                  order, limiter, boundary, leftbin);
      ir_cm_f += ifrac_r;
      if (update_matrix_row) matrix_imap(ifr,idx_r) = ifrac_r/ir_cm_star_1;
    }

    // Corner Case 4 (continue): Rightmost (nu_tet[f+1] >= nu_cm[N-1] >= nu_tet[f] && n0_cm < 1)
    //    (nu_f)                 (nu_fp1)
    //  nu_tet[N-2]           nu_tet[N-1]
    // -----|------------------------|----> inf
    //      |                        |
    //      |       | I_cm_star[N-1] |
    // -------------|================-----> inf
    //          nu_cm[N-1]
    //            (nu1)
    // R = N-1 (right_bin_assigned==false):
    if (idx_r == nfreq1) {
      nu1 = n0_cm*nu_tet(idx_r);
      int nr = getFreqAngIndex(idx_r, iang, nang);
      Real ir_cm_star_r = SQR(SQR(n0_cm))*i0(m,nr,k,j,i)/(n0*n_0);
      Real integral_r = 1./(4*M_PI) * BBIntegral(nu1, nu_fp1, teff, a_rad);
      ir_cm_f += integral_r;
      if (update_matrix_row) matrix_imap(ifr,idx_r) = integral_r/ir_cm_star_r;
    }

    // Corner Case 1: Leftmost (ifr=0 && n0_cm > 1)
    // (nu_f)    (nu_fp1)
    //    0      nu_tet[1] nu_tet[2] nu_tet[3] nu_tet[4] nu_tet[5]
    //    |---------|---------|---------|---------|---------|------> inf
    //  (inu1)      (inu3h)       (inu2)      (inu5h)
    //    |                         |            *            |
    //    |            *            |                         |
    //    |       I_cm_star[0]      |       I_cm_star[1]      |
    //    |=========----------------|-------------------------|----> inf
    // nu_cm[0]                  nu_cm[1]                  nu_cm[2]
    //  (nu1)        (nu3h)       (nu2)        (nu5h)        nu3
    // (L,L+1)=(-1,0) && (R,R+1)=(0,1)
    // L = -1: left_bin_assigned=false
    // R = 0: Do nothing. Algorithm is self-consistent.

    // Corner Case 2: Leftmost (ifr=0 && n0_cm < 1)
    // (nu_f)                                    (nu_fp1)
    //    0                                      nu_tet[1]
    //    |-----------------------------------------|------------------------------> inf
    //                          (inu1h)   (inu1) (inu3h)   (inu2) (inu5h)
    //    |                |       *        |       *        |       *        |
    //    |       *        | ir_cm_star_rm1 |  ir_cm_star_r  | ir_cm_star_rp1 |
    //    |  I_cm_star[0]  |  I_cm_star[1]  |  I_cm_star[2]  |  I_cm_star[3]  |
    //    |================|================|=======---------|----------------|----> inf
    // nu_cm[0]         nu_cm[1]         nu_cm[2]         nu_cm[3]         nu_cm[4]
    //                    nu0   (nu1h)    (nu1)   (nu3h)   (nu2)   (nu5h)    nu3
    // (L,L+1)=(-1,0) && R>=1 && R+1>=2
    // L = -1: left_bin_assigned=false
    // R <= N-1: Do nothing. Algorithm is self-consistent.


  } // endelse

  return ir_cm_f;
}

//----------------------------------------------------------------------------------------
//! \fn Real InvMapIntensity
//  \brief Map the radiation intensity from tetrad-frame frequency bins to
//         fluid-frame frequency bins.
KOKKOS_INLINE_FUNCTION
Real InvMapIntensity(const int &ifr, const DvceArray1D<Real> &nu_tet, const ScrArray2D<Real> &ir_cm_update,
                     const int &iang, const Real &n0_cm, const Real &a_rad, int order, int limiter) {
  // target frequency and intensity
  Real ir_cm_star_f = 0.0; // value to be assigned
  Real nu_cm_f = n0_cm*nu_tet(ifr);

  // parameters
  int nang = ir_cm_update.extent_int(0);
  int nfrq = nu_tet.extent_int(0);
  int nfr_ang = nfrq*nang;
  int nfreq1 = nfrq-1;
  Real &nu_e = nu_tet(nfreq1);
  Real nu_cm_e = n0_cm*nu_e;

  // auxiliary variables for intensity mapping
  Real nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h;
  Real inu1, inu2, inu1h, inu3h, inu5h;
  Real ir_cm_1; int boundary;

  // get effective temperature at last frequency bin
  Real ir_cm_e = ir_cm_update(iang, nfreq1);
  Real teff = GetEffTemperature(ir_cm_e, nu_e, a_rad);

  // locate left & right tetrad-frame frequency bins (if exist)
  // -1 <= idx_l   <= N-2
  //  0 <= idx_l+1 <= N-1
  int idx_lp1=0;
  while ((nu_tet(idx_lp1) < nu_cm_f) && (idx_lp1 <= nfreq1)) idx_lp1++;
  int idx_l = idx_lp1-1;
  // -1 <= idx_r   <= N-2
  //  0 <= idx_r+1 <= N-1
  int idx_r=-1, idx_rp1=-1;
  if (ifr+1 <= nfreq1) {
    idx_rp1=fmax(idx_l,0);
    Real nu_cm_fp1 = n0_cm*nu_tet(ifr+1);
    while ((nu_tet(idx_rp1) <= nu_cm_fp1) && (idx_rp1 <= nfreq1)) idx_rp1++;
    idx_r = idx_rp1-1;
  }

  // get mapping coefficients
  if (nu_cm_f >= nu_e) { // only happen when (n0_cm > 1)
    // This covers the rightmost corner case (f = N-1) && (n0_cm > 1)
    // Corner Case 4: Rightmost (nu_cm[f] >= nu_tet[N-1] && n0_cm > 1)
    Real integral_f = 0.0;
    if (ifr+1 <= nfreq1) { // ifr <= N-2
      Real nu_cm_fp1 = n0_cm*nu_tet(ifr+1);
      integral_f = 1./(4*M_PI) * BBIntegral(nu_cm_f, nu_cm_fp1, teff, a_rad);
    } else { // ifr == N-1
      integral_f = 1./(4*M_PI) * a_rad*SQR(SQR(teff)); // from 0 to inf
      integral_f = integral_f - 1./(4*M_PI) * BBIntegral(0, nu_cm_f, teff, a_rad);
    }
    integral_f = fmax(0, integral_f); // avoid negative number in machine precision
    ir_cm_star_f += integral_f;

  } else if ((ifr == nfreq1) && (n0_cm < 1)) { // (f = N-1) && (n0_cm < 1)
    // (f = N-1) && (n0_cm > 1) is covered at the beginning of the if statement
    // prepare left frequency and intensity
    bool left_bin_assigned = AssignFreqIntensityInv(idx_l, nu_tet, ir_cm_update, iang,
                                                    nfreq1, a_rad, teff,
                                                    nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                    inu1h, inu3h, inu5h, inu1, inu2,
                                                    ir_cm_1, boundary);
    // compute left fractional contribution
    Real ifrac_l = 0;
    if (left_bin_assigned) {
      bool leftbin = true;
      Real nu_cm_fp1 = nu2 * 1e12; // set nu_cm_fp1 in this way so it is not invoked in 'IntensityFraction'
      ifrac_l = IntensityFraction(nu_cm_f, nu_cm_fp1,
                                  nu1h, nu1, nu3h, nu2, nu5h,
                                  inu1h, inu1, inu3h, inu2, inu5h,
                                  order, limiter, boundary, leftbin);
      ir_cm_star_f += ifrac_l;
    }

    // add the rest intensity contribution
    for (int f=idx_l+1; f<=nfreq1; ++f) {
      ir_cm_star_f += ir_cm_update(iang, f);
    }

  } else { // (f <= N-2)
    Real nu_cm_fp1 = n0_cm*nu_tet(ifr+1);

    // add the contribution from L+1 to R-1
    for (int f=idx_l+1; f<=idx_r-1; ++f) {
      ir_cm_star_f += ir_cm_update(iang, f);
    }

    // prepare left frequency and intensity
    bool left_bin_assigned = AssignFreqIntensityInv(idx_l, nu_tet, ir_cm_update, iang,
                                                    nfreq1, a_rad, teff,
                                                    nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                    inu1h, inu3h, inu5h, inu1, inu2,
                                                    ir_cm_1, boundary);
    // compute left fractional contribution
    Real ifrac_l = 0;
    if (left_bin_assigned) {
      bool leftbin = true;
      ifrac_l = IntensityFraction(nu_cm_f, nu_cm_fp1,
                                  nu1h, nu1, nu3h, nu2, nu5h,
                                  inu1h, inu1, inu3h, inu2, inu5h,
                                  order, limiter, boundary, leftbin);
      ir_cm_star_f += ifrac_l;
    }

    // prepare right frequency and intensity
    bool right_bin_assigned = AssignFreqIntensityInv(idx_r, nu_tet, ir_cm_update, iang,
                                                     nfreq1, a_rad, teff,
                                                     nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                     inu1h, inu3h, inu5h, inu1, inu2,
                                                     ir_cm_1, boundary);
    // compute right fractional contribution
    Real ifrac_r = 0;
    if ((right_bin_assigned) && (idx_r > idx_l)) {
      bool leftbin = false;
      ifrac_r = IntensityFraction(nu_cm_f, nu_cm_fp1,
                                  nu1h, nu1, nu3h, nu2, nu5h,
                                  inu1h, inu1, inu3h, inu2, inu5h,
                                  order, limiter, boundary, leftbin);
      ir_cm_star_f += ifrac_r;
    }

    // Corner Case 4 (continue): Rightmost (nu_cm[f+1] >= nu_tet[N-1] >= nu_cm[f] && n0_cm > 1)
    // R = N-1 (right_bin_assigned==false):
    if (idx_r == nfreq1) {
      nu1 = nu_tet(idx_r);
      Real ir_cm_r = ir_cm_update(iang, idx_r);
      Real integral_r = 1./(4*M_PI) * BBIntegral(nu1, nu_cm_fp1, teff, a_rad);
      ir_cm_star_f += integral_r;
    }
  } // endelse

  return ir_cm_star_f;
}

//============================= Compton Helper Functions =============================
//----------------------------------------------------------------------------------------
//! \fn void WienInt
//  \brief Integrate Wien's tail from nu_e to inf given jr_cm_e for normalization
KOKKOS_INLINE_FUNCTION
void WienInt(const int &nu_e, const Real &jr_cm_e, const Real &tgas, const Real &arad,
             Real &int_n_nu2_e, Real &int_n_nu4_e, Real &int_n2_nu4_e) {
  Real tgas2 = SQR(tgas);
  Real tgas3 = tgas*tgas2;
  Real tgas4 = tgas*tgas3;
  Real nu_e2 = SQR(nu_e);
  Real nu_e3 = nu_e*nu_e2;
  Real nu_e4 = nu_e*nu_e3;

  // int_{nu_e}^{inf} exp(-nu/tgas) nu^3 dnu / (tgas * exp(-nu_e/tgas))
  Real int_wien_nu3 = 6*tgas3 + 6*tgas2*nu_e + 3*tgas*nu_e2 + nu_e3;

  // int_{nu_e}^{inf} exp(-nu/tgas) nu^2 dnu
  int_n_nu2_e = 2*tgas2 + 2*tgas*nu_e + nu_e2;
  int_n_nu2_e *= jr_cm_e/arad/int_wien_nu3;

  // int_{nu_e}^{inf} exp(-nu/tgas) nu^4 dnu
  int_n_nu4_e = 24*tgas4 + 24*tgas3*nu_e + 12*tgas2*nu_e2 + 4*tgas*nu_e3 + nu_e4;
  int_n_nu4_e *= jr_cm_e/arad/int_wien_nu3;

  // int_{nu_e}^{inf} exp(-nu/tgas)^2 nu^4 dnu
  int_n2_nu4_e = 3*tgas4 + 6*tgas3*nu_e + 6*tgas2*nu_e2 + 4*tgas*nu_e3 + 2*nu_e4;
  int_n2_nu4_e *= 0.25/tgas * SQR(jr_cm_e/arad/int_wien_nu3);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FuncCompTempEst
//  \brief Target function to solve integrated Kompaneets equation for gas temperature estimation
KOKKOS_INLINE_FUNCTION
void FuncCompTempEst(const Real &rat, const Real &c5, const Real &c4, const Real &c0, Real &f, Real &df) {
  Real rat3 = rat*SQR(rat);
  Real rat4 = rat*rat3;

  f = c5*rat*rat4 + c4*rat4 + c0;
  df = 5*c5*rat4 + 4*c4*rat3;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SolCompTempEst
//  \brief Solve integrated Kompaneets equation for gas temperature estimation
KOKKOS_INLINE_FUNCTION
int SolCompTempEst(const Real &c5, const Real &c4, const Real &c0, const Real &x0, Real &x_sol) {
  int num_itr_max = 25;
  Real tol = 1e-12;
  Real f_max = 1e12, f_min = 1e-12; // these are the limits for the temperature changing factor
  Real fac_scan = 2;

  Real f, df;
  FuncCompTempEst(x0, c5, c4, c0, f, df);

  // look for left and right boundaries
  Real xl, xr;
  Real x = x0;
  if (f < 0) {
    xl = x0;
    while ((f < 0) && (f <= f_max)) {
      x *= fac_scan;
      FuncCompTempEst(x, c5, c4, c0, f, df);
    } // endwhile
    if (f < 0) return -1;
    xr = x;
  } else { // f >= 0
    xr = x0;
    while ((f > 0) && (f >= f_min)) {
      x /= fac_scan;
      FuncCompTempEst(x, c5, c4, c0, f, df);
    } // endwhile
    if (f > 0) xl = 0; // f(0)=c0 is guaranteed to be negative
    else xl = x;
  } // endelse

  // start Newton iterations at midpoint
  x = 0.5 * (xl + xr);
  // x = x0;
  FuncCompTempEst(x, c5, c4, c0, f, df);

  // find solution
  for (int i = 0; i < num_itr_max; ++i) {
    Real xnext;
    Real len = xr - xl;
    Real xmid   = 0.5 * (xl + xr);
    Real len_bi = 0.5 * len;

    // try Newton method
    bool use_newton = (fabs(df) > 0);
    if (use_newton) {
        xnext = x - f/df;
        if (!(xnext > xl && xnext < xr) || !isfinite(xnext))
          use_newton = false;
    } // endif

    // check if Newton is better than bisection
    Real fnext, dfnext;
    if (use_newton) {
        FuncCompTempEst(xnext, c5, c4, c0, fnext, dfnext);

        // compare bracket reduction against bisection
        Real len_nw = (fnext > 0.0) ? (xnext - xl) : (xr - xnext);

        // use 0.9 to require Newton to beat bisection clearly
        if (len_nw > 0.9*len_bi) use_newton = false;
    } // endif

    // fall back to bisection
    if (!use_newton) {
        xnext = xmid;
        FuncCompTempEst(xnext, c5, c4, c0, fnext, dfnext);
    }

    // update bracket
    if (fnext <= 0.0) xl = xnext;
    else xr = xnext;

    // convergence check
    x_sol = xnext;
    if (fabs(xr - xl) <= tol * (1.0 + fabs(x_sol)) || fabs(fnext) <= tol) {
        return 0; // success
    } // endif

    x  = xnext;
    f  = fnext;
    df = dfnext;
  } // endfor i

  return 1; // reach maximum iteration
}

//----------------------------------------------------------------------------------------
//! \fn void QuasiSteadySol
//  \brief Solve lambda assuming quasi-steady distribution of photon occupation number,
//         i.e., 2Li_3(1/lambda) = N/tgas^3
KOKKOS_INLINE_FUNCTION
Real QuasiSteadySol(const Real &n_tot, const Real &tgas) {
  // TODO: develop a more accurate approximation for this
  // use Yanfei's approximation temporarily
  Real lambda = 1.0;
  Real n_t3 = n_tot/(tgas*SQR(tgas));
  if ((n_t3 < 2.3739) && (n_t3 > 0.6932)) {
    lambda = 1.948 * pow(n_t3, -1.016) + 0.1907;
  } else if (n_t3 <= 0.6932) {
    lambda = (1.0 + sqrt(1.0 + 0.25 * n_t3))/n_t3;
  }

  return lambda;
}

//----------------------------------------------------------------------------------------
//! \fn void SolveTridiag
//  \brief Solve a tridiagonal system Ax = komp_coeff
//         komp_mat_d(1...nfrq-1): subdiagonal c1_f (komp_mat_d(0) unused)
//         komp_mat_c(0...nfrq-1): diagonal    c2_f
//         komp_mat_u(0...nfrq-2): superdiag   c3_f (komp_mat_u(nfrq-1) unused)
//         komp_coeff(0...nfrq-1): RHS (i.e., -c4_f)
//         Return x=ret(0...nfrq-1)
KOKKOS_INLINE_FUNCTION
bool SolveTridiag(const ScrArray1D<Real> &komp_mat_d, const ScrArray1D<Real> &komp_mat_c,
                  const ScrArray1D<Real> &komp_mat_u, const ScrArray1D<Real> &komp_coeff,
                  const int &nfrq, ScrArray1D<Real> &u_tmp, ScrArray1D<Real> &rhs_tmp,
                  ScrArray1D<Real> &ret) {

  if (nfrq <= 0) return false; // invalid tridiagonal system
  const Real eps = 1e-14;

  for (int ifr=0; ifr<nfrq; ++ifr) {
    u_tmp(ifr)   = 0.0;
    rhs_tmp(ifr) = 0.0;
  }

  // rescaling factor
  Real fac_scale = komp_mat_c(0);
  for (int ifr=0; ifr<nfrq; ++ifr) {
    fac_scale = fmin(fac_scale, fabs(komp_mat_c(ifr)));
  }
  fac_scale = 1./fac_scale;
  if (!(isfinite(fac_scale))) fac_scale = 1.;

  // ifr=0
  Real denom = fac_scale*komp_mat_c(0);
  if (std::fabs(denom) < eps) return false;

  u_tmp(0)   = (nfrq > 1) ? (fac_scale*komp_mat_u(0)/denom) : Real(0);
  rhs_tmp(0) = fac_scale*komp_coeff(0)/denom;

  // forward sweep
  for (int ifr=1; ifr<nfrq; ++ifr) {
      denom = fac_scale*komp_mat_c(ifr) - fac_scale*komp_mat_d(ifr) * u_tmp(ifr - 1);
      if (std::fabs(denom) < eps) return false;
      u_tmp(ifr)   = (ifr < nfrq-1) ? (fac_scale*komp_mat_u(ifr)/denom) : 0;
      rhs_tmp(ifr) = (fac_scale*komp_coeff(ifr) - fac_scale*komp_mat_d(ifr)*rhs_tmp(ifr-1)) / denom;
  }

  // back substitution
  ret(nfrq-1) = rhs_tmp(nfrq-1);
  for (int ifr=nfrq-2; ifr>=0; --ifr) {
      ret(ifr) = rhs_tmp(ifr) - u_tmp(ifr) * ret(ifr+1);
  }

  return true;
}






#endif // RADIATION_RADIATION_MULTI_FREQ_HPP_
