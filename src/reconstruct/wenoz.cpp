//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ppm.cpp
//  \brief WENO-Z reconstruction for a Cartesian-like coordinate with uniform spacing.
//
// REFERENCES:
// Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially non-oscillatory scheme 
// for hyperbolic conservation laws" , JCP, 227, 3191 (2008)

#include <algorithm>    // max()
#include <math.h>

#include "athena.hpp"


template<bool FD=false>
KOKKOS_INLINE_FUNCTION
Real WENOZ(Real w0, Real w1, Real w2, Real w3, Real w4) noexcept {
	
	using T = Real;

        int constexpr minus2 = 0;
        int constexpr minus1 = 1;
        int constexpr plus0 = 2;
        int constexpr plus1 = 3;
        int constexpr plus2 = 4;

	T WENOVAR[5] {};
	WENOVAR[0] = w0;
	WENOVAR[1] = w1;
	WENOVAR[2] = w2;
	WENOVAR[3] = w3;
	WENOVAR[4] = w4;


        const T dFD[3]{1. / 16., 10. / 16., 5. / 16.};

        const T dFV[3]{1. / 10., 3. / 5., 3. / 10.};

        T f[3];
        if (FD) {
            // Finite difference reconstruction

           f[0] = 3. / 8. * WENOVAR[minus2] - 10. / 8. * WENOVAR[minus1] +
                   15. / 8. * WENOVAR[plus0];
            f[1] = -1. / 8. * WENOVAR[minus1] + 6. / 8. * WENOVAR[plus0] +
                   3. / 8. * WENOVAR[plus1];
            f[2] = 3. / 8. * WENOVAR[plus0] + 6. / 8. * WENOVAR[plus1] -
                   1. / 8. * WENOVAR[plus2];
        } else {
            f[0] = 1. / 3. * WENOVAR[minus2] - 7. / 6. * WENOVAR[minus1] +
                   11. / 6. * WENOVAR[plus0];
            f[1] = -1. / 6. * WENOVAR[minus1] + 5. / 6. * WENOVAR[plus0] +
                   1. / 3. * WENOVAR[plus1];
            f[2] = 1. / 3. * WENOVAR[plus0] + 5. / 6. * WENOVAR[plus1] -
                   1. / 6. * WENOVAR[plus2];
        }

        // Smooth WENO weights: Note that these are from Del Zanna et al. 2007
        // (A.18)

        T beta[3];

        const T beta_coeff[2]{13. / 12., 0.25};

        beta[0] = beta_coeff[0] *
                  SQR(WENOVAR[minus2] + WENOVAR[plus0] - 2.0 * WENOVAR[minus1]) +
                  beta_coeff[1] * SQR(WENOVAR[minus2] - 4. * WENOVAR[minus1] +
                                     3. * WENOVAR[plus0]);

        beta[1] = beta_coeff[0] *
                  SQR(WENOVAR[minus1] + WENOVAR[plus1] - 2.0 * WENOVAR[plus0]) +
                  beta_coeff[1] * SQR(WENOVAR[minus1] - WENOVAR[plus1]);

        beta[2] = beta_coeff[0] *
                  SQR(WENOVAR[plus0] + WENOVAR[plus2] - 2.0 * WENOVAR[plus1]) +
                  beta_coeff[1] * SQR(3. * WENOVAR[plus0] - 4. * WENOVAR[plus1] +
                                     WENOVAR[plus2]);

        // Rescale epsilon
        //    constexpr double epsL = 1.e-42;
        const T epsL = 1.e-42;

        // WENO-Z+: Acker et al. 2016

        const T tau_5 = fabs(beta[0] - beta[2]);

        const T indicator[3]{(tau_5) / (beta[0] + epsL), (tau_5) / (beta[1] + epsL),
                             (tau_5) / (beta[2] + epsL)};

        T alpha[3]{1. + SQR(indicator[0]), 1. + SQR(indicator[1]),
                   1. + SQR(indicator[2])};

        T alpha_sum = 0.;
        if (FD) {
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                alpha[i] *= dFD[i];
                alpha_sum += alpha[i];
            };
        } else { // FV

#pragma unroll
            for (int i = 0; i < 3; ++i) {
                alpha[i] *= dFV[i];
                alpha_sum += alpha[i];
            };
        }

        T flux = 0.;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            flux += f[i] * alpha[i]; // / alpha_sum;
        };

        return flux / alpha_sum;

    };


//----------------------------------------------------------------------------------------
//! \fn WENOZ
//  \brief Reconstructs limited fifth-order polynomial in cell i to compute ql(i+1) and qr(i) in [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//
KOKKOS_INLINE_FUNCTION
void WENOZX1(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 

      //---- set L/R states ----
      ql(n,i+1) = WENOZ(q(m,n,k,j,i-2), q(m,n,k,j,i-1), q(m,n,k,j,i-0), q(m,n,k,j,i+1), q(m,n,k,j,i+2));
      qr(n,i  ) = WENOZ(q(m,n,k,j,i+2), q(m,n,k,j,i+1), q(m,n,k,j,i-0), q(m,n,k,j,i-1), q(m,n,k,j,i-2));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZX2
//  \brief Reconstructs limited quadratic slope in cell j to compute ql(i+1) and qr(i) in [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]
//  
KOKKOS_INLINE_FUNCTION
void WENOZX2(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 

      //---- set L/R states ----
      ql_jp1(n,i) = WENOZ(q(m,n,k,j-2,i), q(m,n,k,j-1,i), q(m,n,k,j-0,i), q(m,n,k,j+1,i), q(m,n,k,j+2,i));
      qr_j(n,i  ) = WENOZ(q(m,n,k,j+2,i), q(m,n,k,j+1,i), q(m,n,k,j-0,i), q(m,n,k,j-1,i), q(m,n,k,j-2,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZX3
//  \brief Reconstructs limited quadratic slope in cell k to compute ql(i+1) and qr(i) in [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//  
//  Note: This version works only makes sense with the conservative
//  finite difference flux corrector.
KOKKOS_INLINE_FUNCTION
void WENOZX3(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 

      //---- set L/R states ----
      ql_kp1(n,i) = WENOZ(q(m,n,k-2,j,i), q(m,n,k-1,j,i), q(m,n,k-0,j,i), q(m,n,k+1,j,i), q(m,n,k+2,j,i));
      qr_k(n,i  ) = WENOZ(q(m,n,k+2,j,i), q(m,n,k+1,j,i), q(m,n,k-0,j,i), q(m,n,k-1,j,i), q(m,n,k-2,j,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZ_FDX1
//  \brief Reconstructs limited quadratic slope in cell i to compute ql(i+1) and qr(i) in [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//  
//  Note: This version works only makes sense with the conservative
//  finite difference flux corrector.
KOKKOS_INLINE_FUNCTION
void WENOZ_FDX1(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 

      //---- set L/R states ----
      ql(n,i+1) = WENOZ(q(m,n,k,j,i-2), q(m,n,k,j,i-1), q(m,n,k,j,i-0), q(m,n,k,j,i+1), q(m,n,k,j,i+2));
      qr(n,i  ) = WENOZ(q(m,n,k,j,i+2), q(m,n,k,j,i+1), q(m,n,k,j,i-0), q(m,n,k,j,i-1), q(m,n,k,j,i-2));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZ_FDX2
//  \brief Reconstructs limited quadratic slope in cell j to compute ql(i+1) and qr(i) in [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]
//  
//  Note: This version works only makes sense with the conservative
//  finite difference flux corrector.
KOKKOS_INLINE_FUNCTION
void WENOZ_FDX2(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 

      //---- set L/R states ----
      ql_jp1(n,i) = WENOZ(q(m,n,k,j-2,i), q(m,n,k,j-1,i), q(m,n,k,j-0,i), q(m,n,k,j+1,i), q(m,n,k,j+2,i));
      qr_j(n,i  ) = WENOZ(q(m,n,k,j+2,i), q(m,n,k,j+1,i), q(m,n,k,j-0,i), q(m,n,k,j-1,i), q(m,n,k,j-2,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZ_FDX3
//  \brief Reconstructs limited quadratic slope in cell k to compute ql(i+1) and qr(i) in [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//  
//  Note: This version works only makes sense with the conservative
//  finite difference flux corrector.
KOKKOS_INLINE_FUNCTION
void WENOZ_FDX3(TeamMember_t const &member,const int m,const int k,const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k)
{
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    { 

      //---- set L/R states ----
      ql_kp1(n,i) = WENOZ(q(m,n,k-2,j,i), q(m,n,k-1,j,i), q(m,n,k-0,j,i), q(m,n,k+1,j,i), q(m,n,k+2,j,i));
      qr_k(n,i  ) = WENOZ(q(m,n,k+2,j,i), q(m,n,k+1,j,i), q(m,n,k-0,j,i), q(m,n,k-1,j,i), q(m,n,k-2,j,i));
    });
  }
  return;
}

