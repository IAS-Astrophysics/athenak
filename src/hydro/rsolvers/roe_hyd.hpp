#ifndef HYDRO_RSOLVERS_ROE_HYD_HPP_
#define HYDRO_RSOLVERS_ROE_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file  roe_hyd.hpp
//! \brief Roe's linearized Riemann solver.
//!
//! Computes 1D fluxes using Roe's linearization.  When Roe's method fails because of
//! negative density in the intermediate states, LLF fluxes are used instead (only
//! density, not pressure, is checked in this version).
//!
//! REFERENCES:
//! - P. Roe, "Approximate Riemann solvers, parameter vectors, and difference schemes",
//!   JCP, 43, 357 (1981).

#include <float.h>
#include <algorithm>  // max()
#include <cmath>      // sqrt()

namespace hydro {

// prototype for functions to compute Roe fluxes from eigenmatrices
namespace roe {

KOKKOS_INLINE_FUNCTION
void RoeFluxAdb(const Real wroe[], const Real du[], const Real wli[],
                       const Real gm1, Real flx[], Real eigenvalues[], int &flag);
KOKKOS_INLINE_FUNCTION
void RoeFluxIso(const Real wroe[], const Real du[], const Real wli[],
                       const Real isocs, Real flx[], Real eigenvalues[], int &flag);

} // namespace roe

//----------------------------------------------------------------------------------------
//! \fn void Roe
//! \brief The Roe Riemann solver for hydrodynamics (both ideal gas and isothermal)

KOKKOS_INLINE_FUNCTION
void Roe(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5],wroe[5];
  Real fl[5],fr[5],flxi[5];
  Real ev[5],du[5];
  Real gm1 = eos.gamma - 1.0;
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i) {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);

    // store pressure in L/R primitives
    if (eos.is_ideal) {
      wli[IEN] = eos.IdealGasPressure(wl(IEN,i));
      wri[IEN] = eos.IdealGasPressure(wr(IEN,i));
    }

    //--- Step 2.  Compute Roe-averaged data from left- and right-states

    Real sqrtdl = sqrt(wli[IDN]);
    Real sqrtdr = sqrt(wri[IDN]);
    Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

    wroe[IDN]  = sqrtdl*sqrtdr;
    wroe[IVX] = (sqrtdl*wli[IVX] + sqrtdr*wri[IVX])*isdlpdr;
    wroe[IVY] = (sqrtdl*wli[IVY] + sqrtdr*wri[IVY])*isdlpdr;
    wroe[IVZ] = (sqrtdl*wli[IVZ] + sqrtdr*wri[IVZ])*isdlpdr;

    // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for ideal gas EOS,
    // rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
    Real el,er;
    if (eos.is_ideal) {
      el = wli[IEN]/gm1 + 0.5*wli[IDN]*(SQR(wli[IVX])+SQR(wli[IVY])+SQR(wli[IVZ]));
      er = wri[IEN]/gm1 + 0.5*wri[IDN]*(SQR(wri[IVX])+SQR(wri[IVY])+SQR(wri[IVZ]));
      wroe[IEN] = ((el + wli[IEN])/sqrtdl + (er + wri[IEN])/sqrtdr)*isdlpdr;
    }

    //--- Step 3.  Compute L/R fluxes

    Real mxl = wli[IDN]*wli[IVX];
    Real mxr = wri[IDN]*wri[IVX];

    fl[IDN] = mxl;
    fr[IDN] = mxr;

    fl[IVX] = mxl*wli[IVX];
    fr[IVX] = mxr*wri[IVX];

    fl[IVY] = mxl*wli[IVY];
    fr[IVY] = mxr*wri[IVY];

    fl[IVZ] = mxl*wli[IVZ];
    fr[IVZ] = mxr*wri[IVZ];

    if (eos.is_ideal) {
      fl[IVX] += wli[IEN];
      fr[IVX] += wri[IEN];
      fl[IEN] = (el + wli[IEN])*wli[IVX];
      fr[IEN] = (er + wri[IEN])*wri[IVX];
    } else {
      fl[IVX] += (iso_cs*iso_cs)*wli[IDN];
      fr[IVX] += (iso_cs*iso_cs)*wri[IDN];
    }

    //--- Step 4.  Compute Roe fluxes.

    du[IDN] = wri[IDN]          - wli[IDN];
    du[IVX] = wri[IDN]*wri[IVX] - wli[IDN]*wli[IVX];
    du[IVY] = wri[IDN]*wri[IVY] - wli[IDN]*wli[IVY];
    du[IVZ] = wri[IDN]*wri[IVZ] - wli[IDN]*wli[IVZ];
    if (eos.is_ideal) du[IEN] = er - el;

    flxi[IDN] = 0.5*(fl[IDN] + fr[IDN]);
    flxi[IVX] = 0.5*(fl[IVX] + fr[IVX]);
    flxi[IVY] = 0.5*(fl[IVY] + fr[IVY]);
    flxi[IVZ] = 0.5*(fl[IVZ] + fr[IVZ]);
    if (eos.is_ideal) flxi[IEN] = 0.5*(fl[IEN] + fr[IEN]);

    int llf_flag = 0;
    if (eos.is_ideal) {
      roe::RoeFluxAdb(wroe,du,wli,gm1,flxi,ev,llf_flag);
    } else {
      roe::RoeFluxIso(wroe,du,wli,iso_cs,flxi,ev,llf_flag);
    }

    //--- Step 5.  Overwrite with upwind flux if flow is supersonic

    if (ev[0] >= 0.0) {
      flxi[IDN] = fl[IDN];
      flxi[IVX] = fl[IVX];
      flxi[IVY] = fl[IVY];
      flxi[IVZ] = fl[IVZ];
      if (eos.is_ideal) flxi[IEN] = fl[IEN];
    }
    if (eos.is_ideal) {
      if (ev[4] <= 0.0) {
        flxi[IDN] = fr[IDN];
        flxi[IVX] = fr[IVX];
        flxi[IVY] = fr[IVY];
        flxi[IVZ] = fr[IVZ];
        flxi[IEN] = fr[IEN];
      }
    } else {
      if (ev[3] <= 0.0) {
        flxi[IDN] = fr[IDN];
        flxi[IVX] = fr[IVX];
        flxi[IVY] = fr[IVY];
        flxi[IVZ] = fr[IVZ];
      }
    }

    //--- Step 6.  Overwrite with LLF flux if any of intermediate states are negative

    if (llf_flag != 0) {
      Real cl,cr;
      if (eos.is_ideal) {
        cl = eos.IdealHydroSoundSpeed(wli[IDN], wli[IEN]);
        cr = eos.IdealHydroSoundSpeed(wri[IDN], wri[IEN]);
      } else {
        cl = iso_cs;
        cr = iso_cs;
      }
      Real a  = 0.5*fmax( (fabs(wli[IVX]) + cl), (fabs(wri[IVX]) + cr) );

      flxi[IDN] = 0.5*(fl[IDN] + fr[IDN]) - a*du[IDN];
      flxi[IVX] = 0.5*(fl[IVX] + fr[IVX]) - a*du[IVX];
      flxi[IVY] = 0.5*(fl[IVY] + fr[IVY]) - a*du[IVY];
      flxi[IVZ] = 0.5*(fl[IVZ] + fr[IVZ]) - a*du[IVZ];
      if (eos.is_ideal) {flxi[IEN] = 0.5*(fl[IEN] + fr[IEN]) - a*du[IEN];}
    }

    //--- Step 7. Store results into 3D array of fluxes

    flx(m,IDN,k,j,i) = flxi[IDN];
    flx(m,ivx,k,j,i) = flxi[IVX];
    flx(m,ivy,k,j,i) = flxi[IVY];
    flx(m,ivz,k,j,i) = flxi[IVZ];
    if (eos.is_ideal) flx(m,IEN,k,j,i) = flxi[IEN];
  });
  return;
}

namespace roe {
//----------------------------------------------------------------------------------------
//! \fn RoeFlux()
//  \brief Computes Roe fluxes for the conserved variables, that is
//            F[n] = 0.5*(F_l + F_r) - SUM_m(coeff[m]*rem[n][m])
//  where     coeff[n] = 0.5*ev[n]*SUM_m(dU[m]*lem[n][m])
//  and the rem[n][m] and lem[n][m] are matrices of the L- and R-eigenvectors of Roe's
//  matrix "A". Also returns the eigenvalues through the argument list.
//
// INPUT:
//   wroe: vector of Roe averaged primitive variables
//   du: Ur - Ul, difference in L/R-states in conserved variables
//   wli: Wl, left state in primitive variables
//   flx: (F_l + F_r)/2
//
// OUTPUT:
//   flx: final Roe flux
//   ev: vector of eingenvalues
//   llf_flag: flag set to 1 if d<0 in any intermediate state
//
//  The order of the components in the input vectors should be:
//     (IDN,IVX,IVY,IVZ,[IEN])
//
// REFERENCES:
// - J. Stone, T. Gardiner, P. Teuben, J. Hawley, & J. Simon "Athena: A new code for
//   astrophysical MHD", ApJS, (2008), Appendix A.  Equation numbers refer to this paper.

KOKKOS_INLINE_FUNCTION
void RoeFluxAdb(const Real wroe[], const Real du[], const Real wli[], const Real gm1,
     Real flx[], Real ev[], int &llf_flag) {
  Real v1 = wroe[IVX];
  Real v2 = wroe[IVY];
  Real v3 = wroe[IVZ];

    Real h = wroe[IEN];
    Real vsq = v1*v1 + v2*v2 + v3*v3;
    Real q = h - 0.5*vsq;
    Real cs_sq = (q < 0.0) ? (FLT_MIN) : gm1*q;
    Real cs = sqrt(cs_sq);

    // Compute eigenvalues (eq. B2)
    ev[0] = v1 - cs;
    ev[1] = v1;
    ev[2] = v1;
    ev[3] = v1;
    ev[4] = v1 + cs;

    // Compute projection of dU onto L-eigenvectors using matrix elements from eq. B4
    Real a[5];
    Real na = 0.5/cs_sq;
    a[0]  = du[0]*(0.5*gm1*vsq + v1*cs);
    a[0] -= du[1]*(gm1*v1 + cs);
    a[0] -= du[2]*gm1*v2;
    a[0] -= du[3]*gm1*v3;
    a[0] += du[4]*gm1;
    a[0] *= na;

    a[1]  = du[0]*(-v2);
    a[1] += du[2];

    a[2]  = du[0]*(-v3);
    a[2] += du[3];

    Real qa = gm1/cs_sq;
    a[3]  = du[0]*(1.0 - na*gm1*vsq);
    a[3] += du[1]*qa*v1;
    a[3] += du[2]*qa*v2;
    a[3] += du[3]*qa*v3;
    a[3] -= du[4]*qa;

    a[4]  = du[0]*(0.5*gm1*vsq - v1*cs);
    a[4] -= du[1]*(gm1*v1 - cs);
    a[4] -= du[2]*gm1*v2;
    a[4] -= du[3]*gm1*v3;
    a[4] += du[4]*gm1;
    a[4] *= na;

    Real coeff[5];
    coeff[0] = -0.5*fabs(ev[0])*a[0];
    coeff[1] = -0.5*fabs(ev[1])*a[1];
    coeff[2] = -0.5*fabs(ev[2])*a[2];
    coeff[3] = -0.5*fabs(ev[3])*a[3];
    coeff[4] = -0.5*fabs(ev[4])*a[4];

    // compute density in intermediate states and check that it is positive, set flag
    // This requires computing the [0][*] components of the right-eigenmatrix
    Real dens = wli[IDN] + a[0];  // rem[0][0]=1, so don't bother to compute or store
    if (dens < 0.0) llf_flag=1;

    dens += a[3];  // rem[0][3]=1, so don't bother to compute or store
    if (dens < 0.0) llf_flag=1;

    // Now multiply projection with R-eigenvectors from eq. B3 and SUM into output fluxes
    flx[0] += coeff[0];
    flx[0] += coeff[3];
    flx[0] += coeff[4];

    flx[1] += coeff[0]*(v1 - cs);
    flx[1] += coeff[3]*v1;
    flx[1] += coeff[4]*(v1 + cs);

    flx[2] += coeff[0]*v2;
    flx[2] += coeff[1];
    flx[2] += coeff[3]*v2;
    flx[2] += coeff[4]*v2;

    flx[3] += coeff[0]*v3;
    flx[3] += coeff[2];
    flx[3] += coeff[3]*v3;
    flx[3] += coeff[4]*v3;

    flx[4] += coeff[0]*(h - v1*cs);
    flx[4] += coeff[1]*v2;
    flx[4] += coeff[2]*v3;
    flx[4] += coeff[3]*0.5*vsq;
    flx[4] += coeff[4]*(h + v1*cs);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn RoeFlux()

KOKKOS_INLINE_FUNCTION
void RoeFluxIso(const Real wroe[], const Real du[], const Real wli[], const Real iso_cs,
     Real flx[], Real ev[], int &llf_flag) {
  Real v1 = wroe[IVX];
  Real v2 = wroe[IVY];
  Real v3 = wroe[IVZ];

    //--- Isothermal hydrodynamics

    // Compute eigenvalues (eq. B6)
    ev[0] = v1 - iso_cs;
    ev[1] = v1;
    ev[2] = v1;
    ev[3] = v1 + iso_cs;

    // Compute projection of dU onto L-eigenvectors using matrix elements from eq. B7
    Real a[4];
    a[0]  = du[0]*(0.5 + 0.5*v1/iso_cs);
    a[0] -= du[1]*0.5/iso_cs;

    a[1]  = du[0]*(-v2);
    a[1] += du[2];

    a[2]  = du[0]*(-v3);
    a[2] += du[3];

    a[3]  = du[0]*(0.5 - 0.5*v1/iso_cs);
    a[3] += du[1]*0.5/iso_cs;

    Real coeff[4];
    coeff[0] = -0.5*fabs(ev[0])*a[0];
    coeff[1] = -0.5*fabs(ev[1])*a[1];
    coeff[2] = -0.5*fabs(ev[2])*a[2];
    coeff[3] = -0.5*fabs(ev[3])*a[3];

    // compute density in intermediate states and check that it is positive, set flag
    // This requires computing the [0][*] components of the right-eigenmatrix
    Real dens = wli[IDN] + a[0];  // rem[0][0]=1, so don't bother to compute or store
    if (dens < 0.0) llf_flag=1;

    dens += a[3];  // rem[0][3]=1, so don't bother to compute or store
    if (dens < 0.0) llf_flag=1;

    // Now multiply projection with R-eigenvectors from eq. B3 and SUM into output fluxes
    flx[0] += coeff[0];
    flx[0] += coeff[3];

    flx[1] += coeff[0]*(v1 - iso_cs);
    flx[1] += coeff[3]*(v1 + iso_cs);

    flx[2] += coeff[0]*v2;
    flx[2] += coeff[1];
    flx[2] += coeff[3]*v2;

    flx[3] += coeff[0]*v3;
    flx[3] += coeff[2];
    flx[3] += coeff[3]*v3;
  return;
}
} // namespace roe

} // namespace hydro
#endif // HYDRO_RSOLVERS_ROE_HYD_HPP_
