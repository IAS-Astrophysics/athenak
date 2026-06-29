#ifndef HYDRO_RSOLVERS_ROE_HYD_HPP_
#define HYDRO_RSOLVERS_ROE_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file  roe_hyd.hpp
//! \brief Roe's linearized Riemann solver for hydrodynamics (ideal gas and isothermal).
//! Reads L/R primitives from the global per-face buffers and writes a single flux entry.
//! When Roe's method fails because of negative density in the intermediate states, LLF
//! fluxes are used instead (only density, not pressure, is checked in this version).
//!
//! REFERENCES:
//! - P. Roe, "Approximate Riemann solvers, parameter vectors, and difference schemes",
//!   JCP, 43, 357 (1981).

#include <float.h>
#include <algorithm>  // max()
#include <cmath>      // sqrt()

namespace hydro {

// prototypes for functions to compute Roe fluxes from eigenmatrices
namespace roe {

KOKKOS_INLINE_FUNCTION
void RoeFluxAdb(const Real wroe[], const Real du[], const Real wli[],
                       const Real gm1, Real flx[], Real eigenvalues[], int &flag);
KOKKOS_INLINE_FUNCTION
void RoeFluxIso(const Real wroe[], const Real du[], const Real wli[],
                       const Real isocs, Real flx[], Real eigenvalues[], int &flag);

} // namespace roe

//----------------------------------------------------------------------------------------
//! \fn Roe<ivx>()
//! \brief Compute the Roe flux at face (m,k,j,i) for direction ivx, falling back to LLF
//! when an intermediate state has negative density.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void Roe(const EOS_Data &eos,
         const int m, const int k, const int j, const int i,
         const int is, const int js, const int ks,
         const DvceArray5D<Real> &wl,
         const DvceArray5D<Real> &wr,
         const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;


  const Real gm1 = eos.gamma - 1.0;
  const Real iso_cs = eos.iso_cs;

  Real wli[5], wri[5], wroe[5];
  Real fl[5], fr[5], flxi[5];
  Real ev[5], du[5];

  //--- Step 1.  Load L/R states (map the direction's velocity into the IVX slot)
  wli[IDN] = wl(m, IDN, k, j, i);
  wli[IVX] = wl(m, ivx, k, j, i);
  wli[IVY] = wl(m, ivy, k, j, i);
  wli[IVZ] = wl(m, ivz, k, j, i);

  wri[IDN] = wr(m, IDN, k, j, i);
  wri[IVX] = wr(m, ivx, k, j, i);
  wri[IVY] = wr(m, ivy, k, j, i);
  wri[IVZ] = wr(m, ivz, k, j, i);

  if (eos.is_ideal) {
    wli[IEN] = eos.IdealGasPressure(wl(m, IEN, k, j, i));
    wri[IEN] = eos.IdealGasPressure(wr(m, IEN, k, j, i));
  }

  //--- Step 2.  Roe-averaged data
  Real sqrtdl = sqrt(wli[IDN]);
  Real sqrtdr = sqrt(wri[IDN]);
  Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

  wroe[IDN] = sqrtdl*sqrtdr;
  wroe[IVX] = (sqrtdl*wli[IVX] + sqrtdr*wri[IVX])*isdlpdr;
  wroe[IVY] = (sqrtdl*wli[IVY] + sqrtdr*wri[IVY])*isdlpdr;
  wroe[IVZ] = (sqrtdl*wli[IVZ] + sqrtdr*wri[IVZ])*isdlpdr;

  Real el = 0.0, er = 0.0;
  if (eos.is_ideal) {
    el = wli[IEN]/gm1 + 0.5*wli[IDN]*(SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ]));
    er = wri[IEN]/gm1 + 0.5*wri[IDN]*(SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ]));
    wroe[IEN] = ((el + wli[IEN])/sqrtdl + (er + wri[IEN])/sqrtdr)*isdlpdr;
  }

  //--- Step 3.  L/R fluxes
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

  //--- Step 4.  Roe fluxes
  du[IDN] = wri[IDN]          - wli[IDN];
  du[IVX] = wri[IDN]*wri[IVX] - wli[IDN]*wli[IVX];
  du[IVY] = wri[IDN]*wri[IVY] - wli[IDN]*wli[IVY];
  du[IVZ] = wri[IDN]*wri[IVZ] - wli[IDN]*wli[IVZ];
  if (eos.is_ideal) { du[IEN] = er - el; }

  flxi[IDN] = 0.5*(fl[IDN] + fr[IDN]);
  flxi[IVX] = 0.5*(fl[IVX] + fr[IVX]);
  flxi[IVY] = 0.5*(fl[IVY] + fr[IVY]);
  flxi[IVZ] = 0.5*(fl[IVZ] + fr[IVZ]);
  if (eos.is_ideal) { flxi[IEN] = 0.5*(fl[IEN] + fr[IEN]); }

  int llf_flag = 0;
  if (eos.is_ideal) {
    roe::RoeFluxAdb(wroe, du, wli, gm1, flxi, ev, llf_flag);
  } else {
    roe::RoeFluxIso(wroe, du, wli, iso_cs, flxi, ev, llf_flag);
  }

  //--- Step 5.  Upwind flux if supersonic
  if (ev[0] >= 0.0) {
    flxi[IDN] = fl[IDN];
    flxi[IVX] = fl[IVX];
    flxi[IVY] = fl[IVY];
    flxi[IVZ] = fl[IVZ];
    if (eos.is_ideal) { flxi[IEN] = fl[IEN]; }
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

  //--- Step 6.  LLF fallback if intermediate density negative
  if (llf_flag != 0) {
    Real cl, cr;
    if (eos.is_ideal) {
      cl = eos.IdealHydroSoundSpeed(wli[IDN], wli[IEN]);
      cr = eos.IdealHydroSoundSpeed(wri[IDN], wri[IEN]);
    } else {
      cl = iso_cs;
      cr = iso_cs;
    }
    Real a = 0.5*fmax((fabs(wli[IVX]) + cl), (fabs(wri[IVX]) + cr));
    flxi[IDN] = 0.5*(fl[IDN] + fr[IDN]) - a*du[IDN];
    flxi[IVX] = 0.5*(fl[IVX] + fr[IVX]) - a*du[IVX];
    flxi[IVY] = 0.5*(fl[IVY] + fr[IVY]) - a*du[IVY];
    flxi[IVZ] = 0.5*(fl[IVZ] + fr[IVZ]) - a*du[IVZ];
    if (eos.is_ideal) { flxi[IEN] = 0.5*(fl[IEN] + fr[IEN]) - a*du[IEN]; }
  }

  //--- Step 7.  Store (map IVX slot back to the direction's velocity component)
  flx(m, IDN, k, j, i) = flxi[IDN];
  flx(m, ivx, k, j, i) = flxi[IVX];
  flx(m, ivy, k, j, i) = flxi[IVY];
  flx(m, ivz, k, j, i) = flxi[IVZ];
  if (eos.is_ideal) { flx(m, IEN, k, j, i) = flxi[IEN]; }
}

namespace roe {
//----------------------------------------------------------------------------------------
//! \fn RoeFluxAdb()
//! \brief Computes Roe fluxes for the conserved variables for adiabatic hydrodynamics.
//! Order of components in input vectors is (IDN,IVX,IVY,IVZ,[IEN]).
//!
//! REFERENCES:
//! - J. Stone, T. Gardiner, P. Teuben, J. Hawley, & J. Simon "Athena: A new code for
//!   astrophysical MHD", ApJS, (2008), Appendix A.

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
  Real dens = wli[IDN] + a[0];  // rem[0][0]=1
  if (dens < 0.0) llf_flag=1;

  dens += a[3];  // rem[0][3]=1
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
//! \fn RoeFluxIso()
//! \brief Computes Roe fluxes for the conserved variables for isothermal hydrodynamics.

KOKKOS_INLINE_FUNCTION
void RoeFluxIso(const Real wroe[], const Real du[], const Real wli[], const Real iso_cs,
     Real flx[], Real ev[], int &llf_flag) {
  Real v1 = wroe[IVX];
  Real v2 = wroe[IVY];
  Real v3 = wroe[IVZ];

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
  Real dens = wli[IDN] + a[0];  // rem[0][0]=1
  if (dens < 0.0) llf_flag=1;

  dens += a[3];  // rem[0][3]=1
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
