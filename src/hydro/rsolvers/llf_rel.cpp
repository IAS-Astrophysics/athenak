
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf.cpp
//  \brief Local Lax Friedrichs (LLF) Riemann solver for hydrodynamics
//
//  Computes 1D fluxes using the LLF Riemann solver, also known as Rusanov's method.
//  This flux is very diffusive, even more diffusive than HLLE, and so it is not
//  recommended for use in applications.  However, it is useful for testing, or for
//  problems where other Riemann solvers fail.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void LLF
//  \brief The LLF Riemann solver for hydrodynamics (both adiabatic and isothermal)

KOKKOS_INLINE_FUNCTION
void LLF_rel(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j, const int il, const int iu,
     const int ivx, const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     DvceArray5D<Real> flx)
{


  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5],du[5];
  Real fl[5],fr[5];
  Real gm1 = eos.gamma - 1.0;
//  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    wli[IPR]=wl(IPR,i);

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    wri[IPR]=wr(IPR,i);

    Real u2l = SQR(wli[IVZ]) + SQR(wli[IVY]) + SQR(wli[IVX]);
    Real u2r = SQR(wri[IVZ]) + SQR(wri[IVY]) + SQR(wri[IVX]);
    
    Real u0l  = sqrt(1. + u2l);
    Real u0r  = sqrt(1. + u2r);

    // FIXME ERM: Ideal fluid for now
    Real wgas_l = wli[IDN] + (eos.gamma/gm1) * wli[IPR];
    Real wgas_r = wri[IDN] + (eos.gamma/gm1) * wri[IPR];

//    wri[IVX] /= gammar;
//    wri[IVY] /= gammar;
//    wri[IVZ] /= gammar;

//    wli[IVX] /= gammal;
//    wli[IVY] /= gammal;
//    wli[IVZ] /= gammal;

    //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

    Real lm,lp,qa,qb;
    eos.SoundSpeed_SR(wgas_l, wli[IPR], wli[IVX]/u0l, 1.+u2l, lp, lm);
    eos.SoundSpeed_SR(wgas_r, wri[IPR], wri[IVX]/u0r, 1.+u2r, qb,qa);

    //FIXME ERM: Check this
    qa = fmax(-fmin(lm,qa), 0.);
    Real a = fmax(fmax(lp,qb), qa);

//    a= 0.5;


    //--- Step 3.  Compute L/R fluxes

    fl[IDN] = wli[IDN] * wli[IVX];
    qa = wgas_l * wli[IVX];
    fl[IVX] = qa*wli[IVX] + wli[IPR];
    fl[IVY] = qa*wli[IVY];
    fl[IVZ] = qa*wli[IVZ];

    fr[IDN] = wri[IDN] * wri[IVX];
    qa = wgas_r * wri[IVX];
    fr[IVX] = qa*wri[IVX] + wri[IPR];
    fr[IVY] = qa*wri[IVY];
    fr[IVZ] = qa*wri[IVZ];


    Real el = wgas_l*u0l*u0l - wli[IPR] - wli[IDN]*u0l;
    Real er = wgas_r*u0r*u0r - wri[IPR] - wri[IDN]*u0r;
    fr[IEN] = (er + wri[IPR])*wri[IVX]/u0r;
    fl[IEN] = (el + wli[IPR])*wli[IVX]/u0l;

//    fl[IEN] = (((wli[IPR] / gm1) + wli[IPR]) * u0l + (wli[IDN]/(1.+ u0l)*u2l))*wli[IVX];
//    fr[IEN] = (((wri[IPR] / gm1) + wri[IPR]) * u0r + (wri[IDN]/(1.+ u0r)*u2r))*wri[IVX];
//    fl[IEN] = (((wli[IPR] / gm1) + wli[IPR]) * u0l + (wli[IDN]/(1.+ u0l)*u2l))*wli[IVX];
//    fr[IEN] = (((wri[IPR] / gm1) + wri[IPR]) * u0r + (wri[IDN]/(1.+ u0r)*u2r))*wri[IVX];

    du[IDN] = wri[IDN]*u0r          - wli[IDN] * u0l;
    du[IVX] = wgas_r*u0r*wri[IVX] - wgas_l*u0l*wli[IVX];
    du[IVY] = wgas_r*u0r*wri[IVY] - wgas_l*u0l*wli[IVY];
    du[IVZ] = wgas_r*u0r*wri[IVZ] - wgas_l*u0l*wli[IVZ];
//    du[IEN] = (wri[IPR] / gm1) * u0r*u0r + ( wri[IPR] + wri[IDN]*u0r / (1.+ u0r))*u2r;
//    du[IEN]-= (wli[IPR] / gm1) * u0l*u0l + ( wli[IPR] + wli[IDN]*u0l / (1.+ u0l))*u2l;
    du[IEN] = er - el;

    //--- Step 5. Store results into 3D array of fluxes

    flx(m,IDN,k,j,i) = 0.5*(fl[IDN] + fr[IDN]) - a*du[IDN];
    flx(m,ivx,k,j,i) = 0.5*(fl[IVX] + fr[IVX]) - a*du[IVX];
    flx(m,ivy,k,j,i) = 0.5*(fl[IVY] + fr[IVY]) - a*du[IVY];
    flx(m,ivz,k,j,i) = 0.5*(fl[IVZ] + fr[IVZ]) - a*du[IVZ];
    flx(m,IEN,k,j,i) = 0.5*(fl[IEN] + fr[IEN]) - a*du[IEN];

  });
  return;
}

} // namespace hydro
