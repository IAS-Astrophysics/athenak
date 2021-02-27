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
void LLF(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j, const int il, const int iu,
     const int ivx, const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     DvceArray5D<Real> flx)
{
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5],fave[5];
  Real igm1 = 1.0/(eos.gamma - 1.0);
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    if (eos.is_adiabatic) {wli[IPR]=wl(IPR,i);}

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    if (eos.is_adiabatic) {wri[IPR]=wr(IPR,i);}

    //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

    Real qa,qb;
    if (eos.is_adiabatic) {
      qa = eos.SoundSpeed(wli[IPR],wli[IDN]);
      qb = eos.SoundSpeed(wri[IPR],wri[IDN]);
    } else {
      qa = iso_cs;
      qb = iso_cs;
    }
    Real a  = 0.5*fmax( (fabs(wli[IVX]) + qa), (fabs(wri[IVX]) + qb) );

    //--- Step 3.  Compute average of L/R fluxes

    qa = wli[IDN]*wli[IVX];
    qb = wri[IDN]*wri[IVX];

    fave[IDN] = qa          + qb;
    fave[IVX] = qa*wli[IVX] + qb*wri[IVX];
    fave[IVY] = qa*wli[IVY] + qb*wri[IVY];
    fave[IVZ] = qa*wli[IVZ] + qb*wri[IVZ];

    Real el,er;
    if (eos.is_adiabatic) {
      el = wli[IPR]*igm1 + 0.5*wli[IDN]*(SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ]));
      er = wri[IPR]*igm1 + 0.5*wri[IDN]*(SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ]));
      fave[IVX] += (wli[IPR] + wri[IPR]);
      fave[IEN] = (el + wli[IPR])*wli[IVX] + (er + wri[IPR])*wri[IVX];
    } else {
      fave[IVX] += (iso_cs*iso_cs)*(wli[IDN] + wri[IDN]);
    }

    //--- Step 4. Store results into 3D array of fluxes

    flx(m,IDN,k,j,i) = 0.5*(fave[IDN]) - a*(wri[IDN]          - wli[IDN]);
    flx(m,ivx,k,j,i) = 0.5*(fave[IVX]) - a*(wri[IDN]*wri[IVX] - wli[IDN]*wli[IVX]);
    flx(m,ivy,k,j,i) = 0.5*(fave[IVY]) - a*(wri[IDN]*wri[IVY] - wli[IDN]*wli[IVY]);
    flx(m,ivz,k,j,i) = 0.5*(fave[IVZ]) - a*(wri[IDN]*wri[IVZ] - wli[IDN]*wli[IVZ]);
    if (eos.is_adiabatic) {flx(m,IEN,k,j,i) = 0.5*(fave[IEN]) - a*(er - el);}

  });
  return;
}

} // namespace hydro
