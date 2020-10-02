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

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void LLF
//  \brief The LLF Riemann solver for hydrodynamics (both adiabatic and isothermal)

KOKKOS_INLINE_FUNCTION
void LLF(TeamMember_t const &member, const EOSData &eos, const int il, const int iu,
     const int ivx, const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
     AthenaScratch2D<Real> &flx)
{
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5],du[5];
  Real fl[5],fr[5];
  Real gm1 = eos.gamma - 1.0;
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    if (eos.is_adiabatic) { wli[IPR]=wl(IPR,i); }

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    if (eos.is_adiabatic) { wri[IPR]=wr(IPR,i); }

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

    //--- Step 3.  Compute L/R fluxes

    qa = wli[IDN]*wli[IVX];
    fl[IDN] = qa;
    fl[IVX] = qa*wli[IVX];
    fl[IVY] = qa*wli[IVY];
    fl[IVZ] = qa*wli[IVZ];

    qa = wri[IDN]*wri[IVX];
    fr[IDN] = qa;
    fr[IVX] = qa*wri[IVX];
    fr[IVY] = qa*wri[IVY];
    fr[IVZ] = qa*wri[IVZ];

    Real el,er;
    if (eos.is_adiabatic) {
      el = wli[IPR]/gm1 + 0.5*wli[IDN]*(SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ]));
      er = wri[IPR]/gm1 + 0.5*wri[IDN]*(SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ]));
      fl[IVX] += wli[IPR];
      fr[IVX] += wri[IPR];
      fl[IEN] = (el + wli[IPR])*wli[IVX];
      fr[IEN] = (er + wri[IPR])*wri[IVX];
    } else {
      fl[IVX] += (iso_cs*iso_cs)*wli[IDN];
      fr[IVX] += (iso_cs*iso_cs)*wri[IDN];
    }

    //--- Step 4.  Compute difference in L/R states dU

    du[IDN] = wri[IDN]          - wli[IDN];
    du[IVX] = wri[IDN]*wri[IVX] - wli[IDN]*wli[IVX];
    du[IVY] = wri[IDN]*wri[IVY] - wli[IDN]*wli[IVY];
    du[IVZ] = wri[IDN]*wri[IVZ] - wli[IDN]*wli[IVZ];
    if (eos.is_adiabatic) { du[IEN] = er - el; }

    //--- Step 5. Store results into 3D array of fluxes

    flx(IDN,i) = 0.5*(fl[IDN] + fr[IDN]) - a*du[IDN];
    flx(ivx,i) = 0.5*(fl[IVX] + fr[IVX]) - a*du[IVX];
    flx(ivy,i) = 0.5*(fl[IVY] + fr[IVY]) - a*du[IVY];
    flx(ivz,i) = 0.5*(fl[IVZ] + fr[IVZ]) - a*du[IVZ];
    if (eos.is_adiabatic) {flx(IEN,i) = 0.5*(fl[IEN] + fr[IEN]) - a*du[IEN];}

  });
  return;
}

} // namespace hydro
