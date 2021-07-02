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
  Real fsum[5],du[5];
  Real igm1 = 1.0/(eos.gamma - 1.0);
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Create local references for L/R states (helps compiler vectorize)

    Real &wli_idn = wl(IDN,i);
    Real &wli_ivx = wl(ivx,i);
    Real &wli_ivy = wl(ivy,i);
    Real &wli_ivz = wl(ivz,i);
    Real &wli_ipr = wl(IPR,i);  // should never be referenced for adiabatic EOS

    Real &wri_idn = wr(IDN,i);
    Real &wri_ivx = wr(ivx,i);
    Real &wri_ivy = wr(ivy,i);
    Real &wri_ivz = wr(ivz,i);
    Real &wri_ipr = wr(IPR,i);  // should never be referenced for adiabatic EOS

    //--- Step 2.  Compute sum of L/R fluxes

    Real qa = wli_idn*wli_ivx;
    Real qb = wri_idn*wri_ivx;

    fsum[IDN] = qa         + qb;
    fsum[IVX] = qa*wli_ivx + qb*wri_ivx;
    fsum[IVY] = qa*wli_ivy + qb*wri_ivy;
    fsum[IVZ] = qa*wli_ivz + qb*wri_ivz;

    Real el,er;
    if (eos.is_adiabatic) {
      el = wli_ipr*igm1 + 0.5*wli_idn*(SQR(wli_ivx) + SQR(wli_ivy) + SQR(wli_ivz));
      er = wri_ipr*igm1 + 0.5*wri_idn*(SQR(wri_ivx) + SQR(wri_ivy) + SQR(wri_ivz));
      fsum[IVX] += (wli_ipr + wri_ipr);
      fsum[IEN] = (el + wli_ipr)*wli_ivx + (er + wri_ipr)*wri_ivx;
    } else {
      fsum[IVX] += (iso_cs*iso_cs)*(wli_idn + wri_idn);
    }

    //--- Step 3.  Compute max wave speed in L,R states (see Toro eq. 10.43)

    if (eos.is_adiabatic) {
      qa = eos.SoundSpeed(wli_ipr,wli_idn);
      qb = eos.SoundSpeed(wri_ipr,wri_idn);
    } else {
      qa = iso_cs;
      qb = iso_cs;
    }
    Real a = fmax( (fabs(wli_ivx) + qa), (fabs(wri_ivx) + qb) );

    //--- Step 4.  Compute difference in L/R states dU, multiplied by max wave speed

    du[IDN] = a*(wri_idn         - wli_idn);
    du[IVX] = a*(wri_idn*wri_ivx - wli_idn*wli_ivx);
    du[IVY] = a*(wri_idn*wri_ivy - wli_idn*wli_ivy);
    du[IVZ] = a*(wri_idn*wri_ivz - wli_idn*wli_ivz);
    if (eos.is_adiabatic) du[IEN] = a*(er - el);

    //--- Step 5. Compute the LLF flux at interface (see Toro eq. 10.42).

    flx(m,IDN,k,j,i) = 0.5*(fsum[IDN] - du[IDN]);
    flx(m,ivx,k,j,i) = 0.5*(fsum[IVX] - du[IVX]);
    flx(m,ivy,k,j,i) = 0.5*(fsum[IVY] - du[IVY]);
    flx(m,ivz,k,j,i) = 0.5*(fsum[IVZ] - du[IVZ]);
    if (eos.is_adiabatic) {flx(m,IEN,k,j,i) = 0.5*(fsum[IEN] - du[IEN]);}

  });

  return;
}

} // namespace hydro
