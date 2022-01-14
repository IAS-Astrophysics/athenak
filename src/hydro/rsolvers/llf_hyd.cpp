//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_hyd.cpp
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
//  \brief The LLF Riemann solver for hydrodynamics (both ideal gas and isothermal)

KOKKOS_INLINE_FUNCTION
void LLF(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real gm1 = eos.gamma - 1.0;
  Real igm1 = 1.0/gm1;
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i) {
    //--- Step 1.  Create local references for L/R states (helps compiler vectorize)

    Real &wl_idn = wl(IDN,i);
    Real &wl_ivx = wl(ivx,i);
    Real &wl_ivy = wl(ivy,i);
    Real &wl_ivz = wl(ivz,i);

    Real &wr_idn = wr(IDN,i);
    Real &wr_ivx = wr(ivx,i);
    Real &wr_ivy = wr(ivy,i);
    Real &wr_ivz = wr(ivz,i);

    Real wl_ipr, wr_ipr;
    if (eos.is_ideal) {
      wl_ipr = eos.IdealGasPressure(wl_idn, wl(IEN,i));
      wr_ipr = eos.IdealGasPressure(wr_idn, wr(IEN,i));
    }

    //--- Step 2.  Compute sum of L/R fluxes

    Real qa = wl_idn*wl_ivx;
    Real qb = wr_idn*wr_ivx;

    HydCons1D fsum;
    fsum.d  = qa        + qb;
    fsum.mx = qa*wl_ivx + qb*wr_ivx;
    fsum.my = qa*wl_ivy + qb*wr_ivy;
    fsum.mz = qa*wl_ivz + qb*wr_ivz;

    Real el,er;
    if (eos.is_ideal) {
      el = wl_ipr*igm1 + 0.5*wl_idn*(SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
      er = wr_ipr*igm1 + 0.5*wr_idn*(SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
      fsum.mx += (wl_ipr + wr_ipr);
      fsum.e  = (el + wl_ipr)*wl_ivx + (er + wr_ipr)*wr_ivx;
    } else {
      fsum.mx += (iso_cs*iso_cs)*(wl_idn + wr_idn);
    }

    //--- Step 3.  Compute max wave speed in L,R states (see Toro eq. 10.43)

    if (eos.is_ideal) {
      qa = eos.IdealHydroSoundSpeed(wl_idn, wl_ipr);
      qb = eos.IdealHydroSoundSpeed(wr_idn, wr_ipr);
    } else {
      qa = iso_cs;
      qb = iso_cs;
    }
    Real a = fmax( (fabs(wl_ivx) + qa), (fabs(wr_ivx) + qb) );

    //--- Step 4.  Compute difference in L/R states dU, multiplied by max wave speed

    HydCons1D du;
    du.d  = a*(wr_idn        - wl_idn);
    du.mx = a*(wr_idn*wr_ivx - wl_idn*wl_ivx);
    du.my = a*(wr_idn*wr_ivy - wl_idn*wl_ivy);
    du.mz = a*(wr_idn*wr_ivz - wl_idn*wl_ivz);
    if (eos.is_ideal) du.e = a*(er - el);

    //--- Step 5. Compute the LLF flux at interface (see Toro eq. 10.42).

    flx(m,IDN,k,j,i) = 0.5*(fsum.d  - du.d );
    flx(m,ivx,k,j,i) = 0.5*(fsum.mx - du.mx);
    flx(m,ivy,k,j,i) = 0.5*(fsum.my - du.my);
    flx(m,ivz,k,j,i) = 0.5*(fsum.mz - du.mz);
    if (eos.is_ideal) {flx(m,IEN,k,j,i) = 0.5*(fsum.e - du.e);}
  });

  return;
}

} // namespace hydro
