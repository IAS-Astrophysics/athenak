#ifndef HYDRO_RSOLVERS_HLLE_HYD_HPP_
#define HYDRO_RSOLVERS_HLLE_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_hyd.hpp
//! \brief Contains HLLE Riemann solver for hydrodynamics
//!
//! Computes fluxes using the Harten-Lax-vanLeer-Einfeldt (HLLE) Riemann solver.  This
//! flux is very diffusive, especially for contacts, and so it is not recommended for
//! applications. However it is better than LLF. Einfeldt et al.(1991) prove it is
//! positively conservative (cannot return negative densities or pressure), so it is a
//! useful option when other approximate solvers fail and/or when extra dissipation is
//! needed.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
//! - Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)
//! - A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type
//!   schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983).

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void HLLE
//! \brief The HLLE Riemann solver for hydrodynamics (both ideal gas and isothermal)

KOKKOS_INLINE_FUNCTION
void HLLE(TeamMember_t const &member, const EOS_Data &eos,
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
      wl_ipr = eos.IdealGasPressure(wl(IEN,i));
      wr_ipr = eos.IdealGasPressure(wr(IEN,i));
    }

    //--- Step 2.  Compute Roe-averaged state

    Real sqrtdl = sqrt(wl_idn);
    Real sqrtdr = sqrt(wr_idn);
    Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

    Real wroe_ivx = (sqrtdl*wl_ivx + sqrtdr*wr_ivx)*isdlpdr;
    Real wroe_ivy = (sqrtdl*wl_ivy + sqrtdr*wr_ivy)*isdlpdr;
    Real wroe_ivz = (sqrtdl*wl_ivz + sqrtdr*wr_ivz)*isdlpdr;

    // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for ideal gas EOS,
    // rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
    Real el,er,hroe;
    if (eos.is_ideal) {
      el = wl_ipr*igm1 + 0.5*wl_idn*(SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
      er = wr_ipr*igm1 + 0.5*wr_idn*(SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
      hroe = ((el + wl_ipr)/sqrtdl + (er + wr_ipr)/sqrtdr)*isdlpdr;
    }

    //--- Step 3.  Compute sound speed in L,R, and Roe-averaged states

    Real qa,qb;
    Real a  = iso_cs;
    if (eos.is_ideal) {
      qa = eos.IdealHydroSoundSpeed(wl_idn, wl_ipr);
      qb = eos.IdealHydroSoundSpeed(wr_idn, wr_ipr);
      a = hroe - 0.5*(SQR(wroe_ivx) + SQR(wroe_ivy) + SQR(wroe_ivz));
      a = (a < 0.0) ? 0.0 : sqrt(gm1*a);
    } else {
      qa = iso_cs;
      qb = iso_cs;
    }

    //--- Step 4. Compute the L/R wave speeds based on L/R and Roe-averaged values

    Real al = fmin((wroe_ivx - a),(wl_ivx - qa));
    Real ar = fmax((wroe_ivx + a),(wr_ivx + qb));

    // following min/max set to TINY_NUMBER to fix bug found in converging supersonic flow
    Real bp = (ar > 0.0) ? ar : 1.0e-20;
    Real bm = (al < 0.0) ? al : -1.0e-20;

    //-- Step 5. Compute L/R fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R

    qa = wl_ivx - bm;
    qb = wr_ivx - bp;

    HydCons1D fl, fr;
    fl.d  = wl_idn*qa;
    fr.d  = wr_idn*qb;

    fl.mx = wl_idn*wl_ivx*qa;
    fr.mx = wr_idn*wr_ivx*qb;

    fl.my = wl_idn*wl_ivy*qa;
    fr.my = wr_idn*wr_ivy*qb;

    fl.mz = wl_idn*wl_ivz*qa;
    fr.mz = wr_idn*wr_ivz*qb;

    if (eos.is_ideal) {
      fl.mx += wl_ipr;
      fr.mx += wr_ipr;
      fl.e  = el*qa + wl_ipr*wl_ivx;
      fr.e  = er*qb + wr_ipr*wr_ivx;
    } else {
      fl.mx += (iso_cs*iso_cs)*wl_idn;
      fr.mx += (iso_cs*iso_cs)*wr_idn;
    }

    //--- Step 6. Compute the HLLE flux at interface. Formulae below equivalent to
    // Toro eq. 10.20, or Einfeldt et al. (1991) eq. 4.4b

    qa = 0.0;
    if (bp != bm) qa = 0.5*(bp + bm)/(bp - bm);

    flx(m,IDN,k,j,i) = 0.5*(fl.d  + fr.d ) + qa*(fl.d  - fr.d );
    flx(m,ivx,k,j,i) = 0.5*(fl.mx + fr.mx) + qa*(fl.mx - fr.mx);
    flx(m,ivy,k,j,i) = 0.5*(fl.my + fr.my) + qa*(fl.my - fr.my);
    flx(m,ivz,k,j,i) = 0.5*(fl.mz + fr.mz) + qa*(fl.mz - fr.mz);
    if (eos.is_ideal) flx(m,IEN,k,j,i) = 0.5*(fl.e + fr.e) + qa*(fl.e - fr.e);
  });

  return;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLE_HYD_HPP_
