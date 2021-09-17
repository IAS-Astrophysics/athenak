//======================================================================================== // AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srhyd.cpp
//  \brief Local Lax Friedrichs (LLF) Riemann solver for special relativistic hydro
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
//  \brief The LLF Riemann solver for SR hydrodynamics

KOKKOS_INLINE_FUNCTION
void LLF_SR(TeamMember_t const &member, const EOS_Data &eos, const CoordData &coord, 
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx)
{
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Create local references for L/R states (helps compiler vectorize)
    // Recall in SR the primitive variables are (\rho, u^i, P_gas), where \rho is the
    // mass density in the comoving/fluid frame, u^i = \gamma v^i are the spatial
    // components of the 4-velocity (v^i is the 3-velocity), and P_gas is the pressure.

    Real &wl_idn=wl(IDN,i);
    Real &wl_ivx=wl(ivx,i);
    Real &wl_ivy=wl(ivy,i);
    Real &wl_ivz=wl(ivz,i);
    Real &wl_ipr=wl(IPR,i);

    Real &wr_idn=wr(IDN,i);
    Real &wr_ivx=wr(ivx,i);
    Real &wr_ivy=wr(ivy,i);
    Real &wr_ivz=wr(ivz,i);
    Real &wr_ipr=wr(IPR,i);

    Real u2l = SQR(wl_ivz) + SQR(wl_ivy) + SQR(wl_ivx);
    Real u2r = SQR(wr_ivz) + SQR(wr_ivy) + SQR(wr_ivx);
    
    Real u0l  = sqrt(1. + u2l); // Lorentz factor in L-state
    Real u0r  = sqrt(1. + u2r); // Lorentz factor in R-state

    // FIXME ERM: Ideal fluid for now
    Real wgas_l = wl_idn + gamma_prime * wl_ipr;  // total enthalpy in L-state
    Real wgas_r = wr_idn + gamma_prime * wr_ipr;  // total enthalpy in R-state

    //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

    Real lp_l, lm_l;
    eos.WaveSpeedsSR(wgas_l, wl_ipr, wl_ivx/u0l, (1.0 + u2l), lp_l, lm_l);

    Real lp_r, lm_r;
    eos.WaveSpeedsSR(wgas_r, wr_ipr, wr_ivx/u0r, (1.0 + u2r), lp_r, lm_r);

    Real qa = fmax(-fmin(lm_l,lm_r), 0.0);
    Real a = fmax(fmax(lp_l,lp_r), qa);
    
    //--- Step 3.  Compute sum of L/R fluxes

    qa = wgas_l * wl_ivx;
    Real qb = wgas_r * wr_ivx;

    HydCons1D fsum;
    fsum.d  = wl_idn * wl_ivx + wr_idn * wr_ivx;
    fsum.mx = qa*wl_ivx + qb*wr_ivx + (wl_ipr + wr_ipr);
    fsum.my = qa*wl_ivy + qb*wr_ivy;
    fsum.mz = qa*wl_ivz + qb*wr_ivz;
    fsum.e  = qa*u0l + qb*u0r;

//    Real el = wgas_l*u0l*u0l - wl_ipr - wl_idn*u0l;
//    Real er = wgas_r*u0r*u0r - wr_ipr - wr_idn*u0r;
//    fsum.e  = (er + wr_ipr)*wr_ivx/u0r + (el + wl_ipr)*wl_ivx/u0l;

    //--- Step 4.  Compute difference dU = U_R - U_L multiplied by max wave speed

    HydCons1D du;
    qa = wgas_r*u0r;
    qb = wgas_l*u0l;
    Real er = qa*u0r - wr_ipr;
    Real el = qb*u0l - wl_ipr;
    du.d  = a*(u0r*wr_idn - u0l*wl_idn);
    du.mx = a*( qa*wr_ivx -  qb*wl_ivx);
    du.my = a*( qa*wr_ivy -  qb*wl_ivy);
    du.mz = a*( qa*wr_ivz -  qb*wl_ivz);
    du.e  = a*(er - el);

    //--- Step 5. Store results into 3D array of fluxes

    flx(m,IDN,k,j,i) = 0.5*(fsum.d  - du.d );
    flx(m,ivx,k,j,i) = 0.5*(fsum.mx - du.mx);
    flx(m,ivy,k,j,i) = 0.5*(fsum.my - du.my);
    flx(m,ivz,k,j,i) = 0.5*(fsum.mz - du.mz);
    flx(m,IEN,k,j,i) = 0.5*(fsum.e  - du.e );

    // We evolve tau = E - D
    flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);

  });
  return;
}

} // namespace hydro
