//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_mhd.cpp
//  \brief Local Lax Friedrichs (LLF) Riemann solver for MHD
//
//  Computes 1D fluxes using the LLF Riemann solver, also known as Rusanov's method.
//  This flux is very diffusive, even more diffusive than HLLE, and so it is not
//  recommended for use in applications.  However, it is useful for testing, or for
//  problems where other Riemann solvers fail.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.

// C/C++ headers
#include <algorithm>  // max(), min()

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void LLF
//  \brief The LLF Riemann solver for MHD (both adiabatic and isothermal)

KOKKOS_INLINE_FUNCTION
void LLF(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j,  const int il, const int iu,
     const int ivx, const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez)
{
  int ivy = IVX + ((ivx-IVX) + 1)%3;
  int ivz = IVX + ((ivx-IVX) + 2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;
  Real fsum[7],du[7];
  Real igm1 = 1.0/(eos.gamma - 1.0);
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Create local references for L/R states (helps compiler vectorize)

    Real &wl_idn = wl(IDN,i);
    Real &wl_ivx = wl(ivx,i);
    Real &wl_ivy = wl(ivy,i);
    Real &wl_ivz = wl(ivz,i);
    Real &wl_ipr = wl(IPR,i); // should never be referenced for adiabatic EOS
    Real &wl_iby = bl(iby,i);
    Real &wl_ibz = bl(ibz,i);

    Real &wr_idn = wr(IDN,i);
    Real &wr_ivx = wr(ivx,i);
    Real &wr_ivy = wr(ivy,i);
    Real &wr_ivz = wr(ivz,i);
    Real &wr_ipr = wr(IPR,i); // should never be referenced for adiabatic EOS
    Real &wr_iby = br(iby,i);
    Real &wr_ibz = br(ibz,i);

    Real &bxi = bx(m,k,j,i);

    //--- Step 2.  Compute sum of L/R fluxes

    Real qa = wl_idn*wl_ivx;
    Real qb = wr_idn*wr_ivx;
    Real qc = 0.5*(SQR(wl_iby) + SQR(wl_ibz) - SQR(bxi));
    Real qd = 0.5*(SQR(wr_iby) + SQR(wr_ibz) - SQR(bxi));

    fsum[IDN] = qa        + qb;
    fsum[IVX] = qa*wl_ivx + qb*wr_ivx + qc + qd;
    fsum[IVY] = qa*wl_ivy + qb*wr_ivy - bxi*(wl_iby + wr_iby);
    fsum[IVZ] = qa*wl_ivz + qb*wr_ivz - bxi*(wl_ibz + wr_ibz);
    fsum[5  ] = wl_iby*wl_ivx + wr_iby*wr_ivx - bxi*(wl_ivy + wr_ivy);
    fsum[6  ] = wl_ibz*wl_ivx + wr_ibz*wr_ivx - bxi*(wl_ivz + wr_ivz);

    Real el,er;
    if (eos.is_adiabatic) {
      el = wl_ipr*igm1 + 0.5*wl_idn*(SQR(wl_ivx)+SQR(wl_ivy)+SQR(wl_ivz)) + qc + SQR(bxi);
      er = wr_ipr*igm1 + 0.5*wr_idn*(SQR(wr_ivx)+SQR(wr_ivy)+SQR(wr_ivz)) + qd + SQR(bxi);
      fsum[IVX] += (wl_ipr + wr_ipr);
      fsum[IEN] = (el + wl_ipr + qc)*wl_ivx + (er + wr_ipr + qd)*wr_ivx;
      fsum[IEN] -= bxi*(wl_iby*wl_ivy + wl_ibz*wl_ivz);
      fsum[IEN] -= bxi*(wr_iby*wr_ivy + wr_ibz*wr_ivz);
    } else {
      fsum[IVX] += (iso_cs*iso_cs)*(wl_idn + wr_idn);
    }

    //--- Step 3.  Compute max wave speed in L,R states (see Toro eq. 10.43)

    if (eos.is_adiabatic) {
      qa = eos.FastMagnetosonicSpeed(wl_idn, wl_ipr, bxi, wl_iby, wl_ibz);
      qb = eos.FastMagnetosonicSpeed(wr_idn, wr_ipr, bxi, wr_iby, wr_ibz);
    } else {
      qa = eos.FastMagnetosonicSpeed(wl_idn, bxi, wl_iby, wl_ibz);
      qb = eos.FastMagnetosonicSpeed(wr_idn, bxi, wr_iby, wr_ibz);
    }
    Real a = fmax( (fabs(wl_ivx) + qa), (fabs(wr_ivx) + qb) );

    //--- Step 4.  Compute difference in L/R states dU, multiplied by max wave speed

    du[IDN] = a*(wr_idn        - wl_idn);
    du[IVX] = a*(wr_idn*wr_ivx - wl_idn*wl_ivx);
    du[IVY] = a*(wr_idn*wr_ivy - wl_idn*wl_ivy);
    du[IVZ] = a*(wr_idn*wr_ivz - wl_idn*wl_ivz);
    if (eos.is_adiabatic) du[IEN] = a*(er - el);
    du[5  ] = a*(wr_iby - wl_iby);
    du[6  ] = a*(wr_ibz - wl_ibz);

    //--- Step 5.  Compute the LLF flux at interface (see Toro eq. 10.42).

    flx(m,IDN,k,j,i) = 0.5*(fsum[IDN] - du[IDN]);
    flx(m,ivx,k,j,i) = 0.5*(fsum[IVX] - du[IVX]);
    flx(m,ivy,k,j,i) = 0.5*(fsum[IVY] - du[IVY]);
    flx(m,ivz,k,j,i) = 0.5*(fsum[IVZ] - du[IVZ]);
    if (eos.is_adiabatic) {flx(m,IEN,k,j,i) = 0.5*(fsum[IEN] - du[IEN]);}
    ey(m,k,j,i) = -0.5*(fsum[5] - du[5]);
    ez(m,k,j,i) =  0.5*(fsum[6] - du[6]);
  });

  return;
}

} // namespace mhd
