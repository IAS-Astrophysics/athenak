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
  Real du[7],fl[7],fr[7];
  Real igm1 = 1.0/(eos.gamma - 1.0);
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Create local references for L/R states
    Real &wl_idn=wl(IDN,i);
    Real &wl_ivx=wl(ivx,i);
    Real &wl_ivy=wl(ivy,i);
    Real &wl_ivz=wl(ivz,i);
    Real &wl_iby=bl(iby,i);
    Real &wl_ibz=bl(ibz,i);

    Real &wr_idn=wr(IDN,i);
    Real &wr_ivx=wr(ivx,i);
    Real &wr_ivy=wr(ivy,i);
    Real &wr_ivz=wr(ivz,i);
    Real &wr_iby=br(iby,i);
    Real &wr_ibz=br(ibz,i);

    Real &bxi = bx(m,k,j,i);

    //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

    Real qa,qb;
    if (eos.is_adiabatic) {
      Real &wl_ipr=wl(IPR,i);
      Real &wr_ipr=wr(IPR,i);
      qa = eos.FastMagnetosonicSpeed(wl_idn,wl_ipr,bxi,wl_iby,wl_ibz);
      qb = eos.FastMagnetosonicSpeed(wr_idn,wr_ipr,bxi,wr_iby,wr_ibz);
    } else {
      qa = eos.FastMagnetosonicSpeed(wl_idn,bxi,wl_iby,wl_ibz);
      qb = eos.FastMagnetosonicSpeed(wr_idn,bxi,wr_iby,wr_ibz);
    }
    Real a  = 0.5*fmax( (fabs(wl_ivx) + qa), (fabs(wr_ivx) + qb) );

    //--- Step 3.  Compute L/R fluxes

    qa = wl_idn*wl_ivx;
    Real pbl = 0.5*(bxi*bxi + SQR(wl_iby) + SQR(wl_ibz));
    fl[IDN] = qa;
    fl[IVX] = qa*wl_ivx + pbl - SQR(bxi);
    fl[IVY] = qa*wl_ivy - bxi*wl_iby;
    fl[IVZ] = qa*wl_ivz - bxi*wl_ibz;
    fl[5  ] = wl_iby*wl_ivx - bxi*wl_ivy;
    fl[6  ] = wl_ibz*wl_ivx - bxi*wl_ivz;

    qa = wr_idn*wr_ivx;
    Real pbr = 0.5*(bxi*bxi + SQR(wr_iby) + SQR(wr_ibz));
    fr[IDN] = qa;
    fr[IVX] = qa*wr_ivx + pbr - SQR(bxi);
    fr[IVY] = qa*wr_ivy - bxi*wr_iby;
    fr[IVZ] = qa*wr_ivz - bxi*wr_ibz;
    fr[5  ] = wr_iby*wr_ivx - bxi*wr_ivy;
    fr[6  ] = wr_ibz*wr_ivx - bxi*wr_ivz;

    Real el,er;
    if (eos.is_adiabatic) {
      Real &wl_ipr=wl(IPR,i);
      Real &wr_ipr=wr(IPR,i);
      el = wl_ipr*igm1 + 0.5*wl_idn*(SQR(wl_ivx)+SQR(wl_ivy)+SQR(wl_ivz)) + pbl;
      er = wr_ipr*igm1 + 0.5*wr_idn*(SQR(wr_ivx)+SQR(wr_ivy)+SQR(wr_ivz)) + pbr;
      fl[IVX] += wl_ipr;
      fr[IVX] += wr_ipr;
      fl[IEN] = (el + wl_ipr + pbl - bxi*bxi)*wl_ivx;
      fr[IEN] = (er + wr_ipr + pbr - bxi*bxi)*wr_ivx;
      fl[IEN] -= bxi*(wl_iby*wl_ivy + wl_ibz*wl_ivz);
      fr[IEN] -= bxi*(wr_iby*wr_ivy + wr_ibz*wr_ivz);
    } else {
      fl[IVX] += (iso_cs*iso_cs)*wl_idn;
      fr[IVX] += (iso_cs*iso_cs)*wr_idn;
    }

    //--- Step 4.  Compute difference in L/R states dU

    du[IDN] = wr_idn        - wl_idn;
    du[IVX] = wr_idn*wr_ivx - wl_idn*wl_ivx;
    du[IVY] = wr_idn*wr_ivy - wl_idn*wl_ivy;
    du[IVZ] = wr_idn*wr_ivz - wl_idn*wl_ivz;
    du[5  ] = wr_iby - wl_iby;
    du[6  ] = wr_ibz - wl_ibz;

    //--- Step 5.  Compute the LLF flux at interface (see Toro eq. 10.42).

    flx(m,IDN,k,j,i) = 0.5*(fl[IDN] + fr[IDN]) - a*du[IDN];
    flx(m,ivx,k,j,i) = 0.5*(fl[IVX] + fr[IVX]) - a*du[IVX];
    flx(m,ivy,k,j,i) = 0.5*(fl[IVY] + fr[IVY]) - a*du[IVY];
    flx(m,ivz,k,j,i) = 0.5*(fl[IVZ] + fr[IVZ]) - a*du[IVZ];
    if (eos.is_adiabatic) {
      flx(m,IEN,k,j,i) = 0.5*(fl[IEN] + fr[IEN]) - a*(er - el);
    }
    ey(m,k,j,i) = -0.5*(fl[5  ] + fr[5  ]) + a*du[5  ];
    ez(m,k,j,i) = 0.5*(fl[6  ] + fr[6  ]) - a*du[6  ];
  });

  return;
}

} // namespace mhd
