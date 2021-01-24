//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advect.cpp
//  \brief Riemann solver for pure advection problems (v = constant).  Simply computes the
//  upwind flux of each vriable.

#include <algorithm>  // max(), min()

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Advection
//  \brief An advection Riemann solver for hydrodynamics (both adiabatic and isothermal)

KOKKOS_INLINE_FUNCTION
void Advect(TeamMember_t const &member, const EOS_Data eos,  const int il, const int iu,
     const int ivx, const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     ScrArray2D<Real> &flx)
{
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5];
  Real gm1 = eos.gamma - 1.0;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //  Load L states into local variables, *** overwritten on output ***
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    if (eos.is_adiabatic) { wli[IPR]=wl(IPR,i); }

    //  Compute upwind fluxes

    if (wli[IVX] >= 0.0) {
      Real mxl = wli[IDN]*wli[IVX];
      flx(IDN,i) = mxl;
      flx(ivx,i) = mxl*wli[IVX];
      flx(ivy,i) = mxl*wli[IVY];
      flx(ivz,i) = mxl*wli[IVZ];
      if (eos.is_adiabatic) {flx(IEN,i) = (wli[IPR]/gm1 + 0.5*mxl*wli[IVX])*wli[IVX];}
    } else {
      Real mxr = wr(IDN,i)*wr(ivx,i);
      flx(IDN,i) = mxr;
      flx(ivx,i) = mxr*wr(ivx,i);
      flx(ivy,i) = mxr*wr(ivy,i);
      flx(ivz,i) = mxr*wr(ivz,i);
      if (eos.is_adiabatic) {flx(IEN,i) = (wr(IPR,i)/gm1 + 0.5*mxr*wr(ivx,i))*wr(ivx,i);}
    }

  });

  return;
}

} // namespace hydro
