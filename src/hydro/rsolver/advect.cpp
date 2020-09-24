//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advect.cpp
//  \brief Riemann solver for pure advection problems (v = constant).  Simply computes the
//  upwind flux of each vriable.

#include <algorithm>  // max(), min()

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Advection
//  \brief An advection Riemann solver for hydrodynamics (both adiabatic and isothermal)

KOKKOS_INLINE_FUNCTION
void Advect(TeamMember_t const &member, const EOSData eos,  const int il, const int iu,
     const int ivx, const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
     AthenaScratch2D<Real> &flx)
{
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5];
  Real gm1 = eos.gamma - 1.0;

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

    //--- Step 3.  Compute upwind fluxes

    if (wli[IVX] >= 0.0) {

      Real mxl = wli[IDN]*wli[IVX];
      flx(IDN,i) = mxl;
      flx(ivx,i) = mxl*wli[IVX];
      flx(ivy,i) = mxl*wli[IVY];
      flx(ivz,i) = mxl*wli[IVZ];
      if (eos.is_adiabatic) { flx(IEN,i) = (wli[IPR]/gm1 + 0.5*mxl*wli[IVX])*wli[IVX]; }

    } else {

      Real mxr = wri[IDN]*wri[IVX];
      flx(IDN,i) = mxr;
      flx(ivx,i) = mxr*wri[IVX];
      flx(ivy,i) = mxr*wri[IVY];
      flx(ivz,i) = mxr*wri[IVZ];
      if (eos.is_adiabatic) { flx(IEN,i) = (wri[IPR]/gm1 + 0.5*mxr*wri[IVX])*wri[IVX]; }

    }
  });

  return;
}

} // namespace hydro
