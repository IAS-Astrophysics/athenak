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
  Real gm1 = eos.gamma - 1.0;

  par_for_inner(member, il, iu, [&](const int i)
  {
    if (wl(IVX,i) >= 0.0) {

      Real mxl = wl(IDN,i)*wl(IVX,i);
      flx(IDN,i) = mxl;
      flx(ivx,i) = mxl*wl(IVX,i);
      flx(ivy,i) = mxl*wl(IVY,i);
      flx(ivz,i) = mxl*wl(IVZ,i);
      if (eos.is_adiabatic) {flx(IEN,i) = (wl(IPR,i)/gm1 + 0.5*mxl*wl(IVX,i))*wl(IVX,i);}

    } else {

      Real mxr = wr(IDN,i)*wr(IVX,i);
      flx(IDN,i) = mxr;
      flx(ivx,i) = mxr*wr(IVX,i);
      flx(ivy,i) = mxr*wr(IVY,i);
      flx(ivz,i) = mxr*wr(IVZ,i);
      if (eos.is_adiabatic) {flx(IEN,i) = (wr(IPR,i)/gm1 + 0.5*mxr*wr(IVX,i))*wr(IVX,i);}

    }
  });

  return;
}

} // namespace hydro
