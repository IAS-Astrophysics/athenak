//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advection.cpp
//  \brief Riemann solver for pure advection problems (v = constant).  Simply computes the
//  upwind flux of each vriable.

#include <algorithm>  // max(), min()

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "hydro/eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "hydro/rsolver/rsolver.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// Advection constructor
  
Advection::Advection(Hydro *phyd, ParameterInput *pin) 
  : RiemannSolver(phyd, pin) {

  void RSolver(const int il, const  int iu, const int dir,
    const AthenaArray<Real> &wl, const AthenaArray<Real> &wr, AthenaArray<Real> &flx);
  
}

//----------------------------------------------------------------------------------------
//! \fn void Advection::RSolver
//  \brief An advection Riemann solver for hydrodynamics (both adiabatic and isothermal)

void Advection::RSolver(const int il, const int iu, const int ivx,
      const AthenaArray<Real> &wl, const AthenaArray<Real> &wr, AthenaArray<Real> &flx) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5];
  Real gm1, iso_cs;
  if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) {
    gm1 = pmy_hydro->peos->GetGamma() - 1.0;
  }
  if (pmy_hydro->hydro_eos == HydroEOS::isothermal) {
    iso_cs = pmy_hydro->peos->SoundSpeed(wli);  // wli is just "dummy argument"
  }

  for (int i=il; i<=iu; ++i) {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) { wli[IPR]=wl(IPR,i); }

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) { wri[IPR]=wr(IPR,i); }

    //--- Step 3.  Compute upwind fluxes

    if (wli[IVX] >= 0.0) {

      Real mxl = wli[IDN]*wli[IVX];
      flx(IDN,i) = mxl;
      flx(IVX,i) = mxl*wli[IVX];
      flx(IVY,i) = mxl*wli[IVY];
      flx(IVZ,i) = mxl*wli[IVZ];
      if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) {
        flx(IEN,i) = wli[IPR]*wli[IVX]/gm1;
      }

    } else {

      Real mxr = wri[IDN]*wri[IVX];
      flx(IDN,i) = mxr;
      flx(IVX,i) = mxr*wri[IVX];
      flx(IVY,i) = mxr*wri[IVY];
      flx(IVZ,i) = mxr*wri[IVZ];
      if (pmy_hydro->hydro_eos == HydroEOS::adiabatic) {
        flx(IEN,i) = wri[IPR]*wri[IVX]/gm1;
      }

    }
  }

  return;
}

} // namespace hydro
