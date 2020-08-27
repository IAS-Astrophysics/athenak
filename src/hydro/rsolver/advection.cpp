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
  
Advection::Advection(Mesh* pm, ParameterInput* pin, int igid) : 
  RiemannSolver(pm, pin, igid)
{
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
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  if (pmb->phydro->hydro_eos == HydroEOS::adiabatic) {
    gm1 = pmb->phydro->peos->GetGamma() - 1.0;
  }
  if (pmb->phydro->hydro_eos == HydroEOS::isothermal) {
    iso_cs = pmb->phydro->peos->SoundSpeed(wli);  // wli is just "dummy argument"
  }

  for (int i=il; i<=iu; ++i) {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    if (pmb->phydro->hydro_eos == HydroEOS::adiabatic) { wli[IPR]=wl(IPR,i); }

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    if (pmb->phydro->hydro_eos == HydroEOS::adiabatic) { wri[IPR]=wr(IPR,i); }

    //--- Step 3.  Compute upwind fluxes

    if (wli[IVX] >= 0.0) {

      Real mxl = wli[IDN]*wli[IVX];
      flx(IDN,i) = mxl;
      flx(ivx,i) = mxl*wli[IVX];
      flx(ivy,i) = mxl*wli[IVY];
      flx(ivz,i) = mxl*wli[IVZ];
      if (pmb->phydro->hydro_eos == HydroEOS::adiabatic) {
        flx(IEN,i) = wli[IPR]*wli[IVX]/gm1;
      }

    } else {

      Real mxr = wri[IDN]*wri[IVX];
      flx(IDN,i) = mxr;
      flx(ivx,i) = mxr*wri[IVX];
      flx(ivy,i) = mxr*wri[IVY];
      flx(ivz,i) = mxr*wri[IVZ];
      if (pmb->phydro->hydro_eos == HydroEOS::adiabatic) {
        flx(IEN,i) = wri[IPR]*wri[IVX]/gm1;
      }

    }
  }

  return;
}

} // namespace hydro
