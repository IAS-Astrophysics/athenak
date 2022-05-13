#ifndef HYDRO_RSOLVERS_LLF_GRHYD_HPP_
#define HYDRO_RSOLVERS_LLF_GRHYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_grhyd.hpp
//! \brief LLF Riemann solver for general relativistic hydrodynamics.

#include "coordinates/cell_locations.hpp"
#include "llf_hyd_singlestate.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void LLF_GR
//! \brief The LLF Riemann solver for GR hydrodynamics

KOKKOS_INLINE_FUNCTION
void LLF_GR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx) {
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  par_for_inner(member, il, iu, [&](const int i) {
    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x1v,x2v,x3v;
    if (ivx == IVX) {
      x1v = LeftEdgeX  (i-is, indcs.nx1, x1min, x1max);
      x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    } else if (ivx == IVY) {
      x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      x2v = LeftEdgeX  (j-js, indcs.nx2, x2min, x2max);
      x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    } else {
      x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      x3v = LeftEdgeX  (k-ks, indcs.nx3, x3min, x3max);
    }

    // Extract left/right primitives.  Note 1/2/3 always refers to x1/2/3 dirs
    HydPrim1D wli,wri;
    wli.d  = wl(IDN,i);
    wli.vx = wl(IVX,i);
    wli.vy = wl(IVY,i);
    wli.vz = wl(IVZ,i);

    wri.d  = wr(IDN,i);
    wri.vx = wr(IVX,i);
    wri.vy = wr(IVY,i);
    wri.vz = wr(IVZ,i);

    wli.p = eos.IdealGasPressure(wl(IEN,i));
    wri.p = eos.IdealGasPressure(wr(IEN,i));

    // Call LLF solver on single interface state
    HydCons1D flux;
    SingleStateLLF_GRHyd(wli, wri, x1v, x2v, x3v, ivx, coord, eos, flux);

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,IEN,k,j,i) = flux.e;
    flx(m,IVX,k,j,i) = flux.mx;
    flx(m,IVY,k,j,i) = flux.my;
    flx(m,IVZ,k,j,i) = flux.mz;
  });

  return;
}
} // namespace hydro
#endif // HYDRO_RSOLVERS_LLF_GRHYD_HPP_
