#ifndef HYDRO_RSOLVERS_LLF_GRHYD_HPP_
#define HYDRO_RSOLVERS_LLF_GRHYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_grhyd.hpp
//! \brief LLF Riemann solver for general relativistic hydrodynamics.  Per-face
//! implementation for the split-kernel path: reads the L/R primitives from the global
//! per-face buffers, computes the face coordinate/metric, and writes a single flux entry.

#include "coordinates/cell_locations.hpp"
#include "llf_hyd_singlestate.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn LLF_GR<ivx>()
//! \brief Compute the GR LLF flux at face (m,k,j,i) for direction ivx.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void LLF_GR(const EOS_Data &eos, const RegionIndcs &indcs,
            const DualArray1D<RegionSize> &size, const CoordData &coord,
            const int m, const int k, const int j, const int i,
            const int is, const int js, const int ks,
            const DvceArray5D<Real> &wl,
            const DvceArray5D<Real> &wr,
            const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;


  // Extract position of interface
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

  // Extract left/right primitives.
  HydPrim1D wli, wri;
  wli.d  = wl(m,IDN,k,j,i);
  wli.vx = wl(m,ivx,k,j,i);
  wli.vy = wl(m,ivy,k,j,i);
  wli.vz = wl(m,ivz,k,j,i);
  wli.e  = wl(m,IEN,k,j,i);

  wri.d  = wr(m,IDN,k,j,i);
  wri.vx = wr(m,ivx,k,j,i);
  wri.vy = wr(m,ivy,k,j,i);
  wri.vz = wr(m,ivz,k,j,i);
  wri.e  = wr(m,IEN,k,j,i);

  // Call LLF solver on single interface state
  HydCons1D flux;
  SingleStateLLF_GRHyd(wli, wri, x1v, x2v, x3v, ivx, coord, eos, flux);

  // Store results in 3D array of fluxes
  flx(m,IDN,k,j,i) = flux.d;
  flx(m,ivx,k,j,i) = flux.mx;
  flx(m,ivy,k,j,i) = flux.my;
  flx(m,ivz,k,j,i) = flux.mz;
  flx(m,IEN,k,j,i) = flux.e;
}
} // namespace hydro
#endif // HYDRO_RSOLVERS_LLF_GRHYD_HPP_
