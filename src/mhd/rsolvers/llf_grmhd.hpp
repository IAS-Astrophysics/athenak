#ifndef MHD_RSOLVERS_LLF_GRMHD_HPP_
#define MHD_RSOLVERS_LLF_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_grmhd.hpp
//! \brief LLF Riemann solver for general relativistic MHD.  Per-face implementation for
//! the split-kernel path.

#include "coordinates/cell_locations.hpp"
#include "llf_mhd_singlestate.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn LLF_GR<ivx>()
//! \brief The LLF Riemann solver for GR MHD, single face (m,k,j,i).
template <int ivx>
KOKKOS_INLINE_FUNCTION
void LLF_GR(const EOS_Data &eos, const RegionIndcs &indcs,
            const DualArray1D<RegionSize> &size, const CoordData &coord,
            const int m, const int k, const int j, const int i,
            const int is, const int js, const int ks,
            const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
            const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
            const DvceArray4D<Real> &bx,
            const DvceArray5D<Real> &flx,
            const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;


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

  // Extract left/right primitives.  Note 1/2/3 always refers to x1/2/3 dirs
  MHDPrim1D wli,wri;
  wli.d  = wl(m,IDN,k,j,i);
  wli.vx = wl(m,ivx,k,j,i);
  wli.vy = wl(m,ivy,k,j,i);
  wli.vz = wl(m,ivz,k,j,i);
  wli.by = bl(m,iby,k,j,i);
  wli.bz = bl(m,ibz,k,j,i);

  wri.d  = wr(m,IDN,k,j,i);
  wri.vx = wr(m,ivx,k,j,i);
  wri.vy = wr(m,ivy,k,j,i);
  wri.vz = wr(m,ivz,k,j,i);
  wri.by = br(m,iby,k,j,i);
  wri.bz = br(m,ibz,k,j,i);

  wli.e = wl(m,IEN,k,j,i);
  wri.e = wr(m,IEN,k,j,i);

  // Extract normal magnetic field
  Real bxi = bx(m,k,j,i);

  // Call LLF solver on single interface state
  MHDCons1D flux;
  SingleStateLLF_GRMHD(wli, wri, bxi, x1v, x2v, x3v, ivx, coord, eos, flux);

  // Store results in 3D array of fluxes
  flx(m,IDN,k,j,i) = flux.d;
  flx(m,ivx,k,j,i) = flux.mx;
  flx(m,ivy,k,j,i) = flux.my;
  flx(m,ivz,k,j,i) = flux.mz;
  flx(m,IEN,k,j,i) = flux.e;
  ey(m,k,j,i) = flux.by;
  ez(m,k,j,i) = flux.bz;
}
} // namespace mhd
#endif // MHD_RSOLVERS_LLF_GRMHD_HPP_
