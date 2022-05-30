#ifndef MHD_RSOLVERS_LLF_GRMHD_HPP_
#define MHD_RSOLVERS_LLF_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_grmhd.hpp
//! \brief LLF Riemann solver for general relativistic MHD.

#include "coordinates/cell_locations.hpp"
#include "llf_mhd_singlestate.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void LLF_GR
//! \brief The LLF Riemann solver for GR MHD

KOKKOS_INLINE_FUNCTION
void LLF_GR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  // Cyclic permutation of array indices
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;

  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  par_for_inner(member, il, iu, [&](const int i) {
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
    wli.d  = wl(IDN,i);
    wli.vx = wl(IVX,i);
    wli.vy = wl(IVY,i);
    wli.vz = wl(IVZ,i);
    wli.by = bl(iby,i);
    wli.bz = bl(ibz,i);

    wri.d  = wr(IDN,i);
    wri.vx = wr(IVX,i);
    wri.vy = wr(IVY,i);
    wri.vz = wr(IVZ,i);
    wri.by = br(iby,i);
    wri.bz = br(ibz,i);

    wli.e = wl(IEN,i);
    wri.e = wr(IEN,i);

    // Extract normal magnetic field
    Real &bxi = bx(m,k,j,i);

    // Call LLF solver on single interface state
    MHDCons1D flux;
    SingleStateLLF_GRMHD(wli, wri, bxi, x1v, x2v, x3v, ivx, coord, eos, flux);

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,IEN,k,j,i) = flux.e;
    flx(m,IVX,k,j,i) = flux.mx;
    flx(m,IVY,k,j,i) = flux.my;
    flx(m,IVZ,k,j,i) = flux.mz;
    ey(m,k,j,i) = flux.by;
    ez(m,k,j,i) = flux.bz;
  });

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_LLF_GRMHD_HPP_
