#ifndef MHD_RSOLVERS_LLF_MHD_HPP_
#define MHD_RSOLVERS_LLF_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_mhd.hpp
//! \brief Local Lax Friedrichs (LLF) Riemann solver for non-relativistic MHD.  Per-face
//! implementation for the split-kernel path.

#include "llf_mhd_singlestate.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn LLF<ivx>()
//! \brief The LLF Riemann solver for MHD (both ideal gas and isothermal), single face.
//!
//! Indexing convention for wl/wr/bl/br: face-indexed in the face-normal axis,
//! cell-indexed in the transverse axes, all origin-shifted by ks/js/is.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void LLF(const EOS_Data &eos,
         const int m, const int k, const int j, const int i,
         const int is, const int js, const int ks,
         const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
         const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
         const DvceArray4D<Real> &bx,
         const DvceArray5D<Real> &flx,
         const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez) {
  constexpr int ivy = IVX + ((ivx-IVX) + 1)%3;
  constexpr int ivz = IVX + ((ivx-IVX) + 2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;


  // Extract left/right primitives
  MHDPrim1D wli, wri;
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

  if (eos.is_ideal) {
    wli.e = wl(m,IEN,k,j,i);
    wri.e = wr(m,IEN,k,j,i);
  }

  // Extract normal magnetic field
  Real bxi = bx(m,k,j,i);

  // Call LLF solver on single interface state
  MHDCons1D flux;
  SingleStateLLF_MHD(wli,wri,bxi,eos,flux);

  // Store results in 3D array of fluxes
  flx(m,IDN,k,j,i) = flux.d;
  flx(m,ivx,k,j,i) = flux.mx;
  flx(m,ivy,k,j,i) = flux.my;
  flx(m,ivz,k,j,i) = flux.mz;
  if (eos.is_ideal) {flx(m,IEN,k,j,i) = flux.e;}
  ey(m,k,j,i) = flux.by;
  ez(m,k,j,i) = flux.bz;
}
} // namespace mhd
#endif // MHD_RSOLVERS_LLF_MHD_HPP_
