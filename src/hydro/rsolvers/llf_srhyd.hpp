#ifndef HYDRO_RSOLVERS_LLF_SRHYD_HPP_
#define HYDRO_RSOLVERS_LLF_SRHYD_HPP_
//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srhyd.hpp
//! \brief Local Lax Friedrichs (LLF) Riemann solver for special relativistic hydro.
//! Per-face implementation for the split-kernel path: reads the L/R primitives from the
//! global per-face buffers and writes a single flux entry.

#include "llf_hyd_singlestate.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn LLF_SR<ivx>()
//! \brief Compute the SR LLF flux at face (m,k,j,i) for direction ivx.  Buffer indexing
//! for wl/wr: face-indexed in the face-normal axis, cell-indexed in the transverse axes,
//! all origin-shifted by ks/js/is.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void LLF_SR(const EOS_Data &eos,
            const int m, const int k, const int j, const int i,
            const int is, const int js, const int ks,
            const DvceArray5D<Real> &wl,
            const DvceArray5D<Real> &wr,
            const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;


  // Extract left/right primitives
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
  SingleStateLLF_SRHyd(wli, wri, eos, flux);

  // Store results into 3D array of fluxes
  flx(m,IDN,k,j,i) = flux.d;
  flx(m,ivx,k,j,i) = flux.mx;
  flx(m,ivy,k,j,i) = flux.my;
  flx(m,ivz,k,j,i) = flux.mz;
  flx(m,IEN,k,j,i) = flux.e;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_LLF_SRHYD_HPP_
