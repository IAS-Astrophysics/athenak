#ifndef MHD_RSOLVERS_LLF_SRMHD_HPP_
#define MHD_RSOLVERS_LLF_SRMHD_HPP_
//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srmhd.hpp
//! \brief Local Lax-Friedrichs (LLF) Riemann solver for special relativistic MHD.

#include "llf_mhd_singlestate.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void LLF
//! \brief The LLF Riemann solver for SR MHD

KOKKOS_INLINE_FUNCTION
void LLF_SR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;

  par_for_inner(member, il, iu, [&](const int i) {
    // Extract left/right primitives
    MHDPrim1D wli,wri;
    wli.d  = wl(IDN,i);
    wli.vx = wl(ivx,i);
    wli.vy = wl(ivy,i);
    wli.vz = wl(ivz,i);
    wli.by = bl(iby,i);
    wli.bz = bl(ibz,i);

    wri.d  = wr(IDN,i);
    wri.vx = wr(ivx,i);
    wri.vy = wr(ivy,i);
    wri.vz = wr(ivz,i);
    wri.by = br(iby,i);
    wri.bz = br(ibz,i);

    wli.e = wl(IEN,i);
    wri.e = wr(IEN,i);

    // Extract normal magnetic field
    Real &bxi = bx(m,k,j,i);

    // Call LLF solver on single interface state
    MHDCons1D flux;
    SingleStateLLF_SRMHD(wli,wri,bxi,eos,flux);

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,IEN,k,j,i) = flux.e;
    flx(m,ivx,k,j,i) = flux.mx;
    flx(m,ivy,k,j,i) = flux.my;
    flx(m,ivz,k,j,i) = flux.mz;
    ey(m,k,j,i) = flux.by;
    ez(m,k,j,i) = flux.bz;
  });

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_LLF_SRMHD_HPP_
