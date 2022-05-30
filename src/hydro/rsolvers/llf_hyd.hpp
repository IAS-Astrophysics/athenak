#ifndef HYDRO_RSOLVERS_LLF_HYD_HPP_
#define HYDRO_RSOLVERS_LLF_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_hyd.hpp
//! \brief Local Lax Friedrichs (LLF) Riemann solver for hydrodynamics

#include "llf_hyd_singlestate.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void LLF
//! \brief Wrapper function for the LLF Riemann solver for hydrodynamics (both ideal gas
//! and isothermal) which calls single state LLF solver.

KOKKOS_INLINE_FUNCTION
void LLF(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  par_for_inner(member, il, iu, [&](const int i) {
    // Extract left/right primitives
    HydPrim1D wli, wri;
    wli.d  = wl(IDN,i);
    wli.vx = wl(ivx,i);
    wli.vy = wl(ivy,i);
    wli.vz = wl(ivz,i);

    wri.d  = wr(IDN,i);
    wri.vx = wr(ivx,i);
    wri.vy = wr(ivy,i);
    wri.vz = wr(ivz,i);

    if (eos.is_ideal) {
      wli.e = wl(IEN,i);
      wri.e = wr(IEN,i);
    }

    // Call LLF solver on single interface state
    HydCons1D flux;
    SingleStateLLF_Hyd(wli,wri,eos,flux);

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,ivx,k,j,i) = flux.mx;
    flx(m,ivy,k,j,i) = flux.my;
    flx(m,ivz,k,j,i) = flux.mz;
    if (eos.is_ideal) {flx(m,IEN,k,j,i) = flux.e;}
  });

  return;
}
} // namespace hydro
#endif // HYDRO_RSOLVERS_LLF_HYD_HPP_
