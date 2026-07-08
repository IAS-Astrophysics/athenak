#ifndef MHD_RSOLVERS_DUAL_EINT_MHD_HPP_
#define MHD_RSOLVERS_DUAL_EINT_MHD_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dual_eint_mhd.hpp
//! \brief auxiliary internal-energy flux helper for Newtonian MHD

namespace mhd {

KOKKOS_INLINE_FUNCTION
Real DualEnergyInternalEnergyFloor(const EOS_Data &eos, const Real dens_in) {
  const Real dens = fmax(dens_in, eos.dfloor);
  Real eint_floor = eos.pfloor/(eos.gamma - 1.0);
  if (eos.tfloor > 0.0) {
    eint_floor = fmax(eint_floor, dens*eos.tfloor/(eos.gamma - 1.0));
  }
  if (eos.sfloor > 0.0) {
    eint_floor = fmax(eint_floor, dens*eos.sfloor*pow(dens, eos.gamma - 1.0)/
                                  (eos.gamma - 1.0));
  }
  return eint_floor;
}

KOKKOS_INLINE_FUNCTION
void FloorDualEnergyFaceStates(TeamMember_t const &member, const EOS_Data &eos,
                               const int dual_idx, const int il, const int iu,
                               ScrArray2D<Real> &wl, ScrArray2D<Real> &wr) {
  par_for_inner(member, il, iu, [&](const int i) {
    wl(dual_idx, i) = fmax(wl(dual_idx, i),
                           DualEnergyInternalEnergyFloor(eos, wl(IDN, i)));
    wr(dual_idx, i) = fmax(wr(dual_idx, i),
                           DualEnergyInternalEnergyFloor(eos, wr(IDN, i)));
  });
}

KOKKOS_INLINE_FUNCTION
void UpwindDualEnergyFlux(TeamMember_t const &member, const EOS_Data &eos,
                          const int dual_idx, const int m, const int k, const int j,
                          const int il, const int iu, const ScrArray2D<Real> &wl,
                          const ScrArray2D<Real> &wr, DvceArray5D<Real> flx,
                          DvceArray5D<Real> vf) {
  par_for_inner(member, il, iu, [&](const int i) {
    const Real mass_flux = flx(m, IDN, k, j, i);
    if (mass_flux > 0.0) {
      const Real dens = fmax(wl(IDN, i), eos.dfloor);
      flx(m, dual_idx, k, j, i) = mass_flux*(wl(dual_idx, i)/dens);
      vf(m, 0, k, j, i) = mass_flux/dens;
    } else if (mass_flux < 0.0) {
      const Real dens = fmax(wr(IDN, i), eos.dfloor);
      flx(m, dual_idx, k, j, i) = mass_flux*(wr(dual_idx, i)/dens);
      vf(m, 0, k, j, i) = mass_flux/dens;
    } else {
      flx(m, dual_idx, k, j, i) = 0.0;
      vf(m, 0, k, j, i) = 0.0;
    }
  });
}

} // namespace mhd

#endif // MHD_RSOLVERS_DUAL_EINT_MHD_HPP_
