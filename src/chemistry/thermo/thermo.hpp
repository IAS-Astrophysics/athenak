#ifndef CHEMISTRY_THERMO_THERMO_HPP_
#define CHEMISTRY_THERMO_THERMO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file thermo.hpp
//  \brief Definitions for heating and cooling processes

#include "athena.hpp"
#include "units/units.hpp"

namespace chemistry {
/*!
 * \brief Stores the thermodynamic properties for chemistry.
 *
 */
struct Thermo {
  Thermo() {}
  ~Thermo() = default;

  // ----- Physical Constants -----
  static constexpr Real alpha_GD_ = 3.2e-34;  // DESPOTIC

  //! \fn Real Thermo::CvCold(const Real xH2, const Real xHe_total, const Real
  //! xe)
  //! \brief This computes the specific heat (C_v) of a cold gas, i.e. assuming
  //! that H2 rotational and vibrational levels not excited.
  //!
  //! xH2, xe = nH2 or ne / nH
  //! xHe_total = xHeI + xHeII = 0.1 for solar value.
  //! Return: specific heat per H atom.
  KOKKOS_FUNCTION static Real CvCold(const Real xH2, const Real xHe_total,
                                     const Real xe) {
    const Real xH20 = (xH2 > 0.5) ? 0.5 : xH2;
    return 1.5 * units::Units::k_boltzmann_cgs *
           ((1. - 2. * xH20) + xH20 + xHe_total + xe);
  }
};
}  // namespace chemistry

#endif  // CHEMISTRY_THERMO_THERMO_HPP_
