#ifndef UTILS_TOV_TOV_POLYTROPE_HPP_
#define UTILS_TOV_TOV_POLYTROPE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov_polytrope.hpp
//  \brief Fixed polytrope EOS for use with TOVStar

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tov_utils.hpp"

namespace tov {

class PolytropeEOS {
 private:
  Real kappa;
  Real gamma;

 public:
  explicit PolytropeEOS(ParameterInput* pin) {
    kappa = pin->GetReal("problem", "kappa");
    gamma = pin->GetReal("mhd", "gamma");
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    return kappa*Kokkos::pow(rho, gamma);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    return Kokkos::pow(P/kappa, 1.0/gamma);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    return rho + kappa*Kokkos::pow(rho, gamma)/(gamma - 1.0);
  }
};

} // namespace tov

#endif // UTILS_TOV_TOV_POLYTROPE_HPP_
