#ifndef UTILS_TOV_TOV_PIECEWISE_POLY_HPP_
#define UTILS_TOV_TOV_PIECEWISE_POLY_HPP_
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
#include "eos/primitive-solver/piecewise_polytrope.hpp"

namespace tov {

class PiecewisePolytropeEOS: public Primitive::PiecewisePolytrope {
 public:
  explicit PiecewisePolytropeEOS(ParameterInput *pin) {
    ReadParametersFromInput("mhd", pin);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    Real nb = rho/mb;
    int p = FindPiece(nb);
    return GetColdPressure(nb, p);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    Real rhob = GetDensityFromColdPressure(P);
    return rhob;
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    Real nb = rho/mb;
    int p = FindPiece(nb);
    return GetColdEnergy(nb, p);
  }
};

} // namespace tov

#endif  // UTILS_TOV_TOV_PIECEWISE_POLY_HPP_
