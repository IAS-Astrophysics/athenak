#ifndef EOS_PRIMITIVE_SOLVER_POLYTROPE_HPP_
#define EOS_PRIMITIVE_SOLVER_POLYTROPE_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file polytrope.hpp
//  \brief An isothermal polytropic equation of state.

#include <math.h>
#include <stdio.h>
#include <float.h>

#include <limits>
#include <string>

#include "ps_types.hpp"
#include "eos_policy_interface.hpp"
#include "unit_system.hpp"

namespace Primitive {

class Polytrope : public EOSPolicyInterface {
 private:
  /// Parameters for the EOS
  Real K;
  Real gamma;
  Real igammam1;
 
 protected:
  /// Constructor
  Polytrope() {
    K = 100.;
    gamma = 2.;
    igammam1 = 1.0/(gamma - 1.0);
    n_species = 0;
    mb = 1.0;

    min_n = 0.0;
    max_n = std::numeric_limits<Real>::max();
    min_T = 0.0;
    max_T = std::numeric_limits<Real>::max();
    for (int i = 0; i < MAX_SPECIES; i++) {
      min_Y[i] = 0.0;
      max_Y[i] = 1.0;
    }
    eos_units = MakeNuclear();
  }

  /// Being an isothermal polytrope, the temperature is ill-defined. We just return 0.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    return 0.;
  }
  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    return 0.;
  }

  /// Calculate the energy density using the polytrope.
  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    return n*mb + K*igammam1*pow(n*mb, gamma);
  }

  /// Calculate the pressure using the polytrope.
  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    return K*igammam1*pow(n*mb, gamma);
  }

  /// Calculate the entropy per baryon using the polytrope.
  KOKKOS_INLINE_FUNCTION Real Entropy(Real n, Real T, Real *Y) const {
    return 0;
  }

  /// Calculate the enthalpy per baryon using the polytrope.
  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    return mb + mb*K*gamma*igammam1*pow(mb*n, gamma - 1.);
  }

  /// Get the minimum enthalpy per baryon using the polytrope.
  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const {
    return mb;
  }

  /// Calculate the sound speed for a polytrope.
  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    return sqrt(gamma*K*pow(mb*n, gamma - 1.));
  }

  /// Calculate the internal energy per mass.
  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T, Real *Y) const {
    return gamma*igammam1*K*pow(mb*n, gamma - 1.);
  }

  /// Calculate the minimum pressure at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return Pressure(n, 0., Y);
  }

  /// Calculat the maximum pressure at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    return Pressure(n, 0., Y);
  }

  /// Calculate the minimum energy at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    return Energy(n, 0., Y);
  }

  /// Calculate the maximum energy at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    return Energy(n, 0., Y);
  }

 public:
  /// Set the adiabatic index for the ideal gas.
  /// The range \f$1 < \gamma < 1\f$ is imposed. The lower
  /// constraint ensures that enthalpy is finite, and the upper
  /// bound keeps the sound speed causal.
  KOKKOS_INLINE_FUNCTION void SetGamma(Real g) {
    gamma = (g <= 1.0) ? 1.00001 : ((g >= 2.0) ? 2.00001 : g);
    igammam1 = 1.0/(gamma - 1.0);
  }

  /// Get the adiabatic index.
  KOKKOS_INLINE_FUNCTION Real GetGamma() const {
    return gamma;
  }

  /// Set the entropy constant for a simple polytrope
  KOKKOS_INLINE_FUNCTION void SetKappa(Real k) {
    K = k;
  }

  /// Get the entropy constant for a simple polytrope.
  KOKKOS_INLINE_FUNCTION Real GetKappa() const {
    return K;
  }

  /// Set the number of species. Throw an exception if
  /// the number of species is invalid.
  KOKKOS_INLINE_FUNCTION void SetNSpecies(int n) {
    if (n > MAX_SPECIES || n < 0) {
      // FIXME: We need to abort here.
      return;
    }
    n_species = n;
  }

  /// Set the EOS unit system.
  KOKKOS_INLINE_FUNCTION void SetEOSUnitSystem(UnitSystem units) {
    eos_units = units;
  }

};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_POLYTROPE_HPP_
