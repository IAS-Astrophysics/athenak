#ifndef EOS_PRIMITIVE_SOLVER_IDEALGAS_HPP_
#define EOS_PRIMITIVE_SOLVER_IDEALGAS_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file idealgas.hpp
//  \brief Defines an ideal gas equation of state.

#include <math.h>
#include <stdexcept>
#include <limits>

#include "ps_types.hpp"
#include "eos_policy_interface.hpp"
#include "unit_system.hpp"

namespace Primitive {

class IdealGas : public EOSPolicyInterface {
 protected:
  /// Adiabatic index
  Real gamma;
  Real gammam1;

  /// Constructor
  IdealGas() {
    gamma = 5.0/3.0;
    gammam1 = gamma - 1.0;
    mb = 1.0;

    min_n = 0.0;
    max_n = std::numeric_limits<Real>::max();
    min_T = 0.0;
    max_T = std::numeric_limits<Real>::max();
    n_species = 0;
    for (int i = 0; i < MAX_SPECIES; i++) {
      min_Y[i] = 0.0;
      max_Y[i] = 1.0;
    }

    //eos_units = &Nuclear;
    eos_units = MakeNuclear();
  }

  /*~IdealGas() {
    delete eos_units;
  }*/

  /// Calculate the temperature using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    return gammam1*(e - mb*n)/n;
  }

  /// Calculate the temperature using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    return p/n;
  }

  /// Calculate the energy density using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    return n*(mb + T/gammam1);
  }

  /// Calculate the pressure using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    return n*T;
  }

  /// Calculate the entropy per baryon using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real Entropy(Real n, Real T, Real *Y) const {
    // FIXME: implement or force to abort
    return 0;
  }

  /// Calculate the enthalpy per baryon using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    return mb + gamma/gammam1*T;
  }

  /// Get the minimum enthalpy per baryon according to the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const {
    return mb;
  }

  /// Calculate the sound speed for an ideal gas.
  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    return sqrt(gamma*gammam1*T/(gammam1*mb + gamma*T));
  }

  /// Calculate the internal energy per mass
  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T, Real *Y) const {
    return T/(mb*gammam1);
  }

  /// Calculate the minimum pressure at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return 0.0;
  }

  /// Calculate the maximum pressure at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    // Note that max_T is already set to numeric_limits<Real>::max!
    return max_T;
  }

  /// Calculate the minimum energy density at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    return n*mb;
  }

  /// Calculate the maximum energy density at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    // Note that max_T is already set to numeric_limits<Real>::max!
    return max_T;
  }

 public:
  /// Set the adiabatic index for the ideal gas.
  /// The range \f$1 < \gamma < 1\f$ is imposed. The lower
  /// constraint ensures that enthalpy is finite, and the upper
  /// bound keeps the sound speed causal.
  KOKKOS_INLINE_FUNCTION void SetGamma(Real g) {
    gamma = (g <= 1.0) ? 1.00001 : ((g >= 2.0) ? 2.00001 : g);
    gammam1 = gamma - 1.0;
  }

  /// Get the adiabatic index.
  KOKKOS_INLINE_FUNCTION Real GetGamma() const {
    return gamma;
  }

  /// Set the baryon mass
  KOKKOS_INLINE_FUNCTION void SetBaryonMass(Real m) {
    mb = m;
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

#endif  // EOS_PRIMITIVE_SOLVER_IDEALGAS_HPP_
