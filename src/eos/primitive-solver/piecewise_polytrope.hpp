#ifndef EOS_PRIMITIVE_SOLVER_PIECEWISE_POLYTROPE_HPP_
#define EOS_PRIMITIVE_SOLVER_PIECEWISE_POLYTROPE_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file piecewise_polytrope.hpp
//  \brief Defines a piecewise-polytropic equation of state.
//
//  Each individual piece satisfies the form
//  \f$P_\textrm{cold} = P_i \frac{\rho}{\rho_i}^{\gamma_i}\f$,
//  for some density \f$\rho > \rho_i\f$. There is an additional
//  finite-temperature portion added on top using the ideal gas
//  law:
//  \f$P_\textrm{therm} = nk_B T\f$

#include <math.h>
#include <stdio.h>
#include <float.h>

#include <stdexcept>
#include <limits>
#include <string>

#include "ps_types.hpp"
#include "eos_policy_interface.hpp"
#include "unit_system.hpp"

namespace Primitive {

#define MAX_PIECES 7

class PiecewisePolytrope : public EOSPolicyInterface {
 private:
  /// Number of polytropes in the EOS
  int n_pieces;

  /// Parameters for the EOS
  Real density_pieces[MAX_PIECES+1];
  Real a_pieces[MAX_PIECES+1];
  Real gamma_pieces[MAX_PIECES+1];
  Real pressure_pieces[MAX_PIECES+1];
  Real gamma_thermal;
  bool initialized;

 protected:
  /// Constructor
  PiecewisePolytrope() {
    n_pieces = 0;
    initialized = false;
    n_species = 0;
    gamma_thermal = 5.0/3.0;
    for (int i = 0; i < MAX_SPECIES; i++) {
      min_Y[i] = 0.0;
      max_Y[i] = 1.0;
    }
    eos_units = MakeNuclear();
  }

  /// Destructor
  //~PiecewisePolytrope();

  /// Calculate the temperature using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    int p = FindPiece(n);
    Real e_cold = GetColdEnergy(n, p);
    return (e - e_cold)*(gamma_thermal - 1.0)/n;
  }

  /// Calculate the temperature using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    int i = FindPiece(n);
    Real p_cold = GetColdPressure(n, i);
    return (p - p_cold)/n;
  }

  /// Calculate the energy density using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    int p = FindPiece(n);

    return GetColdEnergy(n, p) + n*T/(gamma_thermal - 1.0);
  }

  /// Calculate the pressure using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    int p = FindPiece(n);

    return GetColdPressure(n, p) + n*T;
  }

  /// Calculate the entropy per baryon using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real Entropy(Real n, Real T, Real *Y) const {
    // FIXME: Error message or abort
    return 0.;
  }

  /// Calculate the enthalpy per baryon using the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    int p = FindPiece(n);
    return (GetColdEnergy(n, p) + GetColdPressure(n, p))/n +
           gamma_thermal/(gamma_thermal - 1.0)*T;
  }

  /// Get the minimum enthalpy per baryon according to the ideal gas law.
  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const {
    return mb;
  }

  /// Calculate the sound speed for an ideal gas.
  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    int p = FindPiece(n);
    Real rho = n*mb;

    Real h_cold = (GetColdEnergy(n, p) + GetColdPressure(n, p))/rho;
    Real h_th = gamma_thermal/(gamma_thermal - 1.0)*T/mb;

    Real P_cold = GetColdPressure(n, p);

    Real csq_cold_w = gamma_pieces[p]*P_cold/rho;
    Real csq_th_w = (gamma_thermal - 1.0)*h_th;

    return sqrt((csq_cold_w + csq_th_w)/(h_th + h_cold));
  }

  /// Calculate the internal energy per mass.
  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T, Real *Y) const {
    int p = FindPiece(n);
    Real eps_cold = GetColdEnergy(n, p)/(n*mb) - 1.0;
    return eps_cold + T/(mb*(gamma_thermal - 1.0));
  }

  /// Calculate the minimum pressure at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    int p = FindPiece(n);
    return GetColdPressure(n, p);
  }

  /// Calculate the maximum pressure at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    // Note that max_T is already set to numeric_limits<Real>::max()!
    return max_T;
  }

  /// Calculate the minimum energy at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    int p = FindPiece(n);

    return GetColdEnergy(n, p);
  }

  /// Calculate the maximum energy at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    // Note that max_T is already set to numeric_limits<Real>::max()!
    return max_T;
  }

 public:
  /// Load the EOS parameters from a file.
  KOKKOS_INLINE_FUNCTION bool ReadParametersFromFile(std::string fname) {
    // TODO(JF): Add this functionality
    return false;
  }

  //! \brief Initialize PiecewisePolytrope from data.
  //
  //  \param[in] densities The dividing densities
  //  \param[in] gammas    The adiabatic index for each polytrope
  //  \param[in] rho_min   The minimum density for the EOS
  //  \param[in] P0        The pressure at the first polytrope division
  //  \param[in] m         The baryon mass
  //  \param[in] n         The number of pieces in the EOS
  KOKKOS_INLINE_FUNCTION bool InitializeFromData(Real *densities, Real *gammas,
                          Real rho_min, Real P0, Real m, int n) {
    // Make sure that we actually *have* polytropes
    if (n <= 0) {
      printf("PiecewisePolytrope: Invalid number of polytropes requested."); // NOLINT
      return false;
    }
    // Before we even try to construct anything, we need to make sure that
    // the densities are ordered properly.
    for (int i = 1; i < n; i++) {
      if(densities[i] <= densities[i-1]) {
        // The densities must be ordered from smallest to largest and strictly
        // increasing.
        printf("PiecewisePolytrope: Densities must be strictly increasing."); // NOLINT
        return false;
      }
    }
    if (rho_min >= densities[0]) {
      printf("PiecewisePolytrope: minimum density above lowest transition."); // NOLINT
      return false;
    }

    // Make sure that we're not trying to allocate too many polytropes
    if (n > MAX_PIECES) {
      printf("PiecewisePolytrope: number of pieces requested exceeds limit."); // NOLINT
      return false;
    }

    // Initialize (most of) the member variables
    n_pieces = n;
    mb = m;
    min_n = 0.0;
    max_n = densities[n-1]/mb;

    // Now we can construct the different pieces.
    density_pieces[0] = densities[0]/m;
    gamma_pieces[0] = gammas[0];
    pressure_pieces[0] = P0;
    if (n > 1) {
      a_pieces[0] = P0/densities[0]*(1.0/(gammas[0] - 1.0) - 1.0/(gammas[1] - 1.0));
    } else {
      a_pieces[0] = 0.0;
    }
    for (int i = 1; i < n; i++) {
      density_pieces[i] = densities[i]/m;
      gamma_pieces[i] = gammas[i];
      pressure_pieces[i] = pressure_pieces[i-1] *
                           pow(densities[i]/densities[i-1],gammas[i]);
      // Because we've rewritten the EOS in terms of temperature, we don't need
      // kappa in its current form. However, we can use it to define the a
      // constants that show up in our equations.
      a_pieces[i] = a_pieces[i-1] + pressure_pieces[i-1]/
                      densities[i-1]*(1.0/(gammas[i-1] - 1.0) - 1.0/(gammas[i] - 1.0));
      // Let's double-check that the density is physical.
      if (gamma_pieces[i] > 2.0) {
        Real rho_max = pow((gammas[i] - 1.0) *
                           (1.0 + a_pieces[i])/(gammas[i]*(gammas[i]-2.0) *
                           densities[i]/pressure_pieces[i]),1.0/(gammas[i]-1.0)) *
                       densities[i];
        if (densities[i] > rho_max) {
          printf("The i = %d" // NOLINT
                 " piece of the polytrope permits superluminal sound speeds: \n"
                 "  rho[i]     = %g\n"
                 "  rho_max[i] = %g\n",i,densities[i],rho_max);
          return false;
        }
      }
    }

    // Set up the default case. Because of thermodynamic constraints,
    // we must force a to be zero. Gamma is fixed by continuity.
    a_pieces[n] = 0.0;
    Real factor = (gammas[0] - 1.0)*rho_min*a_pieces[0];
    Real P_min = P0*pow(rho_min/densities[0],gammas[0]);
    gamma_pieces[n] = (gammas[0]*P_min + factor)/(P_min + factor);
    density_pieces[n] = rho_min/mb;
    pressure_pieces[n] = P_min;

    // Because we're adding in a finite-temperature component via the ideal gas,
    // the only restriction on our temperature is that it needs to be nonnegative.
    min_T = 0.0;
    max_T = DBL_MAX;

    initialized = true;
    return true;
  }

  /// Check if the EOS has been initialized properly.
  KOKKOS_INLINE_FUNCTION bool IsInitialized() const {
    return initialized;
  }

  /// Find out how many polytropes are in the EOS.
  KOKKOS_INLINE_FUNCTION int GetNPieces() const {
    return n_pieces;
  }

  /// Get the adiabatic constant for a particular density.
  KOKKOS_INLINE_FUNCTION Real GetGamma(Real n) const {
    return gamma_pieces[FindPiece(n)];
  }

  /// Set the adiabatic constant for the thermal part.
  KOKKOS_INLINE_FUNCTION void SetThermalGamma(Real g) {
    gamma_thermal = (g <= 1.0) ? 1.00001 : ((g >= 2.0) ? 2.00001 : g);
  }

  /// Get the adiabatic constant for the thermal part.
  KOKKOS_INLINE_FUNCTION Real GetThermalGamma() const {
    return gamma_thermal;
  }

  /// Find the index of the piece that the density aligns with.
  KOKKOS_INLINE_FUNCTION int FindPiece(Real n) const {
    // WARNING: assumes the EOS is initialized!
    // In case the density is below the minimum, we
    // implement a default case that is stored just
    // past the current polytrope.
    int polytrope = n_pieces-1;
    if (n < density_pieces[n_pieces]) {
      return n_pieces;
    }
    for (int i = 0; i < n_pieces; i++) {
      if (n <= density_pieces[i]) {
        polytrope = i;
        break;
      }
    }

    return polytrope;
  }

  /// Polytropic Energy Density
  KOKKOS_INLINE_FUNCTION Real GetColdEnergy(Real n, int p) const {
    return mb*n*(1.0 + a_pieces[p]) + GetColdPressure(n, p)/(gamma_pieces[p] - 1.0);
  }

  /// Polytropic Pressure
  KOKKOS_INLINE_FUNCTION Real GetColdPressure(Real n, int p) const {
    return pressure_pieces[p]*pow((n/density_pieces[p]), gamma_pieces[p]);
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

  /// Set the EOS unit system
  KOKKOS_INLINE_FUNCTION void SetEOSUnitSystem(UnitSystem units) {
    eos_units = units;
  }

  /// Get the maximum number of allowed polytropes
  KOKKOS_INLINE_FUNCTION int GetMaxPieces() const {
    return MAX_PIECES;
  }
};

#undef MAX_PIECES

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_PIECEWISE_POLYTROPE_HPP_
