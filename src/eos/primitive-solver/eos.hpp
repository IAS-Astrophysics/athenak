#ifndef EOS_PRIMITIVE_SOLVER_EOS_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.hpp
//  \brief Defines an equation of state.
//
//  EOS is effectively an interface that describes how to create an
//  equation of state. It must be instantiated with an object implementing
//  the following protected functions:
//    Real Temperature(Real n, Real e, Real *Y)
//    Real TemperatureFromP(Real n, Real p, Real *Y)
//    Real Energy(Real n, Real T, Real *Y)
//    Real Pressure(Real n, Real T, Real *Y)
//    Real Entropy(Real n, Real T, Real *Y)
//    Real Enthalpy(Real n, Real T, Real *Y)
//    Real MinimumEnthalpy()
//    Real SoundSpeed(Real n, Real T, Real *Y)
//    Real SpecificInternalEnergy(Real n, Real T, Real *Y)
//    Real MinimumPressure(Real n, Real *Y)
//    Real MaximumPressure(Real n, Real *Y)
//    Real MinimumEnergy(Real n, Real *Y)
//    Real MaximumEnergy(Real n, Real *Y)
//  And it must also have the following protected member variables
//  (available via EOSPolicyInterface):
//    const int n_species
//    Real mb
//    Real max_rho
//    Real min_rho
//
//  It must also take an object for an error policy that implements the
//  following functions:
//    bool PrimitiveFloor(Real& n, Real& v[3], Real& p)
//    bool ConservedFloor(Real& D, Real& Sd[3], Real& tau, Real& Bu[3])
//    void DensityLimits(Real& n, Real n_min, Real n_max);
//    void TemperatureLimits(Real& T, Real T_min, Real T_max);
//    void SpeciesLimits(Real* Y, Real* Y_min, Real* Y_max, int n_species);
//    void PressureLimits(Real& P, Real P_min, Real P_max);
//    void EnergyLimits(Real& e, Real e_min, Real e_max);
//    void FailureResponse(Real prim[NPRIM])
//  And the following protected variables (available via
//  ErrorPolicyInterface):
//    Real n_atm
//    Real T_atm
//    Real v_max
//    Real max_bsq_field
//    bool fail_conserved_floor
//    bool fail_primitive_floor
//    bool adjust_conserved

#include <math.h>

#include <limits>
#include <cassert>

#include "ps_types.hpp"
#include "unit_system.hpp"

namespace Primitive {

enum class Error;

template <typename EOSPolicy, typename ErrorPolicy>
class EOS : public EOSPolicy, public ErrorPolicy {
 private:
  // EOSPolicy member functions
  using EOSPolicy::TemperatureFromE;
  using EOSPolicy::TemperatureFromP;
  using EOSPolicy::Energy;
  using EOSPolicy::Pressure;
  using EOSPolicy::Enthalpy;
  using EOSPolicy::SoundSpeed;
  using EOSPolicy::SpecificInternalEnergy;
  using EOSPolicy::MinimumEnthalpy;
  using EOSPolicy::MinimumPressure;
  using EOSPolicy::MaximumPressure;
  using EOSPolicy::MinimumEnergy;
  using EOSPolicy::MaximumEnergy;

  // EOSPolicy member variables
  // The number of particle species used by the EOS.
  using EOSPolicy::n_species;
  // The baryon mass
  using EOSPolicy::mb;
  // Maximum density
  using EOSPolicy::max_n;
  // Minimum density
  using EOSPolicy::min_n;
  // Maximum temperature
  using EOSPolicy::max_T;
  // Minimum temperature
  using EOSPolicy::min_T;
  // Maximum Y
  using EOSPolicy::max_Y;
  // Minimum Y
  using EOSPolicy::min_Y;
  // Code unit system
  using EOSPolicy::code_units;
  // EOS unit system
  using EOSPolicy::eos_units;

  // ErrorPolicy member functions
  using ErrorPolicy::PrimitiveFloor;
  using ErrorPolicy::ConservedFloor;
  using ErrorPolicy::MagnetizationResponse;
  using ErrorPolicy::DensityLimits;
  using ErrorPolicy::TemperatureLimits;
  using ErrorPolicy::SpeciesLimits;
  using ErrorPolicy::PressureLimits;
  using ErrorPolicy::EnergyLimits;
  using ErrorPolicy::FailureResponse;

  // ErrorPolicy member variables
  using ErrorPolicy::n_atm;
  using ErrorPolicy::n_threshold;
  using ErrorPolicy::T_atm;
  using ErrorPolicy::Y_atm;
  using ErrorPolicy::v_max;
  using ErrorPolicy::fail_conserved_floor;
  using ErrorPolicy::fail_primitive_floor;
  using ErrorPolicy::adjust_conserved;
  using ErrorPolicy::max_bsq;

  static constexpr bool supports_entropy = std::is_base_of_v<SupportsEntropy, EOSPolicy>;
  static constexpr bool supports_potentials =
    std::is_base_of_v<SupportsChemicalPotentials, EOSPolicy>;

 public:
  //! \fn EOS()
  //  \brief Constructor for the EOS. It sets a default value for the floor.
  //
  //  n_atm gets fixed to 1e-10, and T_atm is set to 1.0. v_max is fixed to
  //  1.0e - 1e15.
  EOS() {
    n_atm = 1e-10;
    n_threshold = 1.0;
    T_atm = 1e-10;
    v_max = 1.0 - 1e-15;
    max_bsq = std::numeric_limits<Real>::max();
    code_units = eos_units;
    for (int i = 0; i < MAX_SPECIES; i++) {
      Y_atm[i] = 0.0;
    }
  }

  //! \fn Real GetTemperatureFromE(Real n, Real e, Real *Y)
  //  \brief Calculate the temperature from number density, energy density, and
  //         particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] e  The energy density
  //  \param[in] Y  An array of particle fractions, expected to be of size n_species.
  //  \return The temperature according to the EOS.
  KOKKOS_INLINE_FUNCTION Real GetTemperatureFromE(Real n, Real e, Real *Y) const {
    return TemperatureFromE(n, e*code_units.PressureConversion(eos_units), Y) *
           eos_units.TemperatureConversion(code_units);
  }

  //! \fn Real GetTemperatureFromP(Real n, Real p, Real *Y)
  //  \brief Calculate the temperature from number density, pressure, and
  //         particle fractions.
  //  \param[in] n  The number density
  //  \param[in] p  The pressure
  //  \param[in] Y  An array of particle fractions, expected to be of size n_species.
  //  \return The temperature according to the EOS.
  KOKKOS_INLINE_FUNCTION Real GetTemperatureFromP(Real n, Real p, Real *Y) const {
    return TemperatureFromP(n, p*code_units.PressureConversion(eos_units), Y) *
           eos_units.TemperatureConversion(code_units);
  }

  //! \fn Real GetEnergy(Real n, Real T, Real *Y)
  //  \brief Get the energy density from the number density, temperature, and
  //         particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The energy density according to the EOS.
  KOKKOS_INLINE_FUNCTION Real GetEnergy(Real n, Real T, const Real *Y) const {
    return Energy(n, T*code_units.TemperatureConversion(eos_units), Y) *
           eos_units.PressureConversion(code_units);
  }

  //! \fn Real GetPressure(Real n, Real T, Real *Y)
  //  \brief Get the pressure from the number density, temperature, and
  //         particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The pressure according to the EOS.
  KOKKOS_INLINE_FUNCTION Real GetPressure(Real n, Real T, Real *Y) const {
    return Pressure(n, T*code_units.TemperatureConversion(eos_units), Y) *
           eos_units.PressureConversion(code_units);
  }

  //! \fn Real GetEntropy(Real n, Real T, Real *Y)
  //  \brief Get the entropy per mass from the number density, temperature,
  //         and particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The entropy per baryon for this EOS.
  KOKKOS_INLINE_FUNCTION Real GetEntropy(Real n, Real T, Real *Y) const {
    if constexpr (supports_entropy) {
      return EOSPolicy::Entropy(n, T*code_units.TemperatureConversion(eos_units), Y)/mb *
             eos_units.EntropyConversion(code_units)/eos_units.MassConversion(code_units);
    } else {
      return std::numeric_limits<Real>::quiet_NaN();
    }
  }

  //! \fn Real GetEnthalpy(Real n, Real T, Real *Y)
  //  \brief Get the enthalpy per mass from the number density, temperature,
  //         and particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The enthalpy per baryon for this EOS.
  KOKKOS_INLINE_FUNCTION Real GetEnthalpy(Real n, Real T, Real *Y) const {
    return Enthalpy(n, T*code_units.TemperatureConversion(eos_units), Y)/mb *
           (eos_units.EnergyConversion(code_units)/eos_units.MassConversion(code_units));
  }

  //! \fn Real GetMinimumEnthalpy()
  //  \brief Get the global minimum for enthalpy per mass from the EOS.
  //
  //  \return the minimum enthalpy per mass.
  KOKKOS_INLINE_FUNCTION Real GetMinimumEnthalpy() const {
    return MinimumEnthalpy()/mb *
           eos_units.EnergyConversion(code_units)/eos_units.MassConversion(code_units);
  }

  //! \fn Real GetSoundSpeed(Real n, Real T, Real *Y)
  //  \brief Get the sound speed from the number density, temperature, and
  //         particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The sound speed for this EOS.
  KOKKOS_INLINE_FUNCTION Real GetSoundSpeed(Real n, Real T, Real *Y) const {
    return SoundSpeed(n, T*code_units.TemperatureConversion(eos_units), Y) *
           eos_units.VelocityConversion(code_units);
  }

  //! \fn Real GetSpecificInternalEnergy(Real n, Real T, Real *Y)
  //  \brief Get the energy per mass from the number density, temperature,
  //         and particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The specific energy for the EOS.
  KOKKOS_INLINE_FUNCTION Real GetSpecificInternalEnergy(Real n, Real T, Real *Y) const {
    return SpecificInternalEnergy(n, T*code_units.TemperatureConversion(eos_units), Y) *
           eos_units.EnergyConversion(code_units)/eos_units.MassConversion(code_units);
  }

  //! \fn Real GetBaryonChemicalPotential(Real n, Real T, Real *Y)
  //  \brief Get the baryon chemical potential from the number density, temperature,
  //         and particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The baryon chemical potential for the EOS.
  KOKKOS_INLINE_FUNCTION Real GetBaryonChemicalPotential(Real n, Real T, Real *Y) const {
    if constexpr (supports_potentials) {
      return EOSPolicy::BaryonChemicalPotential(n,
               T*code_units->TemperatureConversion(*eos_units), Y) *
             eos_units->ChemicalPotentialConversion(*code_units);
    } else {
      return std::numeric_limits<Real>::quiet_NaN();
    }
  }

  //! \fn Real GetChargeChemicalPotential(Real n, Real T, Real *Y)
  //  \brief Get the charge chemical potential from the number density, temperature,
  //         and particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The charge chemical potential for the EOS.
  KOKKOS_INLINE_FUNCTION Real GetChargeChemicalPotential(Real n, Real T, Real *Y) const {
    if constexpr (supports_potentials) {
      return EOSPolicy::ChargeChemicalPotential(n,
               T*code_units->TemperatureConversion(*eos_units), Y) *
             eos_units->ChemicalPotentialConversion(*code_units);
    } else {
      return std::numeric_limits<Real>::quiet_NaN();
    }
  }

  //! \fn Real GetElectronLeptonChemicalPotential(Real n, Real T, Real *Y)
  //  \brief Get the electron-lepton chemical potential from the number density,
  //         temperature, and particle fractions.
  //
  //  \param[in] n  The number density
  //  \param[in] T  The temperature
  //  \param[in] Y  An array of size n_species of the particle fractions.
  //  \return The electron-lepton chemical potential for the EOS.
  KOKKOS_INLINE_FUNCTION Real GetElectronLeptonChemicalPotential(Real n, Real T,
                                                                 Real *Y) const {
    if constexpr (supports_potentials) {
      return EOSPolicy::ElectronLeptonChemicalPotential(n,
               T*code_units->TemperatureConversion(*eos_units), Y) *
             eos_units->ChemicalPotentialConversion(*code_units);
    } else {
      return std::numeric_limits<Real>::quiet_NaN();
    }
  }

  //! \fn Real GetBetaEquilibriumTrapped(Real n, Real e, Real *Yl, Real &T_eq,
  //                                     Real *Y_eq, Real T_guess, Real *Y_guess)
  //  \brief Get the equilibrium temperature and species fractions from the energy and
  //         total lepton fractions
  //
  //  \param[in]    n       The number density
  //  \param[in]    e       The total energy density (fluid plus neutrinos)
  //  \param[in]    Yl      An array of size n_species of the total lepton fractions.
  //  \param[inout] T_eq    The equilibrium temperature.
  //  \param[inout] Y_eq    The equilibrium particle fractions.
  //  \param[in]    T_guess Initial guess for the temperature.
  //  \param[in]    Y_guess Initial guesses for the particle fractions.
  //  \return Whether the equilibrium was successfully found.
  KOKKOS_INLINE_FUNCTION bool GetBetaEquilibriumTrapped(Real n, Real e, Real *Yl,
                                Real &T_eq, Real *Y_eq, Real T_guess, Real *Y_guess) {
    if constexpr (supports_potentials) {
      int ierr = EOSPolicy::BetaEquilibriumTrapped(n,
                   e*code_units->PressureConversion(*eos_units), Yl, T_eq, Y_eq,
                   T_guess*code_units->TemperatureConversion(*eos_units), Y_guess);

      T_eq = T_eq*eos_units->TemperatureConversion(*code_units);

      return ierr==0;
    } else {
      return false;
    }
  }

  //! \fn Real GetTrappedNeutrinos(Real n, Real T, Real *Y, Real n_nu[3], Real e_nu[3])
  // \brief Get the trapped neutrino net number and energy densities.
  //
  //  \param[in]    n    The number density
  //  \param[in]    T    The temperature
  //  \param[in]    Y    An array of size n_species of the particle fractions.
  //  \param[inout] n_nu The net number densities for each neutrino generation.
  //  \param[inout] e_nu The total energy densities for each neutrino generation.
  inline void GetTrappedNeutrinos(Real n, Real T, Real *Y, Real n_nu[3], Real e_nu[3]) {
    if constexpr (supports_potentials) {
      EOSPolicy::TrappedNeutrinos(n, T*code_units->TemperatureConversion(*eos_units), Y,
                                  n_nu, e_nu);

      Real n_units = eos_units->DensityConversion(*code_units);
      Real e_units = eos_units->PressureConversion(*code_units);

      for (int i=0; i<3; ++i) {
        n_nu[i] = n_nu[i]*n_units;
        e_nu[i] = e_nu[i]*e_units;
      }
    }
    return;
  }

  //! \fn Real GetLeptonFractions(Real n, Real *Y, Real n_nu[6], Real *Yl)
  // \brief Get the total lepton fractions for each generation of matter from the species
  //        fractions and the neutrino number densities.
  //
  //  \param[in]    n    The number density (N.B this should already be in EoS units, via
  //                     rho/GetBaryonMass())
  //  \param[in]    Y    The particle fractions.
  //  \param[in]    n_nu The number densities for each neutrino species (e, ae, m, am, t,
  //                     at) (N.B. these are expected to be in code units).
  //  \param[inout] Yl   The total lepton fractions.
  inline void GetLeptonFractions(Real n, Real *Y, Real n_nu[6], Real *Yl) {
    Real n_units = code_units->DensityConversion(*eos_units);

    for (int i=0; i<3; ++i) {
      Yl[i] = Y[i] + n_units*(n_nu[2*i] - n_nu[2*i+1])/n;
    }

    return;
  }

  //! \fn int GetNSpecies() const
  //  \brief Get the number of particle species in this EOS.
  KOKKOS_INLINE_FUNCTION int GetNSpecies() const {
    return n_species;
  }

  //! \fn Real GetBaryonMass() const
  //  \brief Get the baryon mass used by this EOS. Note that
  //         this factor also converts the density.
  KOKKOS_INLINE_FUNCTION Real GetBaryonMass() const {
    return mb*eos_units.MassConversion(code_units) *
              eos_units.DensityConversion(code_units);
  }

  //! \fn bool ApplyPrimitiveFloor(Real& n, Real& vu[3], Real& p, Real& T)
  //  \brief Apply the floor to the primitive variables (in code units).
  //
  //  \param[in,out] n   The number density
  //  \param[in,out] Wvu The velocity vector (contravariant)
  //  \param[in,out] p   The pressure
  //  \param[out]    T   The temperature
  //  \param[in]     Y   An array of size n_species of the particle fractions.
  //
  //  \return true if the primitives were adjusted, false otherwise.
  KOKKOS_INLINE_FUNCTION bool ApplyPrimitiveFloor(Real& n, Real Wvu[3],
                                                  Real& p, Real& T, Real *Y) const {
    bool result = PrimitiveFloor(n, Wvu, T, Y, n_species);
    if (result) {
      p = GetPressure(n, T, Y);
    }
    return result;
  }

  //! \fn bool ApplyConservedFloor(Real& D, Real& Sd[3], Real& tau, Real *Y, Real Bsq)
  //  \brief Apply the floor to the conserved variables (in code units).
  //
  //  \param[in,out] D   The relativistic number density
  //  \param[in,out] Sd  The momentum density vector (covariant)
  //  \param[in,out] tau The tau variable (relativistic energy - D)
  //  \param[in]     Y   An array of size_species of the particle fractions.
  //  \param[in]     Bsq The norm of the magnetic field
  //
  //  \return true if the conserved variables were adjusted, false otherwise.
  KOKKOS_INLINE_FUNCTION bool ApplyConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y,
                                                  Real Bsq) const {
    return ConservedFloor(D, Sd, tau, Y, n_atm*GetBaryonMass(),
                          GetTauFloor(fmax(D,min_n*GetBaryonMass()), Y, Bsq),
                          GetTauFloor(n_atm*GetBaryonMass(), Y_atm, Bsq), n_species);
  }

  //! \fn Real GetDensityFloor() const
  //  \brief Get the density floor used by the EOS ErrorPolicy.
  KOKKOS_INLINE_FUNCTION Real GetDensityFloor() const {
    return n_atm;
  }

  //! \fn Real GetTemperatureFloor() const
  //  \brief Get the temperature floor used by the EOS ErrorPolicy.
  KOKKOS_INLINE_FUNCTION Real GetTemperatureFloor() const {
    return T_atm;
  }

  //! \fn Real GetSpeciesAtmosphere(int i) const
  //  \brief Get the atmosphere abundance used by the EOS ErrorPolicy for species i.
  KOKKOS_INLINE_FUNCTION Real GetSpeciesAtmosphere(int i) const {
    assert((i < n_species) && "Not enough species");
    return Y_atm[i];
  }

  //! \fn Real GetThreshold() const
  //  \brief Get the threshold factor used by the EOS ErrorPolicy.
  KOKKOS_INLINE_FUNCTION Real GetThreshold() const {
    return n_threshold;
  }

  //! \fn Real GetTauFloor(Real D, Real *Y)
  //  \brief Get the tau floor used by the EOS ErrorPolicy based
  //         on the current particle composition.
  //
  //  \param[in] Y A n_species-sized array of particle fractions.
  KOKKOS_INLINE_FUNCTION Real GetTauFloor(Real D, const Real *Y, Real Bsq) const {
    return GetEnergy(D/GetBaryonMass(), T_atm, Y) - D + 0.5*Bsq;
  }

  //! \fn void SetDensityFloor(Real floor)
  //  \brief Set the density floor used by the EOS ErrorPolicy.
  //         Also adjusts the pressure and tau floor to be consistent.
  KOKKOS_INLINE_FUNCTION void SetDensityFloor(Real floor) {
    n_atm = (floor >= min_n) ? floor : min_n;
  }

  //! \fn void SetTemperatureFloor(Real floor)
  //  \brief Set the temperature floor (in code units) used by the EOS ErrorPolicy.
  KOKKOS_INLINE_FUNCTION void SetTemperatureFloor(Real floor) {
    T_atm = (floor >= min_T) ? floor : min_T;
  }

  //! \fn void SetSpeciesAtmospher(Real atmo, int i)
  //  \brief Set the atmosphere abundance used by the EOS ErrorPolicy for species i.
  KOKKOS_INLINE_FUNCTION void SetSpeciesAtmosphere(Real atmo, int i) {
    assert((i < n_species) && "Not enough species");
    Y_atm[i] = fmin(max_Y[i], fmax(min_Y[i], atmo));
  }

  //! \fn void SetThreshold(Real threshold)
  //  \brief Set the threshold factor for the density floor.
  KOKKOS_INLINE_FUNCTION void SetThreshold(Real threshold) {
    n_threshold = (threshold >= 0.0) ? threshold : 0.0;
  }

  //! \fn Real GetMaxVelocity() const
  //  \brief Get the maximum velocity according to the ErrorPolicy.
  KOKKOS_INLINE_FUNCTION Real GetMaxVelocity() const {
    return v_max;
  }

  //! \fn void SetMaxVelocity(Real v)
  //  \brief Set the maximum velocity in the ErrorPolicy.
  //
  //  The velocity will be automatically restricted to the range [0,1 - 1e-15].
  //
  //  \param[in] v The maximum velocity
  KOKKOS_INLINE_FUNCTION void SetMaxVelocity(Real v) {
    v_max = (v >= 0) ? ((v <= 1.0-1e-15) ? v : 1.0-1.0e-15) : 0.0;
  }

  //! \brief Get the maximum number density (in EOS units) permitted by the EOS.
  KOKKOS_INLINE_FUNCTION Real GetMaximumDensity() const {
    return max_n;
  }

  //! \brief Get the minimum number density (in EOS units) permitted by the EOS.
  KOKKOS_INLINE_FUNCTION Real GetMinimumDensity() const {
    return min_n;
  }

  //! \brief Get the maximum temperature  (in EOS units) permitted by the EOS.
  KOKKOS_INLINE_FUNCTION Real GetMaximumTemperature() const {
    return max_T;
  }

  //! \brief Get the minimum temperature (in EOS units) permitted by the EOS.
  KOKKOS_INLINE_FUNCTION Real GetMinimumTemperature() const {
    return min_T;
  }

  //! \brief Get the minimum fraction permitted by the EOS.
  KOKKOS_INLINE_FUNCTION Real GetMinimumSpeciesFraction(int i) const {
    return min_Y[i];
  }

  //! \brief Get the maximum fraction permitted by the EOS.
  KOKKOS_INLINE_FUNCTION Real GetMaximumSpeciesFraction(int i) const {
    return max_Y[i];
  }

  //! \fn const bool IsConservedFlooringFailure() const
  //  \brief Find out if the EOSPolicy fails flooring the conserved variables.
  //
  // \return true or false
  KOKKOS_INLINE_FUNCTION bool IsConservedFlooringFailure() const {
    return fail_conserved_floor;
  }

  //! \fn const bool IsPrimitiveFlooringFailure() const
  //  \brief Find out if the EOSPolicy fails flooring the primitive variables.
  //
  //  \return true or false
  KOKKOS_INLINE_FUNCTION bool IsPrimitiveFlooringFailure() const {
    return fail_primitive_floor;
  }

  //! \fn const bool KeepPrimAndConConsistent() const
  //  \brief Find out if the EOSPolicy wants the conserved variables to be
  //         adjusted to match the primitive variables.
  //
  //  \return true or false
  KOKKOS_INLINE_FUNCTION bool KeepPrimAndConConsistent() const {
    return adjust_conserved;
  }

  //! \brief Get the maximum squared magnetic field permitted by the ErrorPolicy
  KOKKOS_INLINE_FUNCTION Real GetMaximumMagnetization() const {
    return max_bsq;
  }

  //! \brief Set the maximum squared magnetic field permitted by the ErrorPolicy
  //         Adjusts the input to make sure it's nonnegative (does not
  //         return an error).
  KOKKOS_INLINE_FUNCTION void SetMaximumMagnetization(double bsq) {
    max_bsq = (bsq >= 0) ? bsq : 0.0;
  }

  //! \brief Respond to excess magnetization
  KOKKOS_INLINE_FUNCTION Error DoMagnetizationResponse(Real& bsq, Real b_u[3]) const {
    return MagnetizationResponse(bsq, b_u);
  }

  //! \brief Limit the density to a physical range
  KOKKOS_INLINE_FUNCTION void ApplyDensityLimits(Real& n) const {
    DensityLimits(n, min_n, max_n);
  }

  //! \brief Limit the temperature to a physical range
  KOKKOS_INLINE_FUNCTION void ApplyTemperatureLimits(Real& T) const {
    Real T_eos = T*code_units.TemperatureConversion(eos_units);
    TemperatureLimits(T_eos, min_T, max_T);
    T = T_eos*eos_units.TemperatureConversion(code_units);
  }

  //! \brief Limit Y to a specified range
  KOKKOS_INLINE_FUNCTION bool ApplySpeciesLimits(Real *Y) const {
    return SpeciesLimits(Y, min_Y, max_Y, n_species);
  }

  //! \brief Limit the pressure to a specified range at a given density and composition
  KOKKOS_INLINE_FUNCTION void ApplyPressureLimits(Real& P, Real n, Real* Y) const {
    Real P_eos = P*code_units.PressureConversion(eos_units);
    PressureLimits(P_eos, MinimumPressure(n, Y), MaximumPressure(n, Y));
    P = P_eos*eos_units.PressureConversion(code_units);
  }

  //! \brief Limit the energy density to a specified range at a given density and
  //  composition
  KOKKOS_INLINE_FUNCTION void ApplyEnergyLimits(Real& e, Real n, Real* Y) const {
    Real e_eos = e*code_units.PressureConversion(eos_units);
    EnergyLimits(e_eos, MinimumEnergy(n, Y), MaximumEnergy(n, Y));
    e = e_eos*eos_units.PressureConversion(code_units);
  }

  //! \brief Respond to a failed solve.
  KOKKOS_INLINE_FUNCTION bool DoFailureResponse(Real prim[NPRIM]) const {
    bool result = FailureResponse(prim);
    if (result) {
      // Adjust the pressure to be consistent with the new primitive variables.
      prim[PPR] = GetPressure(prim[PRH], prim[PTM], &prim[PYF]);
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION void SetCodeUnitSystem(UnitSystem units) {
    code_units = units;
  }

  KOKKOS_INLINE_FUNCTION UnitSystem& GetCodeUnitSystem() const {
    return code_units;
  }

  KOKKOS_INLINE_FUNCTION UnitSystem& GetEOSUnitSystem() const {
    return eos_units;
  }
};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_EOS_HPP_
