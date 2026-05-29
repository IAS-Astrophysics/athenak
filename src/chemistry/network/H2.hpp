#ifndef CHEMISTRY_NETWORK_H2_HPP_
#define CHEMISTRY_NETWORK_H2_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file H2.hpp
//  \brief The implementation for the struct for the H2 chemistry network

#include "athena.hpp"
#include "chemistry/thermo/thermo.hpp"
#include "utils/register_array.hpp"

namespace chemistry {
/*!
 * \brief The class for the H2 network. This also serves as a template for other
 * chemistry networks.
 *
 * \details All chemistry networks are required to have a handful of specific
 * things so that the ODE solvers have a common interface to work with. They
 * need:
 *   - A `neqs` variable that specifies the number of equations. That should be
 *     the number of species plus 1 for the internal energy .
 *   - `y` and `f` RegisterArray variables to hold the current state and the
 *     result of evaluating the equations respectively.
 *   - An `evaluate_function` method that computes `f` from `y`
 */
struct H2Settings {
  /// If C_v should be held constant
  bool const_cv;
};

class H2Network {
 public:
  KOKKOS_FUNCTION H2Network(H2Settings const settings, Real const density,
                            Real const density_cgs, Real const mu,
                            Real const hydrogen_mass_cgs,
                            Real const units_time_cgs,
                            Real const units_energy_density_cgs)
      : n_H(density * density_cgs / (mu * hydrogen_mass_cgs)),
        units_time_cgs(units_time_cgs),
        units_energy_density_cgs(units_energy_density_cgs),
        const_cv(settings.const_cv) {}

  // ----- Number of equations -----
  static constexpr int neqs = 3;

  // ----- If Cv is const or not -----
  const bool const_cv;

  // ----- Arrays to store ODE state -----
  RegisterArray<Real, neqs> y;  // The current state
  RegisterArray<Real, neqs> f;  // The results of evaluating the ODEs

  // ----- Species indices within the ODE system ------
  enum {
    IH2,  // H_2
    IH,   // H
    IIE   // internal energy, must be last
  };

  // ----- Names, used for output, must be the same order as the enum -----
  static constexpr std::array<std::string_view, neqs> species_names = {"H2",
                                                                       "H"};

  // ----- cell values -----
  Real const n_H;  // The number density of hydrogen

  // ----- unit conversion factors -----
  Real const units_time_cgs;
  Real const units_energy_density_cgs;

  // ----- Reaction rate constants -----
  static constexpr Real k_gr = 3.0e-17;
  // xi_cr is the primary cosmic-ray ionization rate per H
  static constexpr Real xi_cr = 2.0e-16;
  static constexpr Real k_cr = 3.0 * xi_cr;

  // ----- Member Functions -----
  /*!
   * \brief Get the settings for the H2 network from the input file
   *
   * \param pin The ParameterInput object
   * \return H2Settings The settings for the H2 network
   */
  static H2Settings GetSettings(ParameterInput* pin) {
    return H2Settings{pin->GetOrAddBoolean("chemistry", "h2_constant_cv", false)};
  }

  /*!
   * \brief Compute the temperature in the cell
   *
   * \return Real The temperature in the cell
   */
  KOKKOS_FUNCTION
  Real Temperature() {
    // Constants needed for computing temperature
    static constexpr Real x_He = 0.1;
    static constexpr Real x_e = 0.0;
    const Real x_H2 = (const_cv) ? 0.0 : y(IH2);

    // energy per hydrogen atom
    const Real E_ergs = y(IIE) * units_energy_density_cgs / n_H;

    // Temperature
    return E_ergs / Thermo::CvCold(x_H2, x_He, x_e);
  }

  /*!
   * \brief Compute the cooling term from the temperature
   *
   * \param T The temperature in the cell
   * \return Real The cooling term, i.e. how much the energy decreases. Note
   * that this is a positive value so the energy update should look like `E =
   * HeatingTerm() - CoolingTerm();`
   */
  KOKKOS_FUNCTION
  Real CoolingTerm(Real const& T) {
    return Thermo::alpha_GD_ * n_H * Kokkos::sqrt(T) * T;
  }

  /*!
   * \brief Compute the heating term from the temperature. It's just zero for
   * the H2 network and this function exists for API consistency across
   * networks.
   *
   * \param T The temperature in the cell
   * \return Real The heating term, i.e. how much the energy increases. The
   * energy update should look like `E = HeatingTerm() - CoolingTerm();`
   */
  KOKKOS_FUNCTION
  Real HeatingTerm(Real const& T) { return 0.0; }

  /*!
   * \brief Evaluate the internal energy equation
   *
   * \return Real The result of evaluating the internal energy equation
   */
  KOKKOS_FUNCTION
  Real Edot() {
    const Real T = Temperature();

    static constexpr Real T_floor = 1.0;  // temperature floor for cooling
    if (T < T_floor) {
      return 0;
    } else {
      const Real dEdt = HeatingTerm(T) - CoolingTerm(T);
      // convert to code units
      return units_time_cgs * (dEdt * n_H / units_energy_density_cgs);
    }
  }

  KOKKOS_FUNCTION
  auto CreationRates() {
    RegisterArray<Real, neqs - 1> creation_rates;

    // cr = cosmic ray, gr = dust grain
    const Real rate_cr = k_cr * y(IH2);
    const Real rate_gr = k_gr * n_H * y(IH);

    // H_2 equation
    creation_rates(IH2) = rate_gr;
    // H equation
    creation_rates(IH) = 2 * rate_cr;

    // convert to code units
    for (size_t i = 0; i < neqs - 1; i++) {
      creation_rates(i) *= units_time_cgs;
    }

    return creation_rates;
  }

  KOKKOS_FUNCTION
  auto DestructionRates() {
    RegisterArray<Real, neqs - 1> destruction_rates;

    // cr = cosmic ray, gr = dust grain
    const Real rate_cr = k_cr * y(IH2);
    const Real rate_gr = k_gr * n_H * y(IH);

    // H_2 equation
    destruction_rates(IH2) = rate_cr;
    // H equation
    destruction_rates(IH) = 2 * rate_gr;

    // convert to code units
    for (size_t i = 0; i < neqs - 1; i++) {
      destruction_rates(i) *= units_time_cgs;
    }

    return destruction_rates;
  }

  /*!
   * \brief Updates `f` using the values in `y`
   */
  KOKKOS_FUNCTION
  void evaluate_function() {
    // ----- Internal energy equation -----
    f(IIE) = Edot();

    // ----- Creation & Destruction Rates -----
    auto creation_rates = CreationRates();
    auto destruction_rates = DestructionRates();

    // Compute the changes
    for (size_t i = 0; i < neqs - 1; i++) {
      f(i) = (creation_rates(i) - destruction_rates(i));
    }
  }
};
}  // namespace chemistry
#endif  // CHEMISTRY_NETWORK_H2_HPP_
