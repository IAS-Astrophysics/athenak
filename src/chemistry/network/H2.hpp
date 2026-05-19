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
    IIE = 0,  // internal energy
    IH2 = 1,  // H_2
    IH = 2    // H
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
    return H2Settings{pin->GetOrAddBoolean("problem", "constant_cv", false)};
  }

  /*!
   * \brief Updates `f` using the values in `y`
   */
  KOKKOS_FUNCTION
  void evaluate_function() {
    // ----- Internal energy equation -----
    static constexpr Real x_He = 0.1;
    static constexpr Real x_e = 0.0;
    const Real x_H2 = (const_cv) ? 0.0 : y(IH2);

    static constexpr Real T_floor = 1.;  // temperature floor for cooling
    // energy per hydrogen atom
    const Real E_ergs = y(IIE) * units_energy_density_cgs / n_H;
    const Real T = E_ergs / Thermo::CvCold(x_H2, x_He, x_e);
    if (T < T_floor) {
      f(IIE) = 0;
    } else {
      const Real dEdt = -Thermo::alpha_GD_ * n_H * Kokkos::sqrt(T) * T;
      // convert to code units
      f(IIE) = (dEdt * n_H / units_energy_density_cgs);
    }

    // ----- Abundance equations -----
    // cr = cosmic ray, gr = dust grain
    const Real rate_cr = k_cr * y(IH2);
    const Real rate_gr = k_gr * n_H * y(IH);

    // H_2 equation
    f(IH2) = rate_gr - rate_cr;
    // H equation
    f(IH) = 2 * (rate_cr - rate_gr);

    // ----- Convert all back to code units -----
    for (size_t i = 0; i < neqs; i++) {
      f(i) *= units_time_cgs;
    }
  }
};
}  // namespace chemistry
#endif  // CHEMISTRY_NETWORK_H2_HPP_
