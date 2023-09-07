#ifndef UNITS_UNITS_HPP_
#define UNITS_UNITS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file units.hpp
//! \brief definitions for Units class

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"

namespace units {

//----------------------------------------------------------------------------------------
//! \class Units
//! \brief data and definitions of functions used to store and access units
//  Functions are implemented in units.cpp

class Units {
 public:
  explicit Units(ParameterInput *pin);
  ~Units();

  // data
  // CGS unit per X
  static constexpr Real cm_cgs = 1.0;                           // cm
  static constexpr Real pc_cgs = 3.0856775809623245e+18;        // cm
  static constexpr Real kpc_cgs = 3.0856775809623245e+21;       // cm
  static constexpr Real g_cgs = 1.0;                            // g
  static constexpr Real msun_cgs = 1.98841586e+33;              // g
  static constexpr Real atomic_mass_unit_cgs = 1.660538921e-24; // g
  static constexpr Real s_cgs = 1.0;                            // s
  static constexpr Real yr_cgs = 3.15576e+7;                    // s
  static constexpr Real myr_cgs = 3.15576e+13;                  // s
  static constexpr Real cm_s_cgs = 1.0;                         // cm/s
  static constexpr Real km_s_cgs = 1.0e5;                       // cm/s
  static constexpr Real g_cm3_cgs = 1.0;                        // g/cm^3
  static constexpr Real erg_cgs = 1.0;                          // erg
  static constexpr Real dyne_cm2_cgs = 1.0;                     // dyne/cm^2
  static constexpr Real kelvin_cgs = 1.0;                       // k

  // PHYSICAL CONSTANTS
  static constexpr Real k_boltzmann_cgs = 1.3806488e-16;          // erg/k
  static constexpr Real grav_constant_cgs = 6.67408e-8;           // cm^3/(g*s^2)
  static constexpr Real speed_of_light_cgs = 2.99792458e10;       // cm/s
  static constexpr Real rad_constant_cgs = 7.56573325e-15;        // erg/(cm^3*K^4)
  static constexpr Real electron_rest_mass_energy_cgs = 5.93e9;   // k

  // Power Law Coefficients (cgs)
  static constexpr Real rosseland_coef_cgs = 1.95705e24;
  static constexpr Real planck_minus_rosseland_coef_cgs = 7.04536e25;

  // functions
  // Code scales in cgs
  Real length_cgs() const;
  Real mass_cgs() const;
  Real time_cgs() const;

  // mean molecular weight
  Real mu() const;

  // Derived code scales in cgs
  Real velocity_cgs() const;
  Real density_cgs() const;
  Real energy_cgs() const;
  Real pressure_cgs() const;
  Real temperature_cgs() const;

  // Code unit per X, or X in code units
  Real cm() const;
  Real pc() const;
  Real kpc() const;
  Real g() const;
  Real msun() const;
  Real atomic_mass_unit() const;
  Real s() const;
  Real yr() const;
  Real myr() const;
  Real cm_s() const;
  Real km_s() const;
  Real g_cm3() const;
  Real erg() const;
  Real dyne_cm2() const;
  Real kelvin() const;

  // Physical Constants in code units
  Real k_boltzmann() const;
  Real grav_constant() const;
  Real speed_of_light() const;

 private:
  // Specified code scales in cgs units
  // (Multiply code units to get quantities in cgs units)
  // (cgs unit per code unit)
  Real length_cgs_, mass_cgs_, time_cgs_;

  // mean molecular weight
  const Real mu_;
};

} // namespace units

#endif // UNITS_UNITS_HPP_
