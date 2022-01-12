//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file units.cpp

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "units.hpp"

namespace units {
//----------------------------------------------------------------------------------------
//! \brief Units constructor
Units::Units(ParameterInput *pin) :
  length_cgs_(pin->GetOrAddReal("units", "length_cgs", 1.0)),
  mass_cgs_(pin->GetOrAddReal("units", "mass_cgs", 1.0)),
  time_cgs_(pin->GetOrAddReal("units", "time_cgs", 1.0)),
  mu_(pin->GetOrAddReal("units", "mu", 1.0)) {}

//----------------------------------------------------------------------------------------
//! \brief Units destructor
Units::~Units()
{
}

//----------------------------------------------------------------------------------------
// Code scales in cgs units
Real Units::length_cgs() const { return length_cgs_; }
Real Units::mass_cgs() const { return mass_cgs_; }
Real Units::time_cgs() const { return time_cgs_; }

// mean molecular weight
Real Units::mu() const { return mu_; }

//----------------------------------------------------------------------------------------
// Derived code scales in cgs units
// Converting variables from code unit to physical (cgs) unit
// i.e., variable_in_physical_unit = variable_in_code_unit * code_scale  
Real Units::velocity_cgs() const {
  return Units::length_cgs() / Units::time_cgs();
}
Real Units::density_cgs() const {
  return Units::mass_cgs() / 
         (Units::length_cgs() * Units::length_cgs() * Units::length_cgs());
}
Real Units::energy_cgs() const {
  return Units::mass_cgs() * Units::velocity_cgs() * Units::velocity_cgs();
}
Real Units::pressure_cgs() const {
  return Units::energy_cgs() / 
         (Units::length_cgs() * Units::length_cgs() * Units::length_cgs());
}
Real Units::temperature_cgs() const{
  return Units::velocity_cgs() * Units::velocity_cgs() * Units::mu() *
         Units::atomic_mass_unit_cgs / Units::k_boltzmann_cgs;
}

//----------------------------------------------------------------------------------------
// Code unit per X, or X in code units
// Converting variables from physical (cgs) unit to code unit
// i.e., variable_in_code_unit = variable_in_physical_unit * X

// Length
Real Units::cm() const { return cm_cgs / Units::length_cgs(); }
Real Units::pc() const { return pc_cgs / Units::length_cgs(); }
Real Units::kpc() const { return kpc_cgs / Units::length_cgs(); }

// Mass
Real Units::g() const { return g_cgs / Units::mass_cgs(); }
Real Units::msun() const { return msun_cgs / Units::mass_cgs(); }
Real Units::atomic_mass_unit() const { return atomic_mass_unit_cgs / Units::mass_cgs(); }

// Time
Real Units::s() const { return s_cgs / Units::time_cgs(); }
Real Units::yr() const { return yr_cgs / Units::time_cgs(); }
Real Units::myr() const { return myr_cgs / Units::time_cgs(); }

// Velocity
Real Units::cm_s() const { return cm_s_cgs / Units::velocity_cgs(); }
Real Units::km_s() const { return km_s_cgs / Units::velocity_cgs(); }

// Density
Real Units::g_cm3() const { return g_cm3_cgs / Units::density_cgs(); }

// Energy
Real Units::erg() const { return erg_cgs / Units::energy_cgs(); }

// Pressure
Real Units::dyne_cm2() const { return dyne_cm2_cgs / Units::pressure_cgs(); }

// Temperature
Real Units::kelvin() const { return kelvin_cgs / Units::temperature_cgs(); }

//----------------------------------------------------------------------------------------
// Physical Constants in code units
Real Units::k_boltzmann() const {
  return k_boltzmann_cgs / (Units::energy_cgs() / Units::temperature_cgs());
}

} // namespace units
