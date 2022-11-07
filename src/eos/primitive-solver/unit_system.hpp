#ifndef EOS_UNITS_HPP
#define EOS_UNITS_HPP

//! \file eos_units.hpp
//  \brief contains unit definitions and conversion for the EOS solver.
//  
//  Each unit system is defined as its own struct inside the EOSUnits namespace.
//  TODO: Check that these conversions are correct.

#include "ps_types.hpp"

#define PS_SQR(x) ((x)*(x))
#define PS_CUBE(x) ((x)*(x)*(x))

namespace Primitive {

struct UnitSystem {
  const Real c;    //! Speed of light
  const Real G;    //! Gravitational constant
  const Real kb;   //! Boltzmann constant
  const Real Msun; //! Solar mass
  const Real MeV;  // 10^6 electronvolt

  const Real length;      //! Length unit
  const Real time;        //! Time unit
  const Real density;     //! Number density unit
  const Real mass;        //! Mass unit
  const Real energy;      //! Energy unit
  const Real pressure;    //! Pressure unit
  const Real temperature; //! Temperature unit

  //! \defgroup conversiongroup Conversion Methods
  //  A collection of methods for getting unit
  //  conversions from the original system to the
  //  specified system.
  //  \{
  KOKKOS_INLINE_FUNCTION constexpr Real LengthConversion(UnitSystem& b) const {
    return b.length/length;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real TimeConversion(UnitSystem& b) const {
    return b.time/time;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real VelocityConversion(UnitSystem& b) const {
    return b.length/length * time/b.time;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real DensityConversion(UnitSystem& b) const {
    return b.density/density;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real MassConversion(UnitSystem& b) const {
    return b.mass/mass;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real EnergyConversion(UnitSystem& b) const {
    return b.energy/energy;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real EntropyConversion(UnitSystem& b) const {
    return b.kb/kb;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real PressureConversion(UnitSystem& b) const {
    return b.pressure/pressure;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real TemperatureConversion(UnitSystem& b) const {
    return b.temperature/temperature;
  }
  //! \}
};

// Global static objects for a particular unit system.

//! CGS units
//
//  Fundamental constants are defined using the 2014
//  CODATA values to be consistent with CompOSE. Solar
//  mass is derived from the solar mass parameter given
//  in the 2021 Astronomer's Almanac:
//  GM_S = 1.32712442099e26 cm^3 s^-2
static UnitSystem CGS{
  2.99792458e10, // c, cm/s
  6.67408e-8, // G, cm^3 g^-1 s^-2
  1.38064852e-16, // kb, erg K^-1
  1.98848e33, // Msun, g
  1.6021766208e-6, // MeV, erg

  1.0, // length, cm
  1.0, // time, s
  1.0, // density, g cm^-3
  1.0, // mass, g
  1.0, // energy, erg
  1.0, // pressure, erg/cm^3
  1.0  // temperature, K
};
//! Geometric units with length in kilometers
static UnitSystem GeometricKilometer{
  1.0, // c
  1.0, // G
  1.0, // kb
  CGS.Msun * CGS.G/(CGS.c*CGS.c)*1e-5, // Msun, km
  CGS.MeV * CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5, // MeV, km

  1e-5, // length, km
  CGS.c * 1e-5, // time, km
  1e15, // number density, km^-3
  CGS.G/(CGS.c*CGS.c)*1e-5, // mass, km
  CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5, // energy, km
  CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e10, // pressure, km^-2
  CGS.kb*CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5, // temperature, km
};
//! Geometric units with length in solar masses
static UnitSystem GeometricSolar{
  1.0, // c
  1.0, // G
  1.0, // kb
  1.0, // Msun
  CGS.MeV / (CGS.c*CGS.c), // MeV, Msun

  (CGS.c*CGS.c)/(CGS.G * CGS.Msun), // length, Msun
  PS_CUBE( CGS.c)/(CGS.G * CGS.Msun), // time, Msun
  PS_CUBE( (CGS.G * CGS.Msun)/(CGS.c*CGS.c) ), // number density, Msun^-3
  1.0 / CGS.Msun, // mass, Msun
  1.0 / (CGS.Msun * CGS.c*CGS.c), // energy, Msun
  PS_CUBE( CGS.G/(CGS.c*CGS.c) ) * PS_SQR( CGS.Msun/(CGS.c) ), // pressure, Msun^-2
  CGS.kb / (CGS.Msun * CGS.c*CGS.c), // temperature, Msun
};
//! Nuclear units
static UnitSystem Nuclear{
  1.0, // c
  CGS.G * CGS.MeV/(CGS.c*CGS.c*CGS.c*CGS.c)*1e13, // G, fm
  1.0, // kb
  CGS.Msun * (CGS.c*CGS.c) / CGS.MeV, // Msun, MeV
  1.0, // MeV

  1e13, // length, fm
  CGS.c * 1e13, // time, fm
  1e-39, // number density, fm^-3
  (CGS.c*CGS.c) / CGS.MeV, // mass, MeV
  1.0/CGS.MeV, // energy, MeV
  1e-39/CGS.MeV, // pressure, MeV/fm^3
  CGS.kb/CGS.MeV, // temperature, MeV
};

} // namespace

#undef PS_SQR
#undef PS_CUBE

#endif
