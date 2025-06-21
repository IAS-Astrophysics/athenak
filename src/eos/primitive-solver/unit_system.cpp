//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file unit_system.cpp
//  \brief Defines functions for making new unit systems.
#include "unit_system.hpp"

#define PS_SQR(x) ((x)*(x))
#define PS_CUBE(x) ((x)*(x)*(x))

Primitive::UnitSystem Primitive::MakeCGS() {
  return UnitSystem{
    2.99792458e10, // c, cm/s
    6.67408e-8, // G, cm^3 g^-1 s^-2
    1.38064852e-16, // kb, erg K^-1
    1.98848e33, // Msun, g
    1.6021766208e-6, // MeV, erg

    1.0, // length, cm
    1.0, // time, s
    1.0, // number density cm^-3
    1.0, // volume in cm^3
    1.0, // mass, g
    1.0, // energy, erg
    1.0, // pressure, erg/cm^3
    1.0, // temperature, K
    1.0, // chemical potential, erg
  };
}

Primitive::UnitSystem Primitive::MakeGeometricKilometer() {
  return UnitSystem{
    1.0, // c
    1.0, // G
    1.0, // kb
    CGS.Msun * CGS.G/(CGS.c*CGS.c)*1e-5, // Msun, km
    CGS.MeV * CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5, // MeV, km

    1e-5, // length, km
    CGS.c * 1e-5, // time, km
    1e-39, // number density fm^-3
    1e-15, // volume in km^3
    CGS.G/(CGS.c*CGS.c)*1e-5, // mass, km
    CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5, // energy, km
    CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e10, // pressure, km^-2
    CGS.kb*CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5, // temperature, km
    1.0/CGS.MeV, // chemical potential, MeV
  };
}

Primitive::UnitSystem Primitive::MakeGeometricSolar() {
  // 1 cm in geometric units
  Real const cm = (CGS.c*CGS.c)/(CGS.G * CGS.Msun);
  return UnitSystem{
    1.0, // c
    1.0, // G
    1.0, // kb
    1.0, // Msun
    CGS.MeV / (CGS.c*CGS.c), // MeV, Msun

    cm, // length, Msun
    PS_CUBE( CGS.c)/(CGS.G * CGS.Msun), // time, Msun
    1.0e-39, // number density in fm^-3
    PS_CUBE( cm ), // volume in Msun^3
    1.0 / CGS.Msun, // mass, Msun
    1.0 / (CGS.Msun * CGS.c*CGS.c), // energy, Msun
    PS_CUBE( CGS.G/(CGS.c*CGS.c) ) * PS_SQR( CGS.Msun/(CGS.c) ), // pressure, Msun^-2
//     CGS.kb / (CGS.Msun * CGS.c*CGS.c), // temperature, Msun
    CGS.kb/CGS.MeV, // temperature, MeV
    1.0/CGS.MeV, // chemical potential, MeV
  };
}

Primitive::UnitSystem Primitive::MakeNuclear() {
  return UnitSystem{
    1.0, // c
    CGS.G * CGS.MeV/(CGS.c*CGS.c*CGS.c*CGS.c)*1e13, // G, fm
    1.0, // kb
    CGS.Msun * (CGS.c*CGS.c) / CGS.MeV, // Msun, MeV
    1.0, // MeV

    1e13, // length, fm
    CGS.c * 1e13, // time, fm
    1e-39, // number density, fm^-3
    1e39, // volume in fm^3
    (CGS.c*CGS.c) / CGS.MeV, // mass, MeV
    1.0/CGS.MeV, // energy, MeV
    1e-39/CGS.MeV, // pressure, MeV/fm^3
    CGS.kb/CGS.MeV, // temperature, MeV
    1.0/CGS.MeV, // chemical potential, MeV
  };
}

Primitive::UnitSystem Primitive::MakeMKS() {
  return UnitSystem{
    CGS.c/1e2,        // c
    CGS.G/1e3,        // G
    CGS.kb/1e7,       // kb
    CGS.Msun/1e3,     // Msun
    CGS.MeV/1e7,      // MeV

    1e-2,             // 1 cm in m
    1.0,              // 1 s in s
    1e6,              // 1 cm^{-3} in m^{-3}
    1e-6,             // 1 cm^3 in m^3
    1e-3,             // 1 g in kg
    1e-7,             // 1 erg in J
    0.1,              // 1 dyne/cm in Pa
    1.0,              // 1 K in K
    1e-7,             // 1 erg in J
  };
}

// nm-g-sec, but energy in MeV
Primitive::UnitSystem Primitive::MakeNGS() {
  Real const cm = 1e7; // nm
  Real const s = 1;
  Real const g = 1;
  Real const K = CGS.kb/CGS.MeV;  // 1 K in MeV

  Real const erg = 1/CGS.MeV;
  Real const dyn = g/PS_SQR(s);

  Real const c = CGS.c*cm/s;
  Real const G = CGS.G*PS_CUBE(cm)/g/PS_SQR(s);
  Real const kb = CGS.kb*erg/K;
  Real const Msun = CGS.Msun*g;
  Real const MeV = 1.0;

  return UnitSystem{
    c, G, kb, Msun, MeV, cm, s, 1/PS_CUBE(cm), PS_CUBE(cm), g, erg, dyn, K, erg,
  };
}

#undef PS_SQR
#undef PS_CUBE