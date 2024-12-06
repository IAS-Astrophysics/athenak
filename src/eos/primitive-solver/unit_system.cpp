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
    1.0, // density, g cm^-3
    1.0, // mass, g
    1.0, // energy, erg
    1.0, // pressure, erg/cm^3
    1.0  // temperature, K
  };
}

Primitive::UnitSystem Primitive::MakeGeometricKilometer() {
  return UnitSystem{
    1.0, // c
    1.0, // G
    1.0, // kb
    static_cast<Real>(CGS.Msun * CGS.G/(CGS.c*CGS.c)*1e-5), // Msun, km
    static_cast<Real>(CGS.MeV * CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5), // MeV, km

    1e-5, // length, km
    static_cast<Real>(CGS.c * 1e-5), // time, km
    1e15, // number density, km^-3
    static_cast<Real>(CGS.G/(CGS.c*CGS.c)*1e-5), // mass, km
    static_cast<Real>(CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5), // energy, km
    static_cast<Real>(CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e10), // pressure, km^-2
    static_cast<Real>(CGS.kb*CGS.G/(CGS.c*CGS.c*CGS.c*CGS.c)*1e-5), // temperature, km
  };
}

Primitive::UnitSystem Primitive::MakeGeometricSolar() {
  return UnitSystem{
    1.0, // c
    1.0, // G
    1.0, // kb
    1.0, // Msun
    static_cast<Real>(CGS.MeV / (CGS.c*CGS.c)), // MeV, Msun

    static_cast<Real>((CGS.c*CGS.c)/(CGS.G * CGS.Msun)), // length, Msun
    static_cast<Real>(PS_CUBE( CGS.c)/(CGS.G * CGS.Msun)), // time, Msun
    static_cast<Real>(PS_CUBE( (CGS.G * CGS.Msun)/(CGS.c*CGS.c) )), // number density, Msun^-3
    static_cast<Real>(1.0 / CGS.Msun), // mass, Msun
    static_cast<Real>(1.0 / (CGS.Msun * CGS.c*CGS.c)), // energy, Msun
    static_cast<Real>(PS_CUBE( CGS.G/(CGS.c*CGS.c) ) * PS_SQR( CGS.Msun/(CGS.c) )), // pressure, Msun^-2
//     CGS.kb / (CGS.Msun * CGS.c*CGS.c), // temperature, Msun
    static_cast<Real>(CGS.kb/CGS.MeV), // temperature, MeV
  };
}

Primitive::UnitSystem Primitive::MakeNuclear() {
  return UnitSystem{
    1.0, // c
    static_cast<Real>(CGS.G * CGS.MeV/(CGS.c*CGS.c*CGS.c*CGS.c)*1e13), // G, fm
    1.0, // kb
    static_cast<Real>(CGS.Msun * (CGS.c*CGS.c) / CGS.MeV), // Msun, MeV
    1.0, // MeV

    1e13, // length, fm
    static_cast<Real>(CGS.c * 1e13), // time, fm
    1e-39, // number density, fm^-3
    static_cast<Real>((CGS.c*CGS.c) / CGS.MeV), // mass, MeV
    static_cast<Real>(1.0/CGS.MeV), // energy, MeV
    static_cast<Real>(1e-39/CGS.MeV), // pressure, MeV/fm^3
    static_cast<Real>(CGS.kb/CGS.MeV), // temperature, MeV
  };
}

Primitive::UnitSystem Primitive::MakeMKS() {
  return UnitSystem{
    static_cast<Real>(CGS.c/1e2),        // c
    static_cast<Real>(CGS.G/1e3),        // G
    static_cast<Real>(CGS.kb/1e7),       // kb
    static_cast<Real>(CGS.Msun/1e3),     // Msun
    static_cast<Real>(CGS.MeV/1e7),      // MeV

    1e-2,             // 1 cm in m
    1.0,              // 1 s in s
    1e6,              // 1 cm^-3 in m^-3
    1e-3,             // 1 g in kg
    1e-7,             // 1 erg in J
    0.1,              // 1 dyne/cm in Pa
    1.0,              // 1 K in K
  };
}

#undef PS_SQR
#undef PS_CUBE
