#ifndef HYDRO_EOS_EOS_HPP_
#define HYDRO_EOS_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.hpp
//  \brief defines abstract base class EquationOfState, and various derived classes
//  Each derived class Contains data and functions that implement conserved<->primitive
//  variable conversion for that particular EOS, e.g. adiabatic, isothermal, etc.

#include <cmath> 

#include "athena.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class EquationOfState
//  \brief abstract base class for all EOS classes

class EquationOfState
{
 public:
  EquationOfState(Mesh *pm, ParameterInput *pin, int igid);
  virtual ~EquationOfState() = default;

  // data
  bool adiabatic_eos;

  // functions
  virtual Real GetGamma() {return 0.0;}       // only used in adiabatic EOS

  // folowing pure virtual functions must be overridden in derived EOS classes
  virtual void ConservedToPrimitive(AthenaArray4D<Real> &cons,AthenaArray4D<Real> &prim) = 0;
  virtual Real SoundSpeed(Real prim[5]);
  virtual Real SoundSpeed();

 protected:
  Mesh* pmesh_;
  int my_mbgid_;
  Real density_floor_, pressure_floor_;
};

//----------------------------------------------------------------------------------------
//! \class AdiabaticHydro
//  \brief derived EOS class for nonrelativistic adiabatic hydrodynamics

class AdiabaticHydro : public EquationOfState
{
 public:
  AdiabaticHydro(Mesh* pm, ParameterInput *pin, int igid);
  Real GetGamma() override {return gamma_;}

  // functions that implement methods appropriate to adiabatic hydrodynamics
  void ConservedToPrimitive(AthenaArray4D<Real> &cons,AthenaArray4D<Real> &prim) override;
  Real SoundSpeed(Real prim[5]) override {return std::sqrt(gamma_*prim[IPR]/prim[IDN]);}

 private:
  Real gamma_;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalHydro
//  \brief derived EOS class for nonrelativistic isothermal hydrodynamics

class IsothermalHydro : public EquationOfState
{
 public:
  IsothermalHydro(Mesh* pm, ParameterInput *pin, int igid);

  // functions that implement methods appropriate to isothermal hydrodynamics
  void ConservedToPrimitive(AthenaArray4D<Real> &cons,AthenaArray4D<Real> &prim) override;
  Real SoundSpeed() override {return iso_cs_;}

 private:
  Real iso_cs_;
};

} // namespace hydro

#endif // HYDRO_EOS_EOS_HPP_
