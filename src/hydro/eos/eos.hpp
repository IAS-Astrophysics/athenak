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

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class EquationOfState
//  \brief abstract base class for all EOS classes

class EquationOfState {
 public:
  EquationOfState(Hydro *phyd, std::unique_ptr<ParameterInput> &pin);
  virtual ~EquationOfState() = default;

  Hydro *pmy_hydro;

  // folowing pure virtual functions must be overridden in derived EOS classes
  virtual void ConservedToPrimitive(const int k, const int j, const int il,const  int iu, 
    AthenaArray<Real> &cons, AthenaArray<Real> &prim) = 0;

 protected:
  Real density_floor_, pressure_floor_;
};

//----------------------------------------------------------------------------------------
//! \class AdiabaticHydro
//  \brief derived EOS class for nonrelativistic adiabatic hydrodynamics

class AdiabaticHydro : public EquationOfState {
 public:
  AdiabaticHydro(Hydro *phyd, std::unique_ptr<ParameterInput> &pin);

  Real GetGamma() const {return gamma_;}

  // functions that implement methods appropriate to adiabatic hydrodynamics
  void ConservedToPrimitive(const int k, const int j, const int il,const  int iu, 
    AthenaArray<Real> &cons, AthenaArray<Real> &prim) override;

 private:
  Real gamma_;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalHydro
//  \brief derived EOS class for nonrelativistic isothermal hydrodynamics

class IsothermalHydro : public EquationOfState {
 public:
  IsothermalHydro(Hydro *phyd, std::unique_ptr<ParameterInput> &pin);

  // functions that implement methods appropriate to adiabatic hydrodynamics
  void ConservedToPrimitive(const int k, const int j, const int il,const  int iu,
    AthenaArray<Real> &cons, AthenaArray<Real> &prim) override;

 private:
  Real iso_cs_;
};

} // namespace hydro

#endif // HYDRO_EOS_EOS_HPP_
