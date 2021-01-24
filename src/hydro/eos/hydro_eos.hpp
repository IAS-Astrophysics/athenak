#ifndef HYDRO_EOS_EOS_HPP_
#define HYDRO_EOS_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.hpp
//  \brief Contains data and functions that implement conserved<->primitive
//  variable conversion for various EOS, e.g. adiabatic, isothermal, etc.

#include <cmath> 

#include "athena.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \struct EOSData
//  \brief container for variables associated with EOS

struct EOS_Data
{
  Real gamma;
  Real iso_cs;
  bool is_adiabatic;
  Real density_floor, pressure_floor;
  // sound speed function for adiabatic EOS 
  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed(Real p, Real d) const {return std::sqrt(gamma*p/d);}
};

//----------------------------------------------------------------------------------------
//! \class EquationOfState
//  \brief Abstract base class for Hydro EOS

class EquationOfState
{
 public:
  EquationOfState(MeshBlockPack *pp, ParameterInput *pin);
  ~EquationOfState() = default;

  MeshBlockPack* pmy_pack;
  EOS_Data eos_data;

  // pure virtual function to convert cons to prim, overwritten in each derived class
  virtual void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) = 0;

 private:
};

//----------------------------------------------------------------------------------------
//! \class AdibaticHydro
//  \brief Derived class for Hydro adiabatic EOS

class AdiabaticHydro : public EquationOfState
{
 public:
  AdiabaticHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalHydro
//  \brief Derived class for Hydro isothermal EOS

class IsothermalHydro : public EquationOfState
{ 
 public:
  IsothermalHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
};


#endif // HYDRO_EOS_EOS_HPP_
