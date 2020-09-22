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

enum class EOSType {adiabatic_nr_hydro, isothermal_nr_hydro};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class EquationOfState
//  \brief functions for EOS

class EquationOfState
{
 public:
  EquationOfState(Mesh *pm, ParameterInput *pin, int igid);
  ~EquationOfState() = default;

  // getter functions
  Real GetGamma() {return gamma_;}
  Real GetIsoCs() {return iso_cs_;}
  bool IsAdiabatic() {return adiabatic_eos_;}

  // wrapper function that calls different conversion routines
  void ConservedToPrimitive(AthenaArray4D<Real> &cons,AthenaArray4D<Real> &prim);

  // cons to prim functions for different EOS
  void ConToPrimAdi(AthenaArray4D<Real> &cons,AthenaArray4D<Real> &prim);
  void ConToPrimIso(AthenaArray4D<Real> &cons,AthenaArray4D<Real> &prim);

  // sound speed function for adiabatic EOS 
  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed(Real p, Real d) {return std::sqrt(gamma_*p/d);}

 private:
  Mesh* pmesh_;
  int my_mbgid_;
  Real density_floor_, pressure_floor_;
  Real gamma_;
  Real iso_cs_;
  EOSType eos_type_;    // enum that specifies EOS type
  bool adiabatic_eos_;  // true if EOS is adiabatic
};

} // namespace hydro

#endif // HYDRO_EOS_EOS_HPP_
