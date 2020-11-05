#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_
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

enum EOS_Type {adiabatic_hydro, isothermal_hydro};

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
//  \brief functions for EOS

class EquationOfState
{
 public:
  EquationOfState(Mesh *pm, ParameterInput *pin, int igid);
  ~EquationOfState() = default;

  EOS_Data eos_data;

  // wrapper function that calls different conversion routines
  void ConservedToPrimitive(const AthenaArray4D<Real> &cons, AthenaArray4D<Real> &prim);

  // cons to prim functions for different EOS
  void ConsToPrimAdiHydro(const AthenaArray4D<Real> &cons, AthenaArray4D<Real> &prim);
  void ConsToPrimIsoHydro(const AthenaArray4D<Real> &cons, AthenaArray4D<Real> &prim);
  void ConsToPrimAdiMHD(const AthenaArray4D<Real> &cons, AthenaArray4D<Real> &prim,
                        const FaceArray3D<Real> &bfc, AthenaArray4D<Real> &bcc);
  void ConsToPrimIsoMHD(const AthenaArray4D<Real> &cons, AthenaArray4D<Real> &prim,
                        const FaceArray3D<Real> &bfc, AthenaArray4D<Real> &bcc);

 private:
  Mesh* pmesh_;
  int my_mbgid_;
  EOS_Type eos_type_;    // enum that specifies EOS type
};

#endif // EOS_EOS_HPP_
